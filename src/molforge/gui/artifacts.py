"""What a step produced, written down so it can be looked at.

The runner keeps *descriptions* of outputs, never the objects: a record
that pinned them would hold every intermediate of every run alive for as
long as the server does. That is the right call for a long-lived process
and it leaves the editor with nothing to show but a string, which for a
structure is the least interesting thing about it.

So outputs go to disk instead. Each run gets a directory; each step's
result is written in the format it already has a writer for, and
summarised into a :class:`ResultView` the panel can render without
loading anything heavy::

    artifacts = RunArtifacts.for_run("a1b2c3")
    view = artifacts.store(0, folded_protein)
    view.kind            # "structure"
    view.structures[0]   # a file the API can serve as PDB text

Disk rather than memory is also what makes the results *inspectable*: the
directory is a real place a user can open, and a structure the editor
showed is a file they can keep. Runs are pruned oldest-first so a
dashboard left open for a month doesn't quietly fill a disk.

Nothing here is required for a run to succeed. A failure to write an
artifact is reported in the view and never propagated — a pipeline that
worked must not be reported as failed because its output couldn't be
previewed.
"""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "MAX_ENSEMBLE",
    "MAX_RUNS",
    "ResultView",
    "RunArtifacts",
    "StructureRef",
    "summarize",
]

#: Members of an ensemble written to disk. A diffusion run can return
#: hundreds; a viewer shows a handful, and the manifest records how to
#: reproduce the rest.
MAX_ENSEMBLE = 12

#: Rows kept for a tabular result. A screen of thousands is a job for
#: pandas, not a panel.
MAX_ROWS = 200

#: Run directories kept. The editor is a long-lived local process, and
#: structures are small but not free.
MAX_RUNS = 20


@dataclass
class StructureRef:
    """One structure file the viewer can ask for.

    Attributes:
        key: Stable identifier within the step, used in the URL.
        label: What to call it in the viewer's layer list.
        role: ``"receptor"``, ``"ligand"``, or ``"structure"`` — the
            viewer styles a protein as cartoon and a ligand as sticks,
            and guessing from atom counts gets it wrong for peptides.
        format: File format, always ``"pdb"`` today.
    """

    key: str
    label: str
    role: str = "structure"
    format: str = "pdb"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResultView:
    """A step's output, as the results panel needs it.

    Attributes:
        kind: Which renderer applies — ``"structure"``, ``"ensemble"``,
            ``"poses"``, ``"pockets"``, ``"designs"``, ``"alignment"``,
            ``"value"``, or ``"none"``.
        summary: One line, the same text the node footer shows.
        structures: Structures the viewer can load.
        columns: Column names for a tabular result.
        rows: Rows, already JSON-safe.
        truncated: Set when rows or structures were capped, so the panel
            can say so rather than quietly showing a prefix.
        detail: Extra facts worth printing (an alignment's depth, a
            docking engine's name).
        error: Why there is nothing to show, when something went wrong
            writing it. Never raised — a preview failure is not a run
            failure.
    """

    kind: str = "none"
    summary: str = ""
    structures: list[StructureRef] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    truncated: bool = False
    detail: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "summary": self.summary,
            "structures": [s.to_dict() for s in self.structures],
            "columns": list(self.columns),
            "rows": [list(r) for r in self.rows],
            "truncated": self.truncated,
            "detail": dict(self.detail),
            "error": self.error,
        }


class RunArtifacts:
    """A run's directory of written-down outputs."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    # ---------- locating ----------

    @staticmethod
    def base_dir() -> Path:
        """Where run directories live — under molforge's own cache dir.

        Shares ``$MOLFORGE_CACHE_DIR`` so that a user who has already
        pointed molforge's cache at a big disk doesn't have to discover a
        second setting for the same concern.
        """
        from molforge.cache import default_cache_dir

        return default_cache_dir() / "editor-runs"

    @classmethod
    def for_run(cls, run_id: str, *, base: Path | None = None) -> RunArtifacts:
        root = (cls.base_dir() if base is None else base) / run_id
        return cls(root)

    @classmethod
    def prune(cls, *, base: Path | None = None, keep: int = MAX_RUNS) -> int:
        """Delete all but the ``keep`` most recent run directories.

        Returns the number removed. Best-effort: a directory that can't be
        removed (open elsewhere, permissions) is skipped rather than
        turning housekeeping into an error.
        """
        root = cls.base_dir() if base is None else base
        if not root.is_dir():
            return 0
        runs = sorted(
            (d for d in root.iterdir() if d.is_dir()),
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        removed = 0
        for stale in runs[keep:]:
            try:
                shutil.rmtree(stale)
                removed += 1
            except OSError:
                continue
        return removed

    # ---------- writing ----------

    def step_dir(self, index: int) -> Path:
        return self.root / f"step-{index + 1}"

    def path(self, index: int, key: str) -> Path:
        """Where one structure of one step lives.

        ``key`` comes off a URL, so it is checked rather than trusted: a
        key containing a separator would otherwise address any file the
        server process can read.
        """
        if not key or "/" in key or "\\" in key or ".." in key:
            raise ValueError(f"unsafe artifact key {key!r}")
        return self.step_dir(index) / f"{key}.pdb"

    def store(self, index: int, output: Any, *, upstream: Any = None) -> ResultView:
        """Write a step's output and summarise it.

        Args:
            index: 0-based step position.
            output: Whatever the step returned.
            upstream: The previous step's output. Pockets are the reason
                this exists — a list of pockets has no structure of its
                own, and a pocket drawn without the protein it sits on is
                a sphere in empty space.

        Returns:
            The view, with ``error`` set if writing failed. Never raises.
        """
        try:
            view = summarize(output, upstream=upstream)
            if view.structures:
                self.step_dir(index).mkdir(parents=True, exist_ok=True)
            kept: list[StructureRef] = []
            for ref, structure in zip(view.structures, _structures(output, upstream), strict=False):
                try:
                    _write_pdb(structure, self.path(index, ref.key))
                except Exception as e:  # one unwritable structure
                    view.detail.setdefault("unwritten", []).append(f"{ref.label}: {e}")
                    continue
                kept.append(ref)
            view.structures = kept
            return view
        except Exception as e:  # previewing must not fail a run
            return ResultView(kind="none", error=f"couldn't prepare a preview: {e}")

    def read(self, index: int, key: str) -> str:
        """The PDB text of one stored structure."""
        return self.path(index, key).read_text()


# ---------- summarising ----------


def summarize(output: Any, *, upstream: Any = None) -> ResultView:
    """Describe an output for the panel, without writing anything.

    Dispatches on the result types molforge actually returns rather than
    on a registry, because the set is small, closed, and each one wants a
    different table.
    """
    from molforge.gui.runner import describe_output

    summary = describe_output(output)

    if output is None:
        return ResultView(kind="none")

    if _is_protein(output):
        return ResultView(
            kind="structure",
            summary=summary,
            structures=[StructureRef(key="structure", label=_name(output), role="receptor")],
            detail=_structure_detail(output),
        )

    if _is_docking_result(output):
        return _docking_view(output, summary)

    if isinstance(output, (list, tuple)) and output:
        first = output[0]
        if _is_protein(first):
            return _ensemble_view(output, summary)
        if _is_pocket(first):
            return _pockets_view(output, summary, upstream)
        if _is_designed_sequence(first):
            return _designs_view(output, summary)

    if _is_alignment(output):
        return _alignment_view(output, summary)

    if isinstance(output, (list, tuple)):
        return ResultView(kind="value", summary=summary or "empty")

    return ResultView(kind="value", summary=summary)


def _docking_view(result: Any, summary: str) -> ResultView:
    poses = list(getattr(result, "poses", []) or [])
    structures: list[StructureRef] = []
    if getattr(result, "receptor", None) is not None:
        structures.append(StructureRef(key="receptor", label="receptor", role="receptor"))
    for i, _pose in enumerate(poses[:MAX_ENSEMBLE]):
        structures.append(StructureRef(key=f"pose-{i + 1}", label=f"pose {i + 1}", role="ligand"))
    rows = [
        [
            getattr(p, "rank", i),
            _round(getattr(p, "score", None)),
            _round(getattr(p, "rmsd_lb", None)),
            _round(getattr(p, "rmsd_ub", None)),
        ]
        for i, p in enumerate(poses[:MAX_ROWS])
    ]
    return ResultView(
        kind="poses",
        summary=summary,
        structures=structures,
        # Score first after rank: it is the number the whole run was for.
        columns=["rank", "score", "rmsd lb", "rmsd ub"],
        rows=rows,
        truncated=len(poses) > MAX_ENSEMBLE,
        detail={"engine": getattr(result, "engine", ""), "poses": len(poses)},
    )


def _ensemble_view(members: Sequence[Any], summary: str) -> ResultView:
    shown = list(members)[:MAX_ENSEMBLE]
    return ResultView(
        kind="ensemble",
        summary=summary,
        structures=[
            StructureRef(key=f"model-{i + 1}", label=f"model {i + 1}", role="receptor")
            for i in range(len(shown))
        ],
        columns=["model", "residues", "atoms"],
        rows=[
            [i + 1, getattr(m, "n_residues", None), getattr(m, "n_atoms", None)]
            for i, m in enumerate(shown)
        ],
        truncated=len(members) > MAX_ENSEMBLE,
        detail={"models": len(members)},
    )


def _pockets_view(pockets: Sequence[Any], summary: str, upstream: Any) -> ResultView:
    # A pocket has no structure of its own. Shown on the protein it was
    # detected in — which is the previous step's output — it is a place;
    # shown alone it is three numbers.
    structures: list[StructureRef] = []
    if _is_protein(upstream):
        structures.append(StructureRef(key="protein", label="protein", role="receptor"))

    rows = []
    centers = []
    for i, pocket in enumerate(list(pockets)[:MAX_ROWS]):
        # No `or []` fallback here: Pocket.center is a numpy array, and
        # an array's truth value raises. Every real pocket would have
        # hit it.
        raw = getattr(pocket, "center", None)
        center = [] if raw is None else [_round(float(c)) for c in raw]
        centers.append(center)
        rows.append(
            [
                getattr(pocket, "rank", i),
                _round(getattr(pocket, "score", None)),
                _round(getattr(pocket, "druggability", None)),
                _round(getattr(pocket, "volume", None)),
                ", ".join(str(c) for c in center),
                len(getattr(pocket, "residues", []) or []),
            ]
        )
    return ResultView(
        kind="pockets",
        summary=summary,
        structures=structures,
        columns=["rank", "score", "druggability", "volume Å³", "centre", "lining residues"],
        rows=rows,
        detail={"pockets": len(pockets), "centers": centers},
    )


def _designs_view(designs: Sequence[Any], summary: str) -> ResultView:
    rows = []
    for i, design in enumerate(list(designs)[:MAX_ROWS]):
        sequence = str(getattr(design, "sequence", design))
        rows.append(
            [
                i + 1,
                _round(getattr(design, "score", None)),
                _round(getattr(design, "recovery", None)),
                len(sequence),
                sequence,
            ]
        )
    return ResultView(
        kind="designs",
        summary=summary,
        columns=["#", "score", "recovery", "length", "sequence"],
        rows=rows,
        truncated=len(designs) > MAX_ROWS,
        detail={"designs": len(designs)},
    )


def _alignment_view(alignment: Any, summary: str) -> ResultView:
    hits = list(getattr(alignment, "hits", []) or [])[:MAX_ROWS]
    rows = [
        [
            i + 1,
            getattr(h, "identifier", "") or getattr(h, "description", ""),
            getattr(h, "source_database", ""),
            getattr(h, "pairing_key", ""),
        ]
        for i, h in enumerate(hits)
    ]
    return ResultView(
        kind="alignment",
        summary=summary,
        columns=["#", "hit", "database", "pairing key"],
        rows=rows,
        truncated=len(getattr(alignment, "hits", []) or []) > MAX_ROWS,
        detail={
            "depth": getattr(alignment, "depth", None),
            "columns": getattr(alignment, "n_columns", None),
            "query": (getattr(alignment, "query_sequence", "") or "")[:80],
        },
    )


# ---------- the objects themselves ----------


def _structures(output: Any, upstream: Any) -> list[Any]:
    """The structures to write, in the same order as ``view.structures``."""
    if _is_protein(output):
        return [output]
    if _is_docking_result(output):
        out: list[Any] = []
        if getattr(output, "receptor", None) is not None:
            out.append(output.receptor)
        out.extend(p.ligand for p in list(getattr(output, "poses", []))[:MAX_ENSEMBLE])
        return out
    if isinstance(output, (list, tuple)) and output:
        if _is_protein(output[0]):
            return list(output)[:MAX_ENSEMBLE]
        if _is_pocket(output[0]) and _is_protein(upstream):
            return [upstream]
    return []


def _write_pdb(structure: Any, path: Path) -> None:
    from molforge.io import save

    path.parent.mkdir(parents=True, exist_ok=True)
    save(structure, path, format="pdb")


# ---------- predicates ----------
#
# Duck-typed rather than isinstance: importing molforge.docking to check a
# type would pull in its dependencies on a code path that runs for every
# finished step, including ones that produced nothing of the sort.


def _is_protein(obj: Any) -> bool:
    return hasattr(obj, "n_residues") and hasattr(obj, "atom_array")


def _is_docking_result(obj: Any) -> bool:
    return hasattr(obj, "poses") and hasattr(obj, "receptor")


def _is_pocket(obj: Any) -> bool:
    return hasattr(obj, "center") and hasattr(obj, "druggability")


def _is_designed_sequence(obj: Any) -> bool:
    return hasattr(obj, "sequence") and not hasattr(obj, "atom_array")


def _is_alignment(obj: Any) -> bool:
    return hasattr(obj, "depth") and hasattr(obj, "hits")


def _name(obj: Any) -> str:
    return str(getattr(obj, "name", "") or "structure")


def _structure_detail(protein: Any) -> dict[str, Any]:
    return {
        "residues": getattr(protein, "n_residues", None),
        "atoms": getattr(protein, "n_atoms", None),
        "chains": getattr(protein, "n_chains", None),
    }


def _round(value: Any) -> Any:
    """Numbers at a precision a table can show. Non-numbers pass through."""
    if value is None:
        return None
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return value
