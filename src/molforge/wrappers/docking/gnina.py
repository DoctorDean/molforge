"""gnina docking engine wrapper.

`gnina <https://github.com/gnina/gnina>`_ is a fork of smina (itself a
fork of AutoDock Vina) with integrated convolutional-neural-network
scoring. The headline feature: instead of (or alongside) Vina's
empirical scoring function, gnina rescores poses using a 3D CNN
trained on PDBbind, giving learned predictions of binding pose
quality (``CNNscore``, 0–1) and binding affinity (``CNNaffinity``,
in pK units).

Where Vina-the-engine produces a single Vina-affinity score per
pose, gnina produces three per pose: the Vina-style affinity
(retained as ``minimizedAffinity``), the CNN pose-quality score,
and the CNN affinity estimate.

The wrapper exposes the same :class:`molforge.docking.DockingEngine`
interface as :class:`Vina` and :class:`DiffDock`:

.. code-block:: python

    from molforge.wrappers.docking import Gnina

    result = Gnina().dock(
        receptor=receptor,
        ligand=ligand_sdf,
        center=(10.0, 5.0, -2.0),
        box_size=(20.0, 20.0, 20.0),
    )
    print(result.poses[0].score)
    # CNNscore by default — higher (closer to 1.0) is better.

Same downstream code as the other docking wrappers; users who want
to swap Vina ↔ Gnina change only the constructor.

gnina vs Vina: when to pick which
---------------------------------

Both engines search the same way (gnina inherits Vina's Monte-Carlo
search). They differ in scoring:

- **Vina** ranks poses by an empirical scoring function calibrated
  on PDBbind. Score is interpretable as a binding-energy estimate
  (kcal/mol, more negative = better). Reproducible across runs
  with a fixed seed.
- **Gnina** ranks poses by a CNN trained on PDBbind. The CNN
  combines pose geometry and ligand chemistry into a learned
  binding-quality estimate. Often more accurate than Vina's
  empirical score for redocking and cross-docking benchmarks,
  at the cost of higher per-call latency (the CNN forward pass)
  and a different score interpretation.

For routine docking where Vina is "good enough" and reproducibility
matters most, stick with :class:`Vina`. For pose ranking on hard
targets — flexible binding sites, non-standard ligand chemistry —
gnina's CNN scoring is the modern improvement.

Scoring mode (``cnn_scoring``)
------------------------------

gnina exposes four CNN-scoring modes via ``--cnn_scoring``:

  ``"none"``        Vina-only; gnina behaves like smina.
  ``"rescore"``     CNN rescores Vina-found poses. (gnina default.)
  ``"refinement"``  CNN-driven local optimisation of top Vina hits.
                    ~10× slower than rescore.
  ``"all"``         CNN at every search step. ~1000× slower; rarely
                    the right choice.

The molforge wrapper defaults to ``"rescore"`` — the gnina default
and what most users want. ``"refinement"`` is worth trying for
careful single-target work; ``"all"`` is research-grade only.
"""

from __future__ import annotations

import contextlib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from molforge.cache import get_default_cache
from molforge.core import Protein
from molforge.core import metadata_keys as mk
from molforge.core.provenance import Provenance
from molforge.docking import (
    DockingEngine,
    DockingEngineNotInstalledError,
    DockingResult,
    Pose,
)

if TYPE_CHECKING:
    from os import PathLike


# gnina's CNN modes — the wrapper validates upfront so a typo
# doesn't reach the binary and surface as a long subprocess error.
_KNOWN_CNN_MODES: frozenset[str] = frozenset({"none", "rescore", "refinement", "all"})


# gnina's per-pose ranking criteria, via --pose_sort_order. The
# wrapper exposes these so users can override the default CNNscore
# ordering with CNNaffinity or the Vina-style minimised energy.
_KNOWN_SORT_ORDERS: frozenset[str] = frozenset({"CNNscore", "CNNaffinity", "Energy"})


def _provenance_ref(ref: Protein | str | PathLike[str]) -> str:
    """Return a JSON-native string identifier for a docking input.

    Mirrors the helper in :mod:`molforge.wrappers.docking.vina` —
    Provenance.inputs must be JSON-serialisable so we serialise paths
    and Protein names to strings; the structural ancestry travels
    via ``provenance_parent``, not via this string.
    """
    if isinstance(ref, Protein):
        return ref.name or "<Protein>"
    return str(ref)


class Gnina(DockingEngine):
    """Wrapper around the gnina docking binary.

    Args:
        gnina_executable: Name or absolute path of the ``gnina``
            binary. Defaults to ``"gnina"`` (expected on ``$PATH``).
            Lazy resolution: construction never touches the
            filesystem.
        cnn_scoring: When and how the CNN scoring function is used.
            One of ``"none"``, ``"rescore"`` (default), ``"refinement"``,
            ``"all"``. See module docstring for the trade-offs.
        cnn: CNN model name (passed as ``--cnn``). ``None`` uses
            gnina's default. Examples: ``"crossdock_default2018"``,
            ``"dense_default2018"``, ``"general_default2018_ensemble"``.
        sort_order: How to rank the returned poses. One of
            ``"CNNscore"`` (default), ``"CNNaffinity"``, ``"Energy"``.
            The first pose in the returned ``DockingResult.poses``
            list is the best by this criterion.
        scoring: Empirical scoring function for the Vina-style search.
            ``"vina"`` (default) or ``"vinardo"``.
        seed: Random seed for reproducibility. ``None`` = engine
            default (non-reproducible).
        cpu: Number of CPU threads. ``0`` = let gnina decide.
        timeout: Subprocess timeout in seconds. Defaults to a
            generous 600s — gnina with CNN rescore can take a couple
            minutes on big systems; pass a larger timeout for
            ``cnn_scoring="refinement"`` or ``"all"``.
        verbose: When True, gnina's stdout/stderr streams to the
            console instead of being captured. Useful for debugging.
    """

    name = "Gnina"

    def __init__(
        self,
        *,
        gnina_executable: str = "gnina",
        cnn_scoring: str = "rescore",
        cnn: str | None = None,
        sort_order: str = "CNNscore",
        scoring: str = "vina",
        seed: int | None = None,
        cpu: int = 0,
        timeout: float = 600.0,
        verbose: bool = False,
    ) -> None:
        if cnn_scoring not in _KNOWN_CNN_MODES:
            raise ValueError(
                f"unknown cnn_scoring {cnn_scoring!r}; expected one of {sorted(_KNOWN_CNN_MODES)}"
            )
        if sort_order not in _KNOWN_SORT_ORDERS:
            raise ValueError(
                f"unknown sort_order {sort_order!r}; expected one of {sorted(_KNOWN_SORT_ORDERS)}"
            )
        if scoring not in {"vina", "vinardo"}:
            raise ValueError(f"unknown scoring {scoring!r}; expected 'vina' or 'vinardo'")
        if cpu < 0:
            raise ValueError(f"cpu must be >= 0, got {cpu}")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")

        self.gnina_executable = gnina_executable
        self.cnn_scoring = cnn_scoring
        self.cnn = cnn
        self.sort_order = sort_order
        self.scoring = scoring
        self.seed = seed
        self.cpu = cpu
        self.timeout = timeout
        self.verbose = verbose
        # Direction of Pose.score follows the configured sort_order: "Energy"
        # is a Vina energy (lower is better); the CNN scores ("CNNscore" /
        # "CNNaffinity") are higher-is-better. Read by DockingScorer.from_engine.
        self.score_direction = "lower_is_better" if sort_order == "Energy" else "higher_is_better"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def dock(  # type: ignore[override]  # engine-specific kwargs refine the **kwargs ABC contract
        self,
        receptor: Protein | str | PathLike[str],
        ligand: Protein | str | PathLike[str],
        *,
        center: tuple[float, float, float],
        box_size: tuple[float, float, float] = (20.0, 20.0, 20.0),
        exhaustiveness: int = 8,
        n_poses: int = 9,
        min_rmsd: float = 1.0,
        **_kwargs: object,
    ) -> DockingResult:
        """Dock ``ligand`` against ``receptor`` using gnina.

        Args:
            receptor: A :class:`Protein` or a path to a structure
                file (``.pdb`` / ``.pdbqt`` / ``.mol2``). gnina
                handles format detection internally; molforge writes
                a temp PDB when given a Protein.
            ligand: A :class:`Protein` (with ligand atoms) or a path
                to a ligand file (``.sdf`` / ``.mol`` / ``.mol2`` /
                ``.pdb`` / ``.pdbqt``). For SMILES input, prepare
                the ligand to SDF first via
                :func:`molforge.wrappers.docking.prepare_ligand`.
            center: ``(x, y, z)`` centre of the search box in Å.
            box_size: ``(x, y, z)`` extent of the search box in Å.
            exhaustiveness: gnina's ``--exhaustiveness`` (search
                effort; higher = slower + more thorough). Default
                8 (gnina default).
            n_poses: Maximum number of poses to return.
            min_rmsd: Minimum RMSD between poses to keep both
                (gnina's ``--min_rmsd_filter``).

        Returns:
            A :class:`DockingResult` whose ``poses`` are sorted
            best-first by the configured ``sort_order`` (CNNscore
            by default).

        Raises:
            DockingEngineNotInstalledError: If ``gnina`` isn't on
                ``$PATH``.
            RuntimeError: If gnina exits non-zero or fails to
                produce parseable output.
        """
        # Build Provenance upfront and consult the cache before
        # requiring the binary or spawning gnina — an identical dock
        # returns instantly, even on a machine without gnina installed.
        provenance = self._build_provenance(
            receptor,
            ligand,
            center=center,
            box_size=box_size,
            exhaustiveness=exhaustiveness,
            n_poses=n_poses,
            min_rmsd=min_rmsd,
        )
        cache = get_default_cache()
        cached: DockingResult | None = cache.get(provenance, "docking_result")
        if cached is not None:
            return cached

        gnina_path = self._require_gnina()

        with tempfile.TemporaryDirectory(prefix="molforge_gnina_") as tmp:
            tmp_dir = Path(tmp)
            receptor_path = self._materialise_receptor(receptor, tmp_dir)
            ligand_path = self._materialise_ligand(ligand, tmp_dir)
            out_sdf = tmp_dir / "out.sdf"

            cmd = self._build_command(
                gnina=gnina_path,
                receptor_path=receptor_path,
                ligand_path=ligand_path,
                out_sdf=out_sdf,
                center=center,
                box_size=box_size,
                exhaustiveness=exhaustiveness,
                n_poses=n_poses,
                min_rmsd=min_rmsd,
            )

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=not self.verbose,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as e:
                raise RuntimeError(
                    f"gnina timed out after {self.timeout}s. "
                    "For cnn_scoring='refinement' or 'all' pass a "
                    "larger timeout= to the constructor."
                ) from e

            if proc.returncode != 0:
                raise RuntimeError(
                    f"gnina exited with code {proc.returncode}.\n"
                    f"stderr: {(proc.stderr or '').strip()}"
                )

            if not out_sdf.is_file():
                raise RuntimeError(
                    f"gnina did not produce {out_sdf}. stdout: {(proc.stdout or '').strip()}"
                )

            sdf_text = out_sdf.read_text(encoding="utf-8")

        result = self._parse_sdf_output(
            sdf_text,
            receptor=receptor if isinstance(receptor, Protein) else None,
            provenance_parameters={
                "cnn_scoring": self.cnn_scoring,
                "cnn": self.cnn,
                "sort_order": self.sort_order,
                "scoring": self.scoring,
                "center": list(center),
                "box_size": list(box_size),
                "exhaustiveness": exhaustiveness,
                "n_poses": n_poses,
                "min_rmsd": min_rmsd,
                "seed": self.seed,
                "cpu": self.cpu,
                "gnina_executable": self.gnina_executable,
            },
            provenance_inputs={
                "receptor": _provenance_ref(receptor),
                "ligand": _provenance_ref(ligand),
            },
            provenance_parent=(
                receptor.metadata.get(mk.PROVENANCE)
                if isinstance(receptor, Protein)
                and isinstance(receptor.metadata.get(mk.PROVENANCE), Provenance)
                else None
            ),
            provenance=provenance,
        )
        cache.put(provenance, result, "docking_result")
        return result

    def _build_provenance(
        self,
        receptor: Protein | str | PathLike[str],
        ligand: Protein | str | PathLike[str],
        *,
        center: tuple[float, float, float],
        box_size: tuple[float, float, float],
        exhaustiveness: int,
        n_poses: int,
        min_rmsd: float,
    ) -> Provenance:
        """Construct the Provenance for a :meth:`dock` call.

        Pure function of inputs + constructor parameters, factored out
        so :meth:`dock` can build it upfront for the cache lookup and
        thread the same instance into the parser. Mirrors exactly the
        parameters the parser would otherwise build itself.
        """
        parent = receptor.metadata.get(mk.PROVENANCE) if isinstance(receptor, Protein) else None
        return Provenance.from_engine(
            engine="Gnina",
            parameters={
                "cnn_scoring": self.cnn_scoring,
                "cnn": self.cnn,
                "sort_order": self.sort_order,
                "scoring": self.scoring,
                "center": list(center),
                "box_size": list(box_size),
                "exhaustiveness": exhaustiveness,
                "n_poses": n_poses,
                "min_rmsd": min_rmsd,
                "seed": self.seed,
                "cpu": self.cpu,
                "gnina_executable": self.gnina_executable,
            },
            inputs={
                "receptor": _provenance_ref(receptor),
                "ligand": _provenance_ref(ligand),
            },
            parent=parent if isinstance(parent, Provenance) else None,
        )

    # ------------------------------------------------------------------
    # Resolution + I/O helpers
    # ------------------------------------------------------------------

    def _require_gnina(self) -> str:
        resolved = shutil.which(self.gnina_executable)
        if resolved is None:
            raise DockingEngineNotInstalledError(
                f"gnina executable {self.gnina_executable!r} was not "
                "found on PATH.\n"
                "gnina isn't pip-installable; install via your system "
                "package manager (`brew install gnina` on macOS), "
                "download a release binary from "
                "https://github.com/gnina/gnina/releases, or build "
                "from source.\n"
                "For docking without gnina, use "
                "molforge.wrappers.docking.Vina (the empirical-scoring "
                "engine gnina is based on)."
            )
        return resolved

    @staticmethod
    def _materialise_receptor(receptor: Protein | str | PathLike[str], tmp_dir: Path) -> Path:
        """Return a filesystem path to the receptor. Writes a temp
        PDB if given a Protein; otherwise reuses the existing path.

        gnina handles PDB / PDBQT / MOL2 itself via Open Babel — we
        don't need to pre-prepare to PDBQT the way Vina does.
        """
        if isinstance(receptor, Protein):
            from molforge.io import save

            path = tmp_dir / "receptor.pdb"
            save(receptor, path)
            return path
        return Path(receptor)

    @staticmethod
    def _materialise_ligand(ligand: Protein | str | PathLike[str], tmp_dir: Path) -> Path:
        """Return a filesystem path to the ligand. Writes a temp
        SDF if given a Protein with ligand atoms; otherwise reuses
        the existing path."""
        if isinstance(ligand, Protein):
            from molforge.io import save

            path = tmp_dir / "ligand.sdf"
            save(ligand, path)
            return path
        return Path(ligand)

    def _build_command(
        self,
        *,
        gnina: str,
        receptor_path: Path,
        ligand_path: Path,
        out_sdf: Path,
        center: tuple[float, float, float],
        box_size: tuple[float, float, float],
        exhaustiveness: int,
        n_poses: int,
        min_rmsd: float,
    ) -> list[str]:
        """Assemble the gnina command line.

        Centralised so tests can verify the right flags are passed
        without needing the binary to actually run.
        """
        cx, cy, cz = center
        sx, sy, sz = box_size
        cmd: list[str] = [
            gnina,
            "--receptor",
            str(receptor_path),
            "--ligand",
            str(ligand_path),
            "--out",
            str(out_sdf),
            "--center_x",
            f"{cx}",
            "--center_y",
            f"{cy}",
            "--center_z",
            f"{cz}",
            "--size_x",
            f"{sx}",
            "--size_y",
            f"{sy}",
            "--size_z",
            f"{sz}",
            "--exhaustiveness",
            str(exhaustiveness),
            "--num_modes",
            str(n_poses),
            "--min_rmsd_filter",
            str(min_rmsd),
            "--cnn_scoring",
            self.cnn_scoring,
            "--pose_sort_order",
            self.sort_order,
            "--scoring",
            self.scoring,
        ]
        if self.cnn is not None:
            cmd += ["--cnn", self.cnn]
        if self.seed is not None:
            cmd += ["--seed", str(self.seed)]
        if self.cpu > 0:
            cmd += ["--cpu", str(self.cpu)]
        return cmd

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    # gnina emits SDF with REMARK lines carrying the three scores per
    # MODEL/molecule. The format is documented in the gnina README and
    # confirmed against issue #294 in the gnina repo:
    #     > minimizedAffinity   Vina-style energy (kcal/mol)
    #     > CNNscore             learned pose-quality (0-1, higher better)
    #     > CNNaffinity          learned affinity (pK)
    #     > CNNvariance          ensemble-model uncertainty (optional)
    _REMARK_RE = re.compile(r"^>\s*<\s*(minimizedAffinity|CNNscore|CNNaffinity|CNNvariance)\s*>")

    def _parse_sdf_output(
        self,
        sdf_text: str,
        *,
        receptor: Protein | None,
        provenance_parameters: dict[str, Any],
        provenance_inputs: dict[str, Any],
        provenance_parent: Provenance | None,
        provenance: Provenance | None = None,
    ) -> DockingResult:
        """Parse gnina's SDF output into a :class:`DockingResult`.

        Pose ranking is whatever gnina's ``--pose_sort_order`` told
        it to be; molforge respects gnina's order (poses[0] is best).
        Each pose's ``score`` is the value of the configured sort
        criterion — so ``poses[0].score`` is the headline number a
        user typically wants to look at.
        """
        from molforge.io.sdf import read_sdf_string

        per_pose_scores = self._extract_remarks(sdf_text)
        ligands = read_sdf_string(sdf_text)

        if not ligands:
            raise RuntimeError(
                "gnina's SDF output contained no parseable molecules. "
                "Check the gnina log for the actual error."
            )
        if len(ligands) != len(per_pose_scores):
            # gnina should emit one REMARK block per molecule, but
            # the SDF reader may have dropped malformed entries.
            # Pad with empty score dicts so indexing stays safe.
            while len(per_pose_scores) < len(ligands):
                per_pose_scores.append({})

        # All poses from one dock() call share a single Provenance.
        # Same pattern as Vina / DiffDock / ProteinMPNN. A prebuilt
        # ``provenance`` (passed by dock() so the cache key and the
        # stored result share one instance) wins; direct callers that
        # pass only parameters/inputs/parent get one built here.
        shared_prov = (
            provenance
            if provenance is not None
            else Provenance.from_engine(
                engine="Gnina",
                parameters=provenance_parameters,
                inputs=provenance_inputs,
                parent=provenance_parent,
            )
        )

        poses: list[Pose] = []
        # Pick the score-by-sort-order so poses[i].score is the
        # number the user would expect given the chosen ranking.
        score_key = self.sort_order  # "CNNscore" / "CNNaffinity" / "Energy"
        for i, (ligand, scores) in enumerate(zip(ligands, per_pose_scores, strict=False)):
            # "Energy" sort uses minimizedAffinity in the SDF — gnina
            # writes the empirical energy as minimizedAffinity, so
            # bridge the two names here.
            if score_key == "Energy":
                primary = scores.get("minimizedAffinity")
            else:
                primary = scores.get(score_key)

            poses.append(
                Pose(
                    ligand=ligand,
                    # primary may be None when gnina's output is
                    # missing the key (cnn_scoring="none" omits the
                    # CNN keys; if user sorts by CNNscore in that
                    # mode they get None scores — documented as an
                    # invalid combination but we don't crash).
                    score=float(primary) if primary is not None else 0.0,
                    rank=i,
                    metadata={
                        "engine": "Gnina",
                        "vina_affinity": scores.get("minimizedAffinity"),
                        "cnn_score": scores.get("CNNscore"),
                        "cnn_affinity": scores.get("CNNaffinity"),
                        "cnn_variance": scores.get("CNNvariance"),
                        "sort_order": self.sort_order,
                    },
                )
            )

        return DockingResult(
            poses=poses,
            receptor=receptor,
            engine="Gnina",
            metadata={
                mk.PROVENANCE: shared_prov,
                "cnn_scoring": self.cnn_scoring,
                "cnn": self.cnn,
                "sort_order": self.sort_order,
            },
        )

    @staticmethod
    def _extract_remarks(sdf_text: str) -> list[dict[str, float]]:
        """Pull the gnina REMARK lines out of an SDF text blob.

        Returns one dict per molecule; each dict has keys
        ``minimizedAffinity`` / ``CNNscore`` / ``CNNaffinity`` /
        ``CNNvariance`` (any subset of these depending on
        ``cnn_scoring``).

        SDF molecules are delimited by ``$$$$``. Inside each molecule
        gnina writes the score block as SDF data fields, e.g.

            > <minimizedAffinity>
            -8.4

            > <CNNscore>
            0.83

        which is the standard SDF tag format, not actual ``REMARK``
        lines (those are PDB-style). The method name says "remarks"
        for the user-facing concept; the parser handles the SDF tag
        format that gnina actually emits.
        """
        molecules: list[dict[str, float]] = []
        current: dict[str, float] = {}
        current_key: str | None = None
        for raw_line in sdf_text.splitlines():
            line = raw_line.rstrip()
            if line == "$$$$":
                molecules.append(current)
                current = {}
                current_key = None
                continue
            m = Gnina._REMARK_RE.match(line)
            if m:
                current_key = m.group(1)
                continue
            if current_key is not None:
                stripped = line.strip()
                if not stripped:
                    # Blank line ends the data field per SDF spec.
                    current_key = None
                    continue
                # Malformed value: skip rather than crash. gnina
                # is well-behaved here so this is a defensive
                # fallback for partial/truncated outputs.
                with contextlib.suppress(ValueError):
                    current[current_key] = float(stripped)
                current_key = None
        # gnina SDFs end with $$$$, but a defensive tail-emit lets
        # us handle truncated output gracefully too.
        if current:
            molecules.append(current)
        return molecules
