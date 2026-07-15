"""AutoDock Vina wrapper.

[AutoDock Vina](https://vina.scripps.edu/) is the de-facto workhorse for
small-molecule docking. It's fast, deterministic-ish (with a seed),
freely licensed, and supported by a mature Python interface via the
[``vina``](https://pypi.org/project/vina/) PyPI package.

This wrapper handles the parts of Vina that are most tedious in raw
form: receptor / ligand preparation (writing PDBQT files), search-box
specification, parsing the multi-pose PDBQT output, and converting
results back into molforge's :class:`DockingResult` / :class:`Pose`
representation.

What we **don't** do here:
  - Hydrogen / charge assignment on the receptor. Use
    AutoDockTools or ``meeko`` upstream; molforge takes a prepared
    receptor as input.
  - Ligand 3D-conformer generation from SMILES. Use RDKit upstream and
    pass an SDF/PDBQT here. (A convenience for SMILES input via the
    ``[ml]`` / ``[docking]`` extras may land in a future iteration.)

Memory / runtime: Vina is CPU-bound and scales with the number of
rotatable bonds, exhaustiveness, and box volume. For a typical
20 Å box and exhaustiveness=8, a single docking call takes ~10-30
seconds on modern hardware.

Installation: ``pip install 'molforge[docking]'`` plus
``pip install vina meeko`` for the engine itself. The PyPI ``vina``
package bundles the binary.
"""

from __future__ import annotations

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


def _provenance_ref(ref: Protein | str | PathLike[str]) -> str:
    """Return a JSON-native string identifier for a docking input.

    Provenance ``inputs`` must be JSON-serialisable, but Vina takes
    either a :class:`Protein` or a path. For paths we serialise the
    string form; for a :class:`Protein` we use its name or a generic
    marker — the actual provenance ancestry (when present) is carried
    through ``provenance_parent``, not duplicated here.
    """
    if isinstance(ref, Protein):
        return ref.name or "<Protein>"
    return str(ref)


class Vina(DockingEngine):
    """Wrapper around AutoDock Vina.

    Args:
        scoring: Scoring function name. ``"vina"`` (default) or
            ``"vinardo"``.
        seed: Random seed for reproducibility. ``None`` = engine default.
        cpu: Number of CPU threads to use. ``0`` = all available.
        verbosity: Vina's internal verbosity (0 = silent, 1 = some,
            2 = verbose).

    Example:
        >>> from molforge.wrappers.docking import Vina
        >>> import molforge as mf
        >>>
        >>> receptor = mf.load("receptor_prepared.pdbqt")
        >>> result = Vina().dock(
        ...     receptor=receptor,
        ...     ligand="ligand.pdbqt",
        ...     center=(10.0, 5.0, -2.0),
        ...     box_size=(20.0, 20.0, 20.0),
        ...     exhaustiveness=8,
        ...     n_poses=9,
        ... )
        >>> best = result.best
        >>> best.score    # Vina affinity in kcal/mol
        -8.4
    """

    name = "Vina"
    # Vina shells out to a CPU binary, so dock_many can run many docks across
    # OS processes rather than one at a time.
    parallelism = "process"

    def __init__(
        self,
        *,
        scoring: str = "vina",
        seed: int | None = None,
        cpu: int = 0,
        verbosity: int = 0,
    ) -> None:
        self.scoring = scoring
        self.seed = seed
        self.cpu = cpu
        self.verbosity = verbosity
        # The Vina handle is lazily created on each dock() call rather
        # than once on construction — Vina objects bind to a specific
        # receptor, and users typically dock multiple ligands against
        # a single receptor (or vice versa), so binding lazily keeps
        # the interface simple.

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
        energy_range: float = 3.0,
        min_rmsd: float = 1.0,
        **_kwargs: object,
    ) -> DockingResult:
        """Dock ``ligand`` against ``receptor`` using Vina.

        Args:
            receptor: A :class:`Protein`, a path to a PDB / mmCIF file
                (prepared automatically with meeko), or a path to an
                already-prepared ``.pdbqt`` file (used as-is).
            ligand: A :class:`Protein` (with ligand atoms) or a path to
                a ``.sdf`` / ``.mol`` / ``.mol2`` / ``.pdb`` / ``.pdbqt``
                file. For SMILES input, use
                :func:`molforge.wrappers.docking.prepare_ligand` directly
                with ``from_smiles=True`` before calling ``dock``.
            center: ``(x, y, z)`` center of the search box in Å. This
                is the most important parameter — get it wrong and
                you'll dock to nothing.
            box_size: ``(dx, dy, dz)`` size of the search box in Å.
                20-25 Å per side covers a typical pocket.
            exhaustiveness: Sampling intensity. Vina default is 8;
                use 16 or 32 for more thorough searches. Linear in
                runtime.
            n_poses: Maximum number of poses to return.
            energy_range: Maximum energy difference (kcal/mol) between
                best and worst returned pose.
            min_rmsd: Minimum RMSD between distinct poses.
            **_kwargs: Reserved for future Vina options.

        Returns:
            A :class:`DockingResult` with poses sorted best (most negative
            affinity) first. Each :class:`Pose` has ``score`` in kcal/mol
            and ``rmsd_lb`` / ``rmsd_ub`` populated relative to the best pose.

        Raises:
            DockingEngineNotInstalledError: If ``vina`` is not installed.
        """
        # Build the Provenance upfront from inputs + constructor params,
        # before touching the (expensive) engine. If an identical dock
        # has been cached, return it without spawning Vina at all.
        provenance = self._build_provenance(
            receptor,
            ligand,
            center=center,
            box_size=box_size,
            exhaustiveness=exhaustiveness,
            n_poses=n_poses,
            energy_range=energy_range,
            min_rmsd=min_rmsd,
        )
        cache = get_default_cache()
        cached: DockingResult | None = cache.get(provenance, "docking_result")
        if cached is not None:
            return cached

        vina_handle = self._make_vina_handle()

        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            receptor_pdbqt = self._materialize_receptor(receptor, tmpdir)
            ligand_pdbqt = self._materialize_ligand(ligand, tmpdir)

            vina_handle.set_receptor(str(receptor_pdbqt))
            vina_handle.set_ligand_from_file(str(ligand_pdbqt))
            vina_handle.compute_vina_maps(
                center=list(center),
                box_size=list(box_size),
            )
            vina_handle.dock(
                exhaustiveness=exhaustiveness,
                n_poses=n_poses,
                min_rmsd=min_rmsd,
                max_evals=0,
            )
            out_path = tmpdir / "poses.pdbqt"
            vina_handle.write_poses(
                str(out_path),
                n_poses=n_poses,
                energy_range=energy_range,
                overwrite=True,
            )
            text = out_path.read_text(encoding="utf-8", errors="replace")

        result = self._parse_poses_pdbqt(
            text,
            receptor=receptor if isinstance(receptor, Protein) else None,
            run_metadata={
                "center": center,
                "box_size": box_size,
                "exhaustiveness": exhaustiveness,
                "scoring": self.scoring,
                "seed": self.seed,
            },
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
        energy_range: float,
        min_rmsd: float,
    ) -> Provenance:
        """Construct the Provenance for a :meth:`dock` call.

        Factored out so :meth:`dock` can build it upfront for the cache
        lookup and thread the same instance into the parser. Pure
        function of inputs + constructor parameters — no engine calls.
        """
        parent = receptor.metadata.get(mk.PROVENANCE) if isinstance(receptor, Protein) else None
        return Provenance.from_engine(
            engine="Vina",
            parameters={
                "center": list(center),
                "box_size": list(box_size),
                "exhaustiveness": exhaustiveness,
                "n_poses": n_poses,
                "energy_range": energy_range,
                "min_rmsd": min_rmsd,
                "scoring": self.scoring,
                "seed": self.seed,
                "cpu": self.cpu,
            },
            inputs={
                "receptor": _provenance_ref(receptor),
                "ligand": _provenance_ref(ligand),
            },
            parent=parent if isinstance(parent, Provenance) else None,
        )

    # ------------------------------------------------------------------
    # Lazy-import helpers (separated for testability)
    # ------------------------------------------------------------------
    def _make_vina_handle(self) -> Any:
        """Construct a vina.Vina instance, raising a clean error if missing."""
        try:
            from vina import Vina as _Vina
        except ImportError as e:
            raise DockingEngineNotInstalledError(
                "AutoDock Vina requires the `vina` PyPI package. Install with:\n"
                "    pip install vina meeko\n"
                "and ensure `pip install 'molforge[docking]'` for RDKit.\n"
                f"Underlying error: {e}"
            ) from e

        kwargs: dict[str, Any] = {
            "sf_name": self.scoring,
            "cpu": self.cpu,
            "verbosity": self.verbosity,
        }
        if self.seed is not None:
            kwargs["seed"] = self.seed
        return _Vina(**kwargs)

    def _materialize_receptor(
        self,
        receptor: Protein | str | PathLike[str],
        tmpdir: Path,
    ) -> Path:
        """Return a path to a PDBQT receptor file.

        - If receptor is already a path to a .pdbqt file, return as-is.
        - Otherwise, run meeko to prepare it (lazy import so users
          without meeko installed still get a clean error message).
        """
        from molforge.wrappers.docking.prep import is_pdbqt_path, prepare_receptor

        if is_pdbqt_path(receptor):
            return Path(receptor)  # type: ignore[arg-type]
        out = tmpdir / "receptor.pdbqt"
        return prepare_receptor(receptor, out)

    def _materialize_ligand(
        self,
        ligand: Protein | str | PathLike[str],
        tmpdir: Path,
    ) -> Path:
        """Return a path to a PDBQT ligand file.

        Accepts a path to an existing .pdbqt, any common chemistry
        file format (.sdf / .mol / .mol2 / .pdb) which is prepared via
        meeko, or a Protein wrapping ligand atoms (also prepped).
        SMILES is not auto-detected — use :func:`prepare_ligand`
        explicitly with ``from_smiles=True``.
        """
        from molforge.wrappers.docking.prep import is_pdbqt_path, prepare_ligand

        if is_pdbqt_path(ligand):
            return Path(ligand)  # type: ignore[arg-type]
        out = tmpdir / "ligand.pdbqt"
        # If `ligand` is a Protein, write it to a PDB temp file first.
        if hasattr(ligand, "atom_array"):
            from molforge.io import write_pdb

            tmp_pdb = tmpdir / "ligand_input.pdb"
            write_pdb(ligand, tmp_pdb)  # type: ignore[arg-type]
            return prepare_ligand(tmp_pdb, out)
        return prepare_ligand(ligand, out)

    # ------------------------------------------------------------------
    # Output parsing (testable in isolation, no Vina needed)
    # ------------------------------------------------------------------
    def _parse_poses_pdbqt(
        self,
        text: str,
        *,
        receptor: Protein | None = None,
        run_metadata: dict[str, object] | None = None,
        provenance_parameters: dict[str, Any] | None = None,
        provenance_inputs: dict[str, Any] | None = None,
        provenance_parent: Provenance | None = None,
        provenance: Provenance | None = None,
    ) -> DockingResult:
        """Parse Vina's multi-pose PDBQT output into a DockingResult.

        Vina emits a file with this structure:

            MODEL 1
            REMARK VINA RESULT:     -8.4    0.000    0.000
            ... atom records ...
            ENDMDL
            MODEL 2
            REMARK VINA RESULT:     -7.9    1.234    2.345
            ... atom records ...
            ENDMDL
            ...

        We split on MODEL boundaries, pull the affinity (and RMSDs) out
        of the REMARK VINA RESULT line, and parse the atom block via
        :func:`molforge.io.read_pdb_string` (PDBQT atom lines are
        PDB-compatible for fields 1-66; the extra atom-type column at
        77-80 is ignored by our PDB parser).
        """
        from molforge.io.pdbqt import read_pdbqt_string

        poses: list[Pose] = []
        current_lines: list[str] = []
        current_score: float | None = None
        current_rmsd_lb: float | None = None
        current_rmsd_ub: float | None = None
        in_model = False
        rank = 0

        for raw_line in text.splitlines():
            if raw_line.startswith("MODEL"):
                in_model = True
                current_lines = []
                current_score = None
                current_rmsd_lb = None
                current_rmsd_ub = None
                continue
            if raw_line.startswith("ENDMDL"):
                if in_model and current_score is not None and current_lines:
                    # Parse via the canonical PDBQT reader so per-atom
                    # charges and AutoDock atom types make it onto the
                    # ligand Protein, not just coordinates.
                    pdbqt_text = "\n".join(current_lines) + "\nEND\n"
                    ligand = read_pdbqt_string(pdbqt_text)
                    poses.append(
                        Pose(
                            ligand=ligand,
                            score=float(current_score),
                            rank=rank,
                            rmsd_lb=current_rmsd_lb,
                            rmsd_ub=current_rmsd_ub,
                        )
                    )
                    rank += 1
                in_model = False
                continue
            if not in_model:
                continue
            # REMARK VINA RESULT: affinity rmsd_lb rmsd_ub
            if raw_line.startswith("REMARK VINA RESULT:"):
                parts = raw_line.split(":", 1)[1].split()
                if len(parts) >= 1:
                    current_score = float(parts[0])
                if len(parts) >= 2:
                    current_rmsd_lb = float(parts[1])
                if len(parts) >= 3:
                    current_rmsd_ub = float(parts[2])
                continue
            # ATOM / HETATM lines: PDBQT keeps charge + AutoDock atom
            # type in cols 67-80, which read_pdbqt_string captures.
            if raw_line.startswith(("ATOM", "HETATM")):
                current_lines.append(raw_line)

        # If file didn't have explicit MODEL records, treat the whole
        # thing as a single pose (Vina does this when n_poses=1 in
        # some versions).
        if not poses and "REMARK VINA RESULT" in text:
            atom_lines = [ln for ln in text.splitlines() if ln.startswith(("ATOM", "HETATM"))]
            score_line = next(
                ln for ln in text.splitlines() if ln.startswith("REMARK VINA RESULT:")
            )
            parts = score_line.split(":", 1)[1].split()
            score = float(parts[0])
            ligand = read_pdbqt_string("\n".join(atom_lines) + "\nEND\n")
            poses.append(Pose(ligand=ligand, score=score, rank=0))

        poses.sort(key=lambda p: p.score)
        for i, p in enumerate(poses):
            p.rank = i

        # Build the result-level metadata, layering Provenance on top
        # of the ad-hoc run_metadata keys for backwards compatibility.
        # A prebuilt ``provenance`` (passed by dock() so the cache key
        # and the stored result share one instance) wins; otherwise we
        # build from the parameters/inputs/parent kwargs for callers
        # (tests) that drive the parser directly.
        result_metadata: dict[str, object] = dict(run_metadata or {})
        if provenance is not None:
            result_metadata[mk.PROVENANCE] = provenance
        elif provenance_parameters is not None or provenance_inputs is not None:
            result_metadata[mk.PROVENANCE] = Provenance.from_engine(
                engine="Vina",
                parameters=provenance_parameters or {},
                inputs=provenance_inputs or {},
                parent=provenance_parent,
            )

        return DockingResult(
            poses=poses,
            receptor=receptor,
            engine="Vina",
            metadata=result_metadata,
        )
