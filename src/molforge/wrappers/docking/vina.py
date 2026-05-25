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

from molforge.core import Protein
from molforge.docking import (
    DockingEngine,
    DockingEngineNotInstalledError,
    DockingResult,
    Pose,
)

if TYPE_CHECKING:
    from os import PathLike


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

        return self._parse_poses_pdbqt(
            text,
            receptor=receptor if isinstance(receptor, Protein) else None,
            run_metadata={
                "center": center,
                "box_size": box_size,
                "exhaustiveness": exhaustiveness,
                "scoring": self.scoring,
                "seed": self.seed,
            },
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
        from molforge.io import read_pdb_string

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
                    pdb_text = "\n".join(current_lines) + "\nEND\n"
                    ligand = read_pdb_string(pdb_text)
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
            # ATOM / HETATM lines: take the first 66 cols (Vina-specific
            # cols 67-80 hold atom types & charges; our PDB parser ignores them).
            if raw_line.startswith(("ATOM", "HETATM")):
                current_lines.append(raw_line[:66])

        # If file didn't have explicit MODEL records, treat the whole
        # thing as a single pose (Vina does this when n_poses=1 in
        # some versions).
        if not poses and "REMARK VINA RESULT" in text:
            atom_lines = [ln[:66] for ln in text.splitlines() if ln.startswith(("ATOM", "HETATM"))]
            score_line = next(
                ln for ln in text.splitlines() if ln.startswith("REMARK VINA RESULT:")
            )
            parts = score_line.split(":", 1)[1].split()
            score = float(parts[0])
            ligand = read_pdb_string("\n".join(atom_lines) + "\nEND\n")
            poses.append(Pose(ligand=ligand, score=score, rank=0))

        poses.sort(key=lambda p: p.score)
        for i, p in enumerate(poses):
            p.rank = i

        return DockingResult(
            poses=poses,
            receptor=receptor,
            engine="Vina",
            metadata=run_metadata or {},
        )
