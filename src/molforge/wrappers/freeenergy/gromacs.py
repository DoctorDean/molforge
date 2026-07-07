"""Parsing for ``gmx_MMPBSA`` output.

``gmx_MMPBSA`` is based on ``MMPBSA.py`` and writes the same
``FINAL_RESULTS_MMPBSA.dat`` structure, so this reuses the shared
section/row helpers in :mod:`molforge.wrappers.freeenergy._common`. Two
things differ from the Amber output and are handled here:

- The delta section is headed ``Delta (Complex - Receptor - Ligand):``
  and its rows are Δ-prefixed — ``ΔVDWAALS``, ``ΔEEL``, ``ΔEGB``/``ΔEPB``,
  ``ΔESURF``/(``ΔENPOLAR`` + ``ΔEDISPER``), ``ΔTOTAL``.
- Each row has five numeric columns (``Average SD(Prop.) SD SEM(Prop.)
  SEM``) rather than three. ``ΔG`` is the first column and the standard
  error of the mean is the last, matching how the Amber parser reads its
  three columns.

As with the Amber parser, the entropy section is not read, so
``delta_g`` is the enthalpic binding total and ``components.entropy`` is
``None``.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from molforge.cache import get_default_cache
from molforge.core import Provenance
from molforge.core import metadata_keys as mk
from molforge.freeenergy import (
    Decomposition,
    FreeEnergyResult,
    MMGBSAEngine,
    MMGBSAEngineNotInstalledError,
)
from molforge.wrappers.freeenergy import _common

if TYPE_CHECKING:
    from os import PathLike

    from numpy.typing import NDArray

    from molforge.core import Protein
    from molforge.md import Trajectory
    from molforge.wrappers.freeenergy._common import Selection

# gmx_MMPBSA delta-section term labels (Δ-prefixed), per solvent model.
_SOLVENT_MODELS = {
    "gb": {
        "header": "GENERALIZED BORN:",
        "vdw": "ΔVDWAALS",
        "eel": "ΔEEL",
        "polar": "ΔEGB",
        "nonpolar": ("ΔESURF",),
        "total": "ΔTOTAL",
        "method": "MM/GBSA",
    },
    "pb": {
        "header": "POISSON BOLTZMANN:",
        "vdw": "ΔVDWAALS",
        "eel": "ΔEEL",
        "polar": "ΔEPB",
        "nonpolar": ("ΔENPOLAR", "ΔEDISPER", "ΔECAVITY"),
        "total": "ΔTOTAL",
        "method": "MM/PBSA",
    },
}


def parse_gmx_mmpbsa_dat(text: str, *, solvent_model: str = "gb") -> FreeEnergyResult:
    """Parse ``gmx_MMPBSA`` output text into a :class:`FreeEnergyResult`.

    Args:
        text: Full contents of a ``gmx_MMPBSA`` ``FINAL_RESULTS_MMPBSA.dat``.
        solvent_model: ``"gb"`` (MM/GBSA, default) or ``"pb"`` (MM/PBSA).

    Returns:
        A result whose ``delta_g`` / ``uncertainty`` come from the
        ``ΔTOTAL`` row (Average and the final SEM column) and whose
        ``components`` hold the per-term breakdown; ``entropy`` is
        ``None``. Frame count and the ΔTOTAL sample standard deviation
        are recorded in ``metadata``.

    Raises:
        ValueError: If ``solvent_model`` is unknown, the requested
            section is absent, or a required term row is missing.
    """
    model = solvent_model.lower()
    spec = _SOLVENT_MODELS.get(model)
    if spec is None:
        raise ValueError(f"solvent_model must be 'gb' or 'pb', got {solvent_model!r}")

    block = _common.differences_block(_common.section(text, str(spec["header"])))

    vdw = _common.row_values(block, str(spec["vdw"]))[0]
    electrostatic = _common.row_values(block, str(spec["eel"]))[0]
    polar = _common.row_values(block, str(spec["polar"]))[0]

    nonpolar_rows = [
        vals
        for term in spec["nonpolar"]
        if (vals := _common.optional_row_values(block, term)) is not None
    ]
    if not nonpolar_rows:
        raise ValueError("no nonpolar solvation term found in delta block")
    nonpolar = sum(vals[0] for vals in nonpolar_rows)

    total = _common.row_values(block, str(spec["total"]))

    # gmx delta rows are (Average, SD(Prop.), SD, SEM(Prop.), SEM); the
    # sample SD is the third-from-last column.
    metadata: dict[str, object] = {
        "solvent_model": model,
        "delta_total_std_dev": total[-3] if len(total) >= 3 else total[-1],
    }
    frames = re.search(r"using\s+(\d+)\s+complex frames", text)
    if frames is not None:
        metadata["n_frames"] = int(frames.group(1))

    return _common.build_free_energy_result(
        vdw=vdw,
        electrostatic=electrostatic,
        polar=polar,
        nonpolar=nonpolar,
        delta_g=total[0],
        uncertainty=total[-1],
        method=str(spec["method"]),
        metadata=metadata,
    )


def parse_gmx_mmpbsa_decomp(text: str, *, section: str = "delta") -> Decomposition:
    """Parse a ``gmx_MMPBSA`` per-residue decomposition.

    gmx_MMPBSA writes the same ``FINAL_DECOMP_MMPBSA.dat`` structure as
    ``MMPBSA.py`` (it reuses that writer), so this reads the requested
    species' "Total Energy Decomposition" block just like
    :func:`~molforge.wrappers.freeenergy.parse_mmpbsa_decomp`. The default
    ``"delta"`` section is the per-residue binding contribution.

    gmx's delta rows carry an extra ``Location`` column (the residue's spot
    in the receptor/ligand topology); it is dropped, and the residue keeps
    its complex-numbering ``resname resnum`` label.

    Args:
        text: Contents of ``FINAL_DECOMP_MMPBSA.dat``.
        section: Which species block to read — ``"delta"`` (default),
            ``"complex"``, ``"receptor"``, or ``"ligand"``.

    Returns:
        A :class:`~molforge.freeenergy.Decomposition` over the residues in
        that block, in report order.

    Raises:
        ValueError: If ``section`` is unknown or the block is absent.
    """
    return _common.parse_decomp(text, section=section)


def _ndx_block(name: str, atoms_1based: NDArray[np.int64], per_line: int) -> str:
    """Format a GROMACS ``.ndx`` group block from 1-based atom numbers."""
    lines = [f"[ {name} ]"]
    for i in range(0, atoms_1based.size, per_line):
        lines.append(" ".join(str(int(a)) for a in atoms_1based[i : i + per_line]))
    return "\n".join(lines) + "\n"


def selection_to_ndx_group(
    topology: Protein, selection: Selection, name: str, *, per_line: int = 15
) -> str:
    """Render a molforge selection as a GROMACS ``.ndx`` index group.

    gmx_MMPBSA identifies the receptor and ligand by index-group number
    (``-cg <receptor> <ligand>``), so a selection is resolved against the
    topology and written as a named group of 1-based atom numbers — the
    numbering GROMACS uses, which matches the topology's atom order (as
    produced by the GROMACS MD wrapper).

    Args:
        topology: Structure the selection is resolved against.
        selection: Field filters or a boolean atom mask.
        name: Group name, written as ``[ name ]``.
        per_line: Atom numbers per line (GROMACS wraps long groups).

    Returns:
        The ``.ndx`` group block (header line plus wrapped atom numbers).

    Raises:
        ValueError: If the selection matches no atoms.
    """
    mask = _common.resolve_selection_mask(topology, selection)
    atoms = np.nonzero(mask)[0] + 1  # GROMACS index files are 1-based
    if atoms.size == 0:
        raise ValueError("selection matches no atoms")
    return _ndx_block(name, atoms, per_line)


def _group_signature(atoms_1based: NDArray[np.int64]) -> dict[str, object]:
    """Compact, JSON-clean signature of an index group for provenance.

    The full atom list can be thousands of entries, so the cache key
    carries the atom count plus a short hash of the (ordered) indices
    rather than the list itself.
    """
    payload = ",".join(str(int(a)) for a in atoms_1based).encode()
    return {"natoms": int(atoms_1based.size), "sha1": hashlib.sha1(payload).hexdigest()[:16]}


class GromacsMMGBSA(MMGBSAEngine):
    """Endpoint free energy via ``gmx_MMPBSA`` (the GROMACS sibling).

    Post-processes a GROMACS MD trajectory: resolves the ``receptor`` /
    ``ligand`` selections to ``.ndx`` index groups, runs ``gmx_MMPBSA``,
    and parses its ``FINAL_RESULTS_MMPBSA.dat``. Like the Amber engine it
    *orchestrates* the tool rather than parameterizing a system: it needs
    a GROMACS structure (``.tpr``) and trajectory (``.xtc``/``.trr``) on
    disk — passed explicitly or carried by a trajectory produced by
    :class:`molforge.wrappers.md.GROMACS` (whose metadata records the run
    directory holding ``md.tpr``, ``md.xtc`` and ``topol.top``). A
    topology (``.top``) is used with ``-cp`` when available. Results are
    cached on the run's :class:`~molforge.core.Provenance`.

    Args:
        executable: ``gmx_MMPBSA`` binary name or path.
        igb: Generalized Born model index passed to MM/GBSA runs.
        verbose: If true, stream tool stdout/stderr instead of capturing.
    """

    name = "GromacsMMGBSA"

    def __init__(
        self, *, executable: str = "gmx_MMPBSA", igb: int = 5, verbose: bool = False
    ) -> None:
        self.executable = executable
        self.igb = igb
        self.verbose = verbose

    def run(  # type: ignore[override]  # concrete kwargs refine the ABC's **kwargs
        self,
        trajectory: Trajectory,
        *,
        receptor: Selection,
        ligand: Selection,
        solvent_model: str = "gb",
        structure: str | PathLike[str] | None = None,
        trajectory_file: str | PathLike[str] | None = None,
        topology: str | PathLike[str] | None = None,
        start_frame: int = 1,
        end_frame: int | None = None,
        interval: int = 1,
        salt_conc: float = 0.0,
        **_kwargs: object,
    ) -> FreeEnergyResult:
        """Estimate ΔG_bind from ``trajectory`` with gmx_MMPBSA.

        An identical repeat returns the cached result without invoking the
        tool.

        Args:
            trajectory: The ensemble to average over; its topology defines
                the atom numbering and its metadata may locate the GROMACS
                inputs.
            receptor: Selection identifying the receptor atoms.
            ligand: Selection identifying the ligand atoms.
            solvent_model: ``"gb"`` (MM/GBSA, default) or ``"pb"``.
            structure: Explicit GROMACS structure (``.tpr``); overrides
                metadata.
            trajectory_file: Explicit trajectory (``.xtc``/``.trr``).
            topology: Explicit GROMACS topology (``.top``); passed with
                ``-cp`` when present.
            start_frame: First frame to analyze (1-based).
            end_frame: Last frame to analyze; defaults to the length.
            interval: Stride between analyzed frames.
            salt_conc: Salt concentration (mol/L).

        Returns:
            A :class:`FreeEnergyResult` from the tool's final block, with
            provenance attached.

        Raises:
            ValueError: If a selection is empty, or the structure /
                trajectory can't be located.
            MMGBSAEngineNotInstalledError: If ``gmx_MMPBSA`` isn't
                installed.
        """
        top = trajectory.topology
        rec_mask = _common.resolve_selection_mask(top, receptor)
        lig_mask = _common.resolve_selection_mask(top, ligand)
        if not rec_mask.any():
            raise ValueError("receptor selection matches no atoms")
        if not lig_mask.any():
            raise ValueError("ligand selection matches no atoms")
        rec_atoms = np.nonzero(rec_mask)[0] + 1
        lig_atoms = np.nonzero(lig_mask)[0] + 1

        structure_path, traj_path, top_path = self._resolve_inputs(
            trajectory, structure, trajectory_file, topology
        )
        end = end_frame if end_frame is not None else trajectory.n_frames

        provenance = Provenance.from_engine(
            engine="GromacsMMGBSA.run",
            parameters={
                "solvent_model": solvent_model.lower(),
                "receptor_group": _group_signature(rec_atoms),
                "ligand_group": _group_signature(lig_atoms),
                "start_frame": start_frame,
                "end_frame": end,
                "interval": interval,
                "salt_conc": salt_conc,
                "igb": self.igb,
            },
            inputs={
                "structure": str(structure_path),
                "trajectory_file": str(traj_path),
                "topology": None if top_path is None else str(top_path),
            },
            parent=_common.as_provenance(trajectory.metadata.get(mk.PROVENANCE)),
        )

        cache = get_default_cache()
        cached: FreeEnergyResult | None = cache.get(provenance, "free_energy_result")
        if cached is not None:
            return cached

        self._require_tool()

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            (run_dir / "mmpbsa.in").write_text(
                _common.build_mmpbsa_input(
                    solvent_model=solvent_model,
                    start_frame=start_frame,
                    end_frame=end,
                    interval=interval,
                    salt_conc=salt_conc,
                    igb=self.igb,
                )
            )
            # Receptor is group 0, ligand group 1 in this minimal index.
            (run_dir / "index.ndx").write_text(
                _ndx_block("receptor", rec_atoms, 15) + _ndx_block("ligand", lig_atoms, 15)
            )
            results_text = self._invoke(run_dir, structure_path, traj_path, top_path)

        result = parse_gmx_mmpbsa_dat(results_text, solvent_model=solvent_model)
        result.provenance = provenance
        result.metadata.update(
            {
                "engine": self.name,
                "receptor_natoms": int(rec_atoms.size),
                "ligand_natoms": int(lig_atoms.size),
                mk.PROVENANCE: provenance,
            }
        )
        cache.put(provenance, result, "free_energy_result")
        return result

    # -- input resolution ---------------------------------------------

    def _resolve_inputs(
        self,
        trajectory: Trajectory,
        structure: str | PathLike[str] | None,
        trajectory_file: str | PathLike[str] | None,
        topology: str | PathLike[str] | None,
    ) -> tuple[Path, Path, Path | None]:
        meta = trajectory.metadata
        structure_path = (
            Path(structure)
            if structure is not None
            else _common.input_from_metadata(meta, "structure", "md.tpr")
        )
        traj_path = (
            Path(trajectory_file)
            if trajectory_file is not None
            else _common.input_from_metadata(meta, "trajectory_file", "md.xtc")
        )
        top_path = (
            Path(topology)
            if topology is not None
            else _common.input_from_metadata(meta, "topology", "topol.top")
        )
        if structure_path is None:
            raise ValueError(
                "no GROMACS structure (.tpr) available. Pass structure=..., or use a "
                "trajectory produced by molforge.wrappers.md.GROMACS (which records one)."
            )
        if traj_path is None:
            raise ValueError(
                "no trajectory file available. Pass trajectory_file=..., or use a "
                "trajectory produced by molforge.wrappers.md.GROMACS."
            )
        if not structure_path.is_file():
            raise ValueError(f"GROMACS structure not found: {structure_path}")
        if not traj_path.is_file():
            raise ValueError(f"trajectory file not found: {traj_path}")
        if top_path is not None and not top_path.is_file():
            top_path = None  # optional -cp; ignore a stale path
        return structure_path, traj_path, top_path

    # -- tool invocation ----------------------------------------------

    def _require_tool(self) -> None:
        if shutil.which(self.executable) is None:
            raise MMGBSAEngineNotInstalledError(
                f"executable {self.executable!r} was not found on PATH.\n"
                "Install gmx_MMPBSA (e.g. `conda install -c conda-forge gmx_mmpbsa`, "
                "or see https://valdes-tresanco-ms.github.io/gmx_MMPBSA/), or pass "
                "executable=... to the constructor."
            )

    def _invoke(
        self, run_dir: Path, structure: Path, traj: Path, top: Path | None
    ) -> str:
        """Run gmx_MMPBSA and return the results text."""
        results = "FINAL_RESULTS_MMPBSA.dat"
        cmd = [
            self.executable,
            "-O",
            "-i", "mmpbsa.in",
            "-cs", str(structure),
            "-ci", "index.ndx",
            "-cg", "0", "1",
            "-ct", str(traj),
            "-o", results,
            "-nogui",
        ]
        if top is not None:
            cmd += ["-cp", str(top)]
        self._run_subprocess(cmd, cwd=run_dir, step="gmx_MMPBSA")

        out = run_dir / results
        if not out.is_file():
            raise RuntimeError(f"gmx_MMPBSA did not produce {results} in {run_dir}")
        return out.read_text()

    def _run_subprocess(self, cmd: list[str], *, cwd: Path, step: str) -> None:
        """Single choke point for tool invocation (mockable in tests)."""
        try:
            subprocess.run(
                cmd,
                check=True,
                cwd=str(cwd),
                capture_output=not self.verbose,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr or "(stderr not captured)"
            raise RuntimeError(
                f"gmx_MMPBSA step `{step}` failed (exit {e.returncode}).\n"
                f"command: {' '.join(cmd)}\nstderr:\n{stderr}"
            ) from e


__all__ = [
    "GromacsMMGBSA",
    "parse_gmx_mmpbsa_dat",
    "parse_gmx_mmpbsa_decomp",
    "selection_to_ndx_group",
]
