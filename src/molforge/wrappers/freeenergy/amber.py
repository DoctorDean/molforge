"""Parsing for Amber ``MMPBSA.py`` output.

This module reads the plain-text ``FINAL_RESULTS_MMPBSA.dat`` that
``MMPBSA.py`` writes and turns its final averaged block into a
:class:`molforge.freeenergy.FreeEnergyResult`. The engine wrapper that
builds inputs and invokes the tool is layered on top of this parser.

The file contains a ``GENERALIZED BORN:`` section and a ``POISSON
BOLTZMANN:`` section; within each, the block headed
``Differences (Complex - Receptor - Ligand):`` holds the binding terms.
That block is the one parsed here — the per-endpoint Complex/Receptor/
Ligand blocks above it are ignored. Term mapping:

===============  ==================  ==================
component        GB (``EGB``)        PB (``EPB``)
===============  ==================  ==================
vdw              ``VDWAALS``         ``VDWAALS``
electrostatic    ``EEL``             ``EEL``
polar solv.      ``EGB``             ``EPB``
nonpolar solv.   ``ESURF``           ``ENPOLAR`` + ``EDISPER``
===============  ==================  ==================

``DELTA TOTAL`` gives ΔG (its Average) and the reported uncertainty (its
Std. Err. of Mean). Entropy is reported in a separate section and is not
parsed yet, so ``delta_g`` here is the enthalpic total and
``components.entropy`` is ``None``.
"""

from __future__ import annotations

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
    FreeEnergyComponents,
    FreeEnergyResult,
    MMGBSAEngine,
    MMGBSAEngineNotInstalledError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from os import PathLike

    from numpy.typing import NDArray

    from molforge.core import Protein
    from molforge.md import Trajectory

    Selection = Mapping[str, object] | NDArray[np.bool_]

# One numeric field: optional sign, digits, optional fraction/exponent.
_NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"

# The two top-level solvent sections, used to bound each one.
_SECTION_HEADERS = ("GENERALIZED BORN:", "POISSON BOLTZMANN:")

_SOLVENT_MODELS = {
    "gb": {
        "header": "GENERALIZED BORN:",
        "polar": "EGB",
        "nonpolar": ("ESURF",),
        "method": "MM/GBSA",
    },
    "pb": {
        "header": "POISSON BOLTZMANN:",
        "polar": "EPB",
        "nonpolar": ("ENPOLAR", "EDISPER", "ECAVITY"),
        "method": "MM/PBSA",
    },
}


def _section(text: str, header: str) -> str:
    """The slice of ``text`` from ``header`` to the next section/EOF."""
    start = text.find(header)
    if start == -1:
        raise ValueError(f"section {header!r} not found in MMPBSA output")
    rest = text[start + len(header) :]
    cut = len(rest)
    for other in _SECTION_HEADERS:
        if other == header:
            continue
        idx = rest.find(other)
        if idx != -1:
            cut = min(cut, idx)
    return rest[:cut]


def _differences_block(section: str) -> str:
    """The ``Differences/Delta (Complex - Receptor - Ligand)`` block."""
    marker = re.search(r"(?im)^.*Complex - Receptor - Ligand.*$", section)
    if marker is None:
        raise ValueError("no 'Complex - Receptor - Ligand' block in section")
    return section[marker.end() :]


def _row(block: str, label: str) -> tuple[float, float, float]:
    """Return ``(average, std_dev, std_err)`` for a labelled row.

    The label is matched at the start of a line, so ``EEL`` does not
    match ``1-4 EEL`` and ``DELTA TOTAL`` does not match ``DELTA G gas``.
    """
    pattern = rf"(?m)^\s*{re.escape(label)}\s+({_NUM})\s+({_NUM})\s+({_NUM})\s*$"
    match = re.search(pattern, block)
    if match is None:
        raise ValueError(f"row {label!r} not found in differences block")
    return float(match.group(1)), float(match.group(2)), float(match.group(3))


def _optional_row(block: str, label: str) -> tuple[float, float, float] | None:
    try:
        return _row(block, label)
    except ValueError:
        return None


def parse_mmpbsa_dat(text: str, *, solvent_model: str = "gb") -> FreeEnergyResult:
    """Parse ``MMPBSA.py`` output text into a :class:`FreeEnergyResult`.

    Args:
        text: Full contents of a ``FINAL_RESULTS_MMPBSA.dat`` file.
        solvent_model: ``"gb"`` to read the Generalized Born section
            (MM/GBSA, default) or ``"pb"`` for Poisson–Boltzmann
            (MM/PBSA).

    Returns:
        A result whose ``delta_g`` and ``uncertainty`` come from the
        ``DELTA TOTAL`` row (Average and Std. Err. of Mean) and whose
        ``components`` hold the per-term breakdown. ``entropy`` is
        ``None`` — the entropy section is not parsed here — so
        ``delta_g`` is the enthalpic binding total. Frame count and the
        ``DELTA TOTAL`` standard deviation are recorded in ``metadata``.

    Raises:
        ValueError: If ``solvent_model`` is unknown, the requested
            section is absent, or a required term row is missing.
    """
    model = solvent_model.lower()
    spec = _SOLVENT_MODELS.get(model)
    if spec is None:
        raise ValueError(f"solvent_model must be 'gb' or 'pb', got {solvent_model!r}")

    block = _differences_block(_section(text, str(spec["header"])))

    vdw = _row(block, "VDWAALS")[0]
    electrostatic = _row(block, "EEL")[0]
    polar = _row(block, str(spec["polar"]))[0]

    nonpolar_rows = [
        row for term in spec["nonpolar"] if (row := _optional_row(block, term)) is not None
    ]
    if not nonpolar_rows:
        raise ValueError("no nonpolar solvation term found in differences block")
    nonpolar = sum(row[0] for row in nonpolar_rows)

    total_avg, total_sd, total_sem = _row(block, "DELTA TOTAL")

    metadata: dict[str, object] = {"solvent_model": model, "delta_total_std_dev": total_sd}
    frames = re.search(r"using\s+(\d+)\s+complex frames", text)
    if frames is not None:
        metadata["n_frames"] = int(frames.group(1))

    return FreeEnergyResult(
        delta_g=total_avg,
        uncertainty=total_sem,
        method=str(spec["method"]),
        components=FreeEnergyComponents(
            vdw=vdw,
            electrostatic=electrostatic,
            polar_solvation=polar,
            nonpolar_solvation=nonpolar,
            entropy=None,
        ),
        metadata=metadata,
    )


__all__ = [
    "AmberMMGBSA",
    "build_mmpbsa_input",
    "parse_mmpbsa_dat",
    "selection_to_amber_mask",
]


# ---------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------


def _resolve_mask(topology: Protein, selection: Selection) -> NDArray[np.bool_]:
    """Resolve a selection to a boolean atom mask over the topology."""
    from collections.abc import Mapping

    arr = topology.atom_array
    if isinstance(selection, Mapping):
        mask = arr.where(**selection)
    else:
        mask = np.asarray(selection, dtype=bool)
        if mask.shape != (len(arr),):
            raise ValueError(
                f"boolean selection has shape {mask.shape}, expected ({len(arr)},)"
            )
    return np.asarray(mask, dtype=bool)


def _collapse_ranges(numbers: Sequence[int]) -> str:
    """Collapse sorted 1-based indices to Amber range syntax (``1-3,5``)."""
    parts: list[str] = []
    start = prev = numbers[0]
    for n in numbers[1:]:
        if n == prev + 1:
            prev = n
            continue
        parts.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = n
    parts.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(parts)


def selection_to_amber_mask(topology: Protein, selection: Selection) -> str:
    """Convert a molforge selection to an Amber residue mask.

    The selection (a field-filter mapping like ``{"entity_type":
    "ligand"}``, forwarded to :meth:`~molforge.core.AtomArray.where`, or a
    boolean atom mask) is resolved against ``topology`` and expressed as a
    residue-number mask over the topology's sequential 1-based residue
    numbering — e.g. ``":1-120"`` or ``":121"``. This is the numbering
    Amber's ``ambmask`` uses, and the form MM/PB(GB)SA expects for
    splitting a complex into receptor and ligand.

    Args:
        topology: Structure the selection is resolved against (the
            complex).
        selection: Field filters or a boolean atom mask.

    Returns:
        An Amber residue mask string beginning with ``":"``.

    Raises:
        ValueError: If the selection matches no atoms, or splits a
            residue (endpoint masks must cover whole residues).
    """
    arr = topology.atom_array
    mask = _resolve_mask(topology, selection)
    if not mask.any():
        raise ValueError("selection matches no atoms")

    residues: list[int] = []
    for position, sl in enumerate(arr.iter_residue_slices(), start=1):
        in_residue = mask[sl]
        if bool(in_residue.all()):
            residues.append(position)
        elif bool(in_residue.any()):
            raise ValueError(
                f"selection splits residue at position {position}; "
                "MM/PB(GB)SA masks must cover whole residues"
            )
    return ":" + _collapse_ranges(residues)


def build_mmpbsa_input(
    *,
    solvent_model: str = "gb",
    start_frame: int = 1,
    end_frame: int,
    interval: int = 1,
    salt_conc: float = 0.0,
    igb: int = 5,
    verbose: int = 1,
) -> str:
    """Build the ``mmpbsa.in`` namelist text for an MM/PB(GB)SA run.

    Args:
        solvent_model: ``"gb"`` (writes a ``&gb`` namelist, MM/GBSA) or
            ``"pb"`` (writes a ``&pb`` namelist, MM/PBSA).
        start_frame: First trajectory frame to analyze (1-based).
        end_frame: Last trajectory frame to analyze (inclusive).
        interval: Stride between analyzed frames.
        salt_conc: Salt concentration in mol/L (``saltcon`` for GB,
            ``istrng`` for PB).
        igb: Generalized Born model index (GB only; 5 = OBC-II, a common
            default).
        verbose: MMPBSA.py ``verbose`` level.

    Returns:
        The input-file text, ready to write to ``mmpbsa.in``.

    Raises:
        ValueError: On an unknown ``solvent_model`` or an invalid frame
            range / interval.
    """
    model = solvent_model.lower()
    if model not in ("gb", "pb"):
        raise ValueError(f"solvent_model must be 'gb' or 'pb', got {solvent_model!r}")
    if start_frame < 1:
        raise ValueError(f"start_frame must be >= 1, got {start_frame}")
    if end_frame < start_frame:
        raise ValueError(f"end_frame ({end_frame}) must be >= start_frame ({start_frame})")
    if interval < 1:
        raise ValueError(f"interval must be >= 1, got {interval}")

    title = "molforge MM/GBSA input" if model == "gb" else "molforge MM/PBSA input"
    lines = [
        title,
        "&general",
        f"   startframe={start_frame}, endframe={end_frame}, "
        f"interval={interval}, verbose={verbose},",
        "/",
    ]
    if model == "gb":
        lines += ["&gb", f"   igb={igb}, saltcon={salt_conc:g},", "/"]
    else:
        lines += ["&pb", f"   istrng={salt_conc:g},", "/"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------


class AmberMMGBSA(MMGBSAEngine):
    """Endpoint free energy via Amber's ``MMPBSA.py``.

    Post-processes an MD trajectory: splits the complex topology into
    complex/receptor/ligand with ``ante-MMPBSA.py`` (using masks derived
    from the ``receptor`` / ``ligand`` selections), runs ``MMPBSA.py``,
    and parses its ``FINAL_RESULTS_MMPBSA.dat``.

    This engine *orchestrates* the tools; it does not parameterize a
    system. It needs an Amber topology (``prmtop``) and a matching
    trajectory file on disk — either passed explicitly or carried by a
    trajectory produced by :class:`molforge.wrappers.md.AMBER` (whose
    metadata records the run directory holding ``system.prmtop`` and
    ``prod.nc``). Anything else raises a clear error rather than trying
    to build a topology.

    Args:
        mmpbsa_executable: ``MMPBSA.py`` binary name or path.
        antemmpbsa_executable: ``ante-MMPBSA.py`` binary name or path.
        igb: Generalized Born model index passed to MM/GBSA runs.
        strip_mask: Amber mask of atoms stripped from the complex before
            splitting (solvent/ions), or ``None`` to strip nothing.
        verbose: If true, stream tool stdout/stderr instead of capturing.
    """

    name = "AmberMMGBSA"

    def __init__(
        self,
        *,
        mmpbsa_executable: str = "MMPBSA.py",
        antemmpbsa_executable: str = "ante-MMPBSA.py",
        igb: int = 5,
        strip_mask: str | None = ":WAT,HOH,Na+,Cl-,K+",
        verbose: bool = False,
    ) -> None:
        self.mmpbsa_executable = mmpbsa_executable
        self.antemmpbsa_executable = antemmpbsa_executable
        self.igb = igb
        self.strip_mask = strip_mask
        self.verbose = verbose

    def run(  # type: ignore[override]  # concrete kwargs refine the ABC's **kwargs
        self,
        trajectory: Trajectory,
        *,
        receptor: Selection,
        ligand: Selection,
        solvent_model: str = "gb",
        prmtop: str | PathLike[str] | None = None,
        trajectory_file: str | PathLike[str] | None = None,
        start_frame: int = 1,
        end_frame: int | None = None,
        interval: int = 1,
        salt_conc: float = 0.0,
        **_kwargs: object,
    ) -> FreeEnergyResult:
        """Estimate ΔG_bind from ``trajectory`` with MM/GBSA or MM/PBSA.

        Results are cached on the run's :class:`~molforge.core.Provenance`;
        an identical repeat returns the cached result without invoking the
        tools.

        Args:
            trajectory: The ensemble to average over; its topology
                defines the complex and its metadata may locate the
                Amber inputs.
            receptor: Selection identifying the receptor atoms.
            ligand: Selection identifying the ligand atoms.
            solvent_model: ``"gb"`` (MM/GBSA, default) or ``"pb"``.
            prmtop: Explicit Amber complex topology; overrides metadata.
            trajectory_file: Explicit trajectory file; overrides metadata.
            start_frame: First frame to analyze (1-based).
            end_frame: Last frame to analyze; defaults to the trajectory
                length.
            interval: Stride between analyzed frames.
            salt_conc: Salt concentration (mol/L).

        Returns:
            A :class:`FreeEnergyResult` from the tool's final block, with
            provenance attached.

        Raises:
            ValueError: If a selection is empty/splits a residue, or the
                Amber topology / trajectory file can't be located.
            MMGBSAEngineNotInstalledError: If the tools aren't installed.
        """
        # Masks first: resolving them needs only the topology and fails
        # fast on a bad selection, and the strings go into provenance.
        receptor_mask = selection_to_amber_mask(trajectory.topology, receptor)
        ligand_mask = selection_to_amber_mask(trajectory.topology, ligand)

        prmtop_path, traj_path = self._resolve_amber_inputs(
            trajectory, prmtop, trajectory_file
        )
        end = end_frame if end_frame is not None else trajectory.n_frames

        provenance = Provenance.from_engine(
            engine="AmberMMGBSA.run",
            parameters={
                "solvent_model": solvent_model.lower(),
                "receptor_mask": receptor_mask,
                "ligand_mask": ligand_mask,
                "start_frame": start_frame,
                "end_frame": end,
                "interval": interval,
                "salt_conc": salt_conc,
                "igb": self.igb,
                "strip_mask": self.strip_mask,
            },
            inputs={"prmtop": str(prmtop_path), "trajectory_file": str(traj_path)},
            parent=_as_provenance(trajectory.metadata.get(mk.PROVENANCE)),
        )

        # An identical run is keyed by this Provenance; return it without
        # touching the tools if we've computed it before.
        cache = get_default_cache()
        cached: FreeEnergyResult | None = cache.get(provenance, "free_energy_result")
        if cached is not None:
            return cached

        self._require_tools()

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            (run_dir / "mmpbsa.in").write_text(
                build_mmpbsa_input(
                    solvent_model=solvent_model,
                    start_frame=start_frame,
                    end_frame=end,
                    interval=interval,
                    salt_conc=salt_conc,
                    igb=self.igb,
                )
            )
            results_text = self._invoke(
                run_dir, prmtop_path, traj_path, receptor_mask, ligand_mask
            )

        result = parse_mmpbsa_dat(results_text, solvent_model=solvent_model)
        result.provenance = provenance
        result.metadata.update(
            {
                "engine": self.name,
                "receptor_mask": receptor_mask,
                "ligand_mask": ligand_mask,
                mk.PROVENANCE: provenance,
            }
        )
        cache.put(provenance, result, "free_energy_result")
        return result

    # -- input resolution ---------------------------------------------

    def _resolve_amber_inputs(
        self,
        trajectory: Trajectory,
        prmtop: str | PathLike[str] | None,
        trajectory_file: str | PathLike[str] | None,
    ) -> tuple[Path, Path]:
        prmtop_path = Path(prmtop) if prmtop is not None else _topology_from_metadata(
            trajectory.metadata, "prmtop", "system.prmtop"
        )
        traj_path = (
            Path(trajectory_file)
            if trajectory_file is not None
            else _topology_from_metadata(trajectory.metadata, "trajectory_file", "prod.nc")
        )
        if prmtop_path is None:
            raise ValueError(
                "no Amber topology (prmtop) available. Pass prmtop=..., or use a "
                "trajectory produced by molforge.wrappers.md.AMBER (which records one)."
            )
        if traj_path is None:
            raise ValueError(
                "no trajectory file available. Pass trajectory_file=..., or use a "
                "trajectory produced by molforge.wrappers.md.AMBER."
            )
        if not prmtop_path.is_file():
            raise ValueError(f"Amber topology not found: {prmtop_path}")
        if not traj_path.is_file():
            raise ValueError(f"trajectory file not found: {traj_path}")
        return prmtop_path, traj_path

    # -- tool invocation ----------------------------------------------

    def _require_tools(self) -> None:
        for exe in (self.antemmpbsa_executable, self.mmpbsa_executable):
            if shutil.which(exe) is None:
                raise MMGBSAEngineNotInstalledError(
                    f"executable {exe!r} was not found on PATH.\n"
                    "Install AmberTools (e.g. `conda install -c conda-forge ambertools`, "
                    "or build from https://ambermd.org/AmberTools.php), or pass "
                    "mmpbsa_executable=/antemmpbsa_executable= to the constructor."
                )

    def _invoke(
        self,
        run_dir: Path,
        prmtop: Path,
        traj: Path,
        receptor_mask: str,
        ligand_mask: str,
    ) -> str:
        """Split the topology and run MMPBSA.py; return the results text."""
        complex_p, receptor_p, ligand_p = "complex.prmtop", "receptor.prmtop", "ligand.prmtop"
        ante = [
            self.antemmpbsa_executable,
            "-p", str(prmtop),
            "-c", complex_p,
            "-r", receptor_p,
            "-l", ligand_p,
            "-m", receptor_mask,
            "-n", ligand_mask,
        ]
        if self.strip_mask:
            ante += ["-s", self.strip_mask]
        self._run_subprocess(ante, cwd=run_dir, step="ante-MMPBSA")

        results = "FINAL_RESULTS_MMPBSA.dat"
        self._run_subprocess(
            [
                self.mmpbsa_executable,
                "-O",
                "-i", "mmpbsa.in",
                "-o", results,
                "-cp", complex_p,
                "-rp", receptor_p,
                "-lp", ligand_p,
                "-y", str(traj),
            ],
            cwd=run_dir,
            step="MMPBSA",
        )
        out = run_dir / results
        if not out.is_file():
            raise RuntimeError(f"MMPBSA.py did not produce {results} in {run_dir}")
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
                f"MMPBSA step `{step}` failed (exit {e.returncode}).\n"
                f"command: {' '.join(cmd)}\nstderr:\n{stderr}"
            ) from e


def _topology_from_metadata(
    metadata: Mapping[str, object], explicit_key: str, run_dir_name: str
) -> Path | None:
    """Locate an Amber input from trajectory metadata.

    Prefers an explicit metadata key; falls back to ``<run_dir>/<name>``
    when the trajectory records a ``run_dir`` (the AMBER wrapper does).
    """
    value = metadata.get(explicit_key)
    if isinstance(value, (str, Path)):
        return Path(value)
    run_dir = metadata.get("run_dir")
    if isinstance(run_dir, (str, Path)):
        candidate = Path(run_dir) / run_dir_name
        if candidate.is_file():
            return candidate
    return None


def _as_provenance(value: object) -> Provenance | None:
    return value if isinstance(value, Provenance) else None
