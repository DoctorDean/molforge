"""Shared helpers for the MM/PB(GB)SA engines.

Amber's ``MMPBSA.py`` and ``gmx_MMPBSA`` share an input-file format and a
results-file structure (gmx_MMPBSA is "based on MMPBSA.py"), so the
selection resolver, the ``mmpbsa.in`` builder, the output-section
extraction, and result assembly live here and are used by both
:mod:`molforge.wrappers.freeenergy.amber` and
:mod:`molforge.wrappers.freeenergy.gromacs`.

The two differ only in the delta section's row labels (``VDWAALS`` vs
``ΔVDWAALS``, ``DELTA TOTAL`` vs ``ΔTOTAL``) and column count (3 vs 5).
:func:`row_values` returns every number on a labelled row, so each parser
takes column ``0`` (the average / ΔG) and column ``-1`` (the standard
error of the mean) regardless of how many columns lie between.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from molforge.core import Provenance
from molforge.freeenergy import (
    Decomposition,
    FreeEnergyComponents,
    FreeEnergyResult,
    ResidueContribution,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from molforge.core import Protein

    Selection = Mapping[str, object] | NDArray[np.bool_]

# One numeric field: optional sign, digits, optional fraction/exponent.
NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"

# The two top-level solvent sections, used to bound each one.
SECTION_HEADERS = ("GENERALIZED BORN:", "POISSON BOLTZMANN:")


# ---------------------------------------------------------------------
# Selections
# ---------------------------------------------------------------------


def resolve_selection_mask(topology: Protein, selection: Selection) -> NDArray[np.bool_]:
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


# ---------------------------------------------------------------------
# Input file
# ---------------------------------------------------------------------


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

    Shared by the Amber and gmx_MMPBSA engines — both consume the same
    ``&general`` / ``&gb`` / ``&pb`` namelists.

    Args:
        solvent_model: ``"gb"`` (writes a ``&gb`` namelist, MM/GBSA) or
            ``"pb"`` (writes a ``&pb`` namelist, MM/PBSA).
        start_frame: First trajectory frame to analyze (1-based).
        end_frame: Last trajectory frame to analyze (inclusive).
        interval: Stride between analyzed frames.
        salt_conc: Salt concentration in mol/L (``saltcon`` for GB,
            ``istrng`` for PB).
        igb: Generalized Born model index (GB only; 5 = OBC-II).
        verbose: MMPBSA ``verbose`` level.

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
# Output file
# ---------------------------------------------------------------------


def section(text: str, header: str) -> str:
    """The slice of ``text`` from ``header`` to the next section/EOF."""
    start = text.find(header)
    if start == -1:
        raise ValueError(f"section {header!r} not found in MMPBSA output")
    rest = text[start + len(header) :]
    cut = len(rest)
    for other in SECTION_HEADERS:
        if other == header:
            continue
        idx = rest.find(other)
        if idx != -1:
            cut = min(cut, idx)
    return rest[:cut]


def differences_block(section_text: str) -> str:
    """The ``Differences/Delta (Complex - Receptor - Ligand)`` block."""
    marker = re.search(r"(?im)^.*Complex - Receptor - Ligand.*$", section_text)
    if marker is None:
        raise ValueError("no 'Complex - Receptor - Ligand' block in section")
    return section_text[marker.end() :]


def row_values(block: str, label: str) -> list[float]:
    """Every number on a labelled row.

    The label is matched at the start of a line, so ``EEL`` does not match
    ``1-4 EEL`` and ``ΔTOTAL`` does not match ``ΔGGAS``. Callers take
    column ``0`` (average / ΔG) and ``-1`` (SEM).
    """
    pattern = rf"(?m)^\s*{re.escape(label)}\s+({NUM}(?:\s+{NUM})*)\s*$"
    match = re.search(pattern, block)
    if match is None:
        raise ValueError(f"row {label!r} not found in differences block")
    return [float(x) for x in match.group(1).split()]


def optional_row_values(block: str, label: str) -> list[float] | None:
    try:
        return row_values(block, label)
    except ValueError:
        return None


# ---------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------


def build_free_energy_result(
    *,
    vdw: float,
    electrostatic: float,
    polar: float,
    nonpolar: float,
    delta_g: float,
    uncertainty: float,
    method: str,
    metadata: dict[str, object],
) -> FreeEnergyResult:
    """Assemble a :class:`FreeEnergyResult` from parsed delta terms.

    Entropy is left ``None`` — the entropy section is not parsed — so
    ``delta_g`` is the enthalpic binding total.
    """
    return FreeEnergyResult(
        delta_g=delta_g,
        uncertainty=uncertainty,
        method=method,
        components=FreeEnergyComponents(
            vdw=vdw,
            electrostatic=electrostatic,
            polar_solvation=polar,
            nonpolar_solvation=nonpolar,
            entropy=None,
        ),
        metadata=metadata,
    )


# ---------------------------------------------------------------------
# Trajectory-metadata input resolution
# ---------------------------------------------------------------------


def input_from_metadata(
    metadata: Mapping[str, object], explicit_key: str, run_dir_name: str
) -> Path | None:
    """Locate a tool input file from trajectory metadata.

    Prefers an explicit metadata key; falls back to ``<run_dir>/<name>``
    when the trajectory records a ``run_dir`` (the MD wrappers do). Used
    by both MM/PB(GB)SA engines to find their structure/topology and
    trajectory files.
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


def as_provenance(value: object) -> Provenance | None:
    """Return ``value`` if it is a :class:`Provenance`, else ``None``."""
    return value if isinstance(value, Provenance) else None


# --- Per-residue decomposition (idecomp) ------------------------------------
#
# MMPBSA.py and gmx_MMPBSA write the same FINAL_DECOMP_MMPBSA.dat structure:
# per-species sections (Complex/Receptor/Ligand/DELTAS), each split into
# "Total"/"Sidechain"/"Backbone Energy Decomposition:" blocks. A data row is a
# residue label followed by 18 numbers — six terms (Internal, van der Waals,
# Electrostatic, Polar Solvation, Non-Polar Solv., TOTAL), each as
# (Avg., Std. Dev., Std. Err. of Mean). Both tools use these three sub-columns
# in decomposition output (the 5-column form is only in the summary results),
# so the row parser is shared; only the block-locating wording differs.

# Index of each term's average within the 18 trailing numbers.
_DECOMP_INTERNAL = 0
_DECOMP_VDW = 3
_DECOMP_ELECTROSTATIC = 6
_DECOMP_POLAR = 9
_DECOMP_NONPOLAR = 12
_DECOMP_TOTAL_AVG = 15
_DECOMP_TOTAL_SEM = 17
_DECOMP_NUMBERS = 18


def parse_decomp_row(line: str) -> ResidueContribution | None:
    """Parse one decomposition data row, or ``None`` if it isn't one.

    Normalises commas to whitespace and splits, so both the comma-separated
    (MMPBSA.py) and whitespace-separated (gmx_MMPBSA) renderings work. A data
    row is recognised structurally: its last 18 tokens must all parse as
    floats and at least one label token must precede them. Header and
    sub-header rows (``Residue``, ``Avg.``, ``van der Waals`` …) fail the
    float check and return ``None``.

    The residue label is the leading ``resname resnum`` pair. The delta
    section's rows carry an extra ``Location`` column (e.g. ``LEU 40 R LEU
    40`` — the residue's position in the receptor/ligand topology), which
    sits between the residue and the numbers and is dropped.

    Args:
        line: A single line from a decomposition block.

    Returns:
        The residue's :class:`ResidueContribution`, or ``None`` for a
        non-data row.
    """
    tokens = line.replace(",", " ").split()
    if len(tokens) < _DECOMP_NUMBERS + 1:
        return None
    tail = tokens[-_DECOMP_NUMBERS:]
    try:
        nums = [float(t) for t in tail]
    except ValueError:
        return None
    label = " ".join(tokens[: -_DECOMP_NUMBERS][:2])
    if not label:
        return None
    return ResidueContribution(
        residue=label,
        total=nums[_DECOMP_TOTAL_AVG],
        uncertainty=abs(nums[_DECOMP_TOTAL_SEM]),
        internal=nums[_DECOMP_INTERNAL],
        vdw=nums[_DECOMP_VDW],
        electrostatic=nums[_DECOMP_ELECTROSTATIC],
        polar_solvation=nums[_DECOMP_POLAR],
        nonpolar_solvation=nums[_DECOMP_NONPOLAR],
    )


def parse_decomp_block(block_text: str) -> list[ResidueContribution]:
    """Every residue contribution in a decomposition block, in order.

    Non-data rows (block/column headers) are skipped via
    :func:`parse_decomp_row`.
    """
    contributions = []
    for line in block_text.splitlines():
        contribution = parse_decomp_row(line)
        if contribution is not None:
            contributions.append(contribution)
    return contributions


def decomp_species_block(
    text: str,
    species_marker: str,
    *,
    block_marker: str = "Total Energy Decomposition:",
) -> str:
    """Slice out one species' decomposition block.

    Locates ``species_marker`` (e.g. ``"DELTAS:"``), then ``block_marker``
    after it (``"Total Energy Decomposition:"`` by default — the same marker
    recurs in every species section, so anchoring on the species first picks
    the right one), and returns the text up to the next
    ``"Energy Decomposition:"`` (the Sidechain/Backbone block) or the end.

    Args:
        text: The full decomposition file.
        species_marker: The species-section header to start from.
        block_marker: The decomposition block within that section.

    Returns:
        The block's text (residue rows plus its own headers).

    Raises:
        ValueError: If either marker is absent.
    """
    species_at = text.find(species_marker)
    if species_at == -1:
        raise ValueError(f"decomposition section {species_marker!r} not found")
    after_species = text[species_at + len(species_marker) :]

    block_at = after_species.find(block_marker)
    if block_at == -1:
        raise ValueError(
            f"{block_marker!r} not found in the {species_marker!r} section"
        )
    after_block = after_species[block_at + len(block_marker) :]

    end = after_block.find("Energy Decomposition:")
    return after_block if end == -1 else after_block[:end]


# Species-section markers, shared by MMPBSA.py and gmx_MMPBSA (gmx uses the
# same "DELTAS:"/"Complex:"/… headers, since it reuses MMPBSA.py's writer).
DECOMP_SECTIONS = {
    "delta": "DELTAS:",
    "complex": "Complex:",
    "receptor": "Receptor:",
    "ligand": "Ligand:",
}


def parse_decomp(text: str, *, section: str = "delta") -> Decomposition:
    """Parse a per-residue decomposition file into a :class:`Decomposition`.

    Shared by the Amber and GROMACS wrappers: both ``MMPBSA.py`` and
    ``gmx_MMPBSA`` write the same ``FINAL_DECOMP_MMPBSA.dat`` structure, so a
    single implementation reads the requested species' "Total Energy
    Decomposition" block for either.

    Args:
        text: Contents of ``FINAL_DECOMP_MMPBSA.dat``.
        section: Which species block — ``"delta"`` (default, the binding
            contribution), ``"complex"``, ``"receptor"``, or ``"ligand"``.

    Returns:
        The per-residue :class:`Decomposition`, in report order.

    Raises:
        ValueError: If ``section`` is unknown or the block is absent.
    """
    try:
        marker = DECOMP_SECTIONS[section.lower()]
    except KeyError:
        raise ValueError(
            f"unknown section {section!r}; choose from {sorted(DECOMP_SECTIONS)}"
        ) from None
    return Decomposition(parse_decomp_block(decomp_species_block(text, marker)))
