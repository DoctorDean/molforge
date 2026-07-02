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

from molforge.freeenergy import FreeEnergyComponents, FreeEnergyResult

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


__all__ = ["parse_mmpbsa_dat"]
