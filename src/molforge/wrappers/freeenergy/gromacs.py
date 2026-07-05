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

import re
from typing import TYPE_CHECKING

from molforge.wrappers.freeenergy import _common

if TYPE_CHECKING:
    from molforge.freeenergy import FreeEnergyResult

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


__all__ = ["parse_gmx_mmpbsa_dat"]
