"""GDT (Global Distance Test) scores: GDT-TS and GDT-HA.

References:
  - Zemla, A. (2003) "LGA: a method for finding 3D similarities in
    protein structures." *Nucleic Acids Res* 31: 3370-3374.
  - CASP evaluation criteria.

GDT-TS (Total Score) is the gold-standard metric used in CASP. It's
the fraction of residues that can be superposed within four distance
cutoffs, averaged together:

    GDT-TS = (P(1 Å) + P(2 Å) + P(4 Å) + P(8 Å)) / 4

where ``P(d)`` is the fraction of model residues whose CA atoms are
within ``d`` Å of their reference counterpart after optimal
superposition.

GDT-HA (High Accuracy) uses tighter cutoffs (0.5/1/2/4 Å) and is the
metric for *near-experimental* predictions.

Both return values in ``[0, 1]``; higher is better. GDT-TS > 0.5 ≈
correct fold; > 0.9 ≈ near-experimental accuracy.

Like TM-score, this implementation assumes the two structures' residue
lists are already in correspondence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

from molforge.structure.superposition import superpose

if TYPE_CHECKING:
    from molforge.core import Protein

_GDT_TS_CUTOFFS = (1.0, 2.0, 4.0, 8.0)
_GDT_HA_CUTOFFS = (0.5, 1.0, 2.0, 4.0)


def _ca_distances_after_superposition(model: Protein, reference: Protein) -> NDArray[np.float64]:
    """Return per-residue CA distances after optimal superposition."""
    from molforge.metrics.tm import _ca_coords

    m_coords = _ca_coords(model)
    r_coords = _ca_coords(reference)
    if m_coords.shape != r_coords.shape:
        raise ValueError(
            f"GDT requires matched residue lists: model has "
            f"{m_coords.shape[0]} CAs, reference has {r_coords.shape[0]}."
        )
    if m_coords.shape[0] < 3:
        raise ValueError(f"GDT requires at least 3 residues, got {m_coords.shape[0]}")
    result = superpose(m_coords, r_coords)
    diff = result.mobile_aligned.astype(np.float64) - r_coords
    return cast("NDArray[np.float64]", np.linalg.norm(diff, axis=1))


def gdt_ts(model: Protein, reference: Protein) -> float:
    """GDT-TS: the CASP standard metric for fold-level prediction quality.

    Args:
        model: The predicted structure.
        reference: The native / target structure.

    Returns:
        GDT-TS in ``[0, 1]``. Higher = better.
    """
    distances = _ca_distances_after_superposition(model, reference)
    n = distances.shape[0]
    fractions = [float((distances < c).sum()) / n for c in _GDT_TS_CUTOFFS]
    return float(np.mean(fractions))


def gdt_ha(model: Protein, reference: Protein) -> float:
    """GDT-HA: high-accuracy variant of GDT-TS with tighter cutoffs.

    Args:
        model: The predicted structure.
        reference: The native / target structure.

    Returns:
        GDT-HA in ``[0, 1]``. Higher = better.
    """
    distances = _ca_distances_after_superposition(model, reference)
    n = distances.shape[0]
    fractions = [float((distances < c).sum()) / n for c in _GDT_HA_CUTOFFS]
    return float(np.mean(fractions))


def gdt_per_cutoff(
    model: Protein,
    reference: Protein,
    *,
    cutoffs: tuple[float, ...] = _GDT_TS_CUTOFFS,
) -> dict[float, float]:
    """Per-cutoff fractions used internally by GDT-TS / GDT-HA.

    Useful for plotting accuracy curves or building custom metrics.

    Args:
        model: Predicted structure to score.
        reference: Reference structure (e.g. native or experimental).
        cutoffs: Distance cutoffs in Å. Defaults to GDT-TS's (1, 2, 4, 8).

    Returns:
        Dict mapping cutoff to fraction of residues within that cutoff
        after optimal superposition.
    """
    distances = _ca_distances_after_superposition(model, reference)
    n = distances.shape[0]
    return {float(c): float((distances < c).sum()) / n for c in cutoffs}
