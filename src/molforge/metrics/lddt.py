"""lDDT (local Distance Difference Test) — alignment-free structure quality.

Reference: Mariani, V. et al. (2013) "lDDT: A local superposition-free
score for comparing protein structures and models using distance
difference tests." *Bioinformatics* 29: 2722-2728.

lDDT measures the fraction of pairwise inter-atom distances that are
preserved between a model and a reference structure, *without
requiring superposition*. This makes it ideal for:

  - Per-residue confidence scores (what AlphaFold's pLDDT and
    ESMFold's pLDDT estimate without seeing the reference).
  - Local quality assessment on flexible / multi-domain structures
    where a single global alignment is misleading.

Algorithm:
  1. Enumerate all pairs of CA atoms in the reference that are within
     ``inclusion_radius`` (default 15 Å). These are the "reference
     distances" the model is graded against.
  2. For each reference pair, compute the absolute difference between
     the reference distance and the corresponding model distance.
  3. Count the pair as "preserved" if the difference is below at least
     one of the tolerance thresholds (default 0.5/1/2/4 Å — matching
     the lDDT paper). The score per pair is the fraction of thresholds
     it passes (0, 0.25, 0.5, 0.75, or 1.0).
  4. Average over all pairs (for a global score) or over all pairs
     involving each residue (for per-residue scores).

The result is a value in ``[0, 1]``; higher = more accurate local
geometry. For AlphaFold-quality predictions, lDDT > 0.7 is typical for
ordered regions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from molforge.core import Protein

_DEFAULT_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)


def lddt(
    model: Protein,
    reference: Protein,
    *,
    inclusion_radius: float = 15.0,
    thresholds: tuple[float, ...] = _DEFAULT_THRESHOLDS,
) -> float:
    """Global lDDT score (alignment-free).

    Args:
        model: Predicted structure.
        reference: Native / target structure.
        inclusion_radius: Reference-distance cutoff in Å. Only pairs
            of residues within this distance in the *reference* are
            included in the score.
        thresholds: Tolerance thresholds. Default ``(0.5, 1, 2, 4)``
            matches the lDDT paper.

    Returns:
        lDDT score in ``[0, 1]``. Higher = better.
    """
    per_residue = lddt_per_residue(
        model,
        reference,
        inclusion_radius=inclusion_radius,
        thresholds=thresholds,
    )
    # Drop NaN entries (residues with no included pairs) before averaging.
    valid = per_residue[~np.isnan(per_residue)]
    if valid.size == 0:
        return 0.0
    return float(valid.mean())


def lddt_per_residue(
    model: Protein,
    reference: Protein,
    *,
    inclusion_radius: float = 15.0,
    thresholds: tuple[float, ...] = _DEFAULT_THRESHOLDS,
) -> NDArray[np.float32]:
    """Per-residue lDDT (the per-residue confidence pLDDT estimates).

    Args:
        model: Predicted structure.
        reference: Native / target structure.
        inclusion_radius: see :func:`lddt`.
        thresholds: see :func:`lddt`.

    Returns:
        ``(n_residues,)`` float32 array. Residues with no pair partners
        within ``inclusion_radius`` get ``NaN``.
    """
    from molforge.metrics.tm import _ca_coords

    m_coords = _ca_coords(model)
    r_coords = _ca_coords(reference)
    if m_coords.shape != r_coords.shape:
        raise ValueError(
            f"lDDT requires matched residue lists: model has "
            f"{m_coords.shape[0]} CAs, reference has {r_coords.shape[0]}."
        )
    n = m_coords.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    # Pairwise reference distances.
    ref_diff = r_coords[None, :, :] - r_coords[:, None, :]
    ref_dist = np.linalg.norm(ref_diff, axis=-1)
    mod_diff = m_coords[None, :, :] - m_coords[:, None, :]
    mod_dist = np.linalg.norm(mod_diff, axis=-1)

    # Include pairs within inclusion_radius in the reference, excluding
    # the diagonal (self-pairs).
    mask = (ref_dist < inclusion_radius) & (ref_dist > 0)

    # Per-pair difference and fraction-of-thresholds-passed.
    diff = np.abs(ref_dist - mod_dist)
    threshold_pass = np.stack(
        [(diff < t).astype(np.float64) for t in thresholds],
        axis=-1,
    )
    pair_score = threshold_pass.mean(axis=-1)  # (n, n)

    # For each residue, average over its included pair-partners.
    out = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        partners = mask[i]
        if partners.any():
            out[i] = float(pair_score[i, partners].mean())
    return out
