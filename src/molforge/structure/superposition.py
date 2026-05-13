"""Structural superposition: aligning one structure onto another.

The standard algorithm is the
[Kabsch / Umeyama](https://doi.org/10.1107/S0567739476001873) method:
given two coordinate sets of matching length, find the rotation that
minimizes their RMSD. This is closed-form (a single SVD of the
covariance matrix) and exact.

The :func:`superpose` function returns the superposed mobile coordinates
along with the rotation, translation, and RMSD. :func:`kabsch_rmsd` is a
shortcut for the common "what's the optimal RMSD" question.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SuperpositionResult:
    """Result of a structural superposition.

    Attributes:
        rotation: ``(3, 3)`` proper rotation matrix.
        translation: ``(3,)`` translation vector.
        rmsd: Root-mean-square deviation of the superposed structures.
        n_atoms: Number of atoms used in the superposition.
        mobile_aligned: ``(n_atoms, 3)`` mobile coords after superposition
            onto the reference.
    """

    rotation: NDArray[np.float64]
    translation: NDArray[np.float64]
    rmsd: float
    n_atoms: int
    mobile_aligned: NDArray[np.float32]


def superpose(
    mobile: NDArray[np.floating],
    reference: NDArray[np.floating],
    *,
    weights: NDArray[np.floating] | None = None,
) -> SuperpositionResult:
    """Superpose ``mobile`` onto ``reference`` by optimal rigid-body fit.

    Implements the Kabsch / Umeyama algorithm via SVD of the
    weighted covariance matrix. The returned rotation is guaranteed to
    be a *proper* rotation (det = +1), not a reflection.

    Args:
        mobile: ``(n, 3)`` coordinates to align.
        reference: ``(n, 3)`` reference coordinates.
        weights: Optional ``(n,)`` per-point weights (e.g. inverse-B-factor
            for X-ray structures, or 1/0 for masking atoms).

    Returns:
        A :class:`SuperpositionResult` with the rotation, translation,
        post-superposition RMSD, and aligned mobile coords.

    Raises:
        ValueError: If shapes mismatch or fewer than 3 atoms are given
            (degenerate; rotation under-determined).
    """
    m = np.asarray(mobile, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    if m.shape != r.shape:
        raise ValueError(f"shape mismatch: mobile {m.shape} vs reference {r.shape}")
    if m.ndim != 2 or m.shape[1] != 3:
        raise ValueError(f"expected (n, 3) coordinates, got {m.shape}")
    n = m.shape[0]
    if n < 3:
        raise ValueError(f"superposition requires at least 3 points, got {n}")

    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (n,):
            raise ValueError(f"weights shape {w.shape} must be ({n},)")
        if (w < 0).any():
            raise ValueError("weights must be non-negative")
    w_sum = w.sum()
    if w_sum <= 0:
        raise ValueError("at least one weight must be > 0")

    # Centroids
    cm = (w[:, None] * m).sum(axis=0) / w_sum
    cr = (w[:, None] * r).sum(axis=0) / w_sum
    m_centered = m - cm
    r_centered = r - cr

    # Covariance: H = (w*m)^T r
    cov = (m_centered * w[:, None]).T @ r_centered
    u, _, vt = np.linalg.svd(cov)

    # Ensure proper rotation (det = +1). If reflective, flip the last
    # column of V to make it a rotation.
    d = np.sign(np.linalg.det(vt.T @ u.T))
    diag = np.diag([1.0, 1.0, d])
    rotation = vt.T @ diag @ u.T
    translation = cr - rotation @ cm

    aligned = (rotation @ m.T).T + translation
    diff = aligned - r
    rmsd = float(np.sqrt((w[:, None] * (diff * diff)).sum() / w_sum / 3.0 * 3.0))
    # Note: standard RMSD divides by n (atoms), with the squared distance per
    # atom = sum over xyz. The simplification above ((diff^2).sum() / n)
    # gives the right answer:
    rmsd = float(np.sqrt(((diff * diff).sum(axis=1) * w).sum() / w_sum))

    return SuperpositionResult(
        rotation=rotation,
        translation=translation,
        rmsd=rmsd,
        n_atoms=n,
        mobile_aligned=aligned.astype(np.float32),
    )


def kabsch_rmsd(
    mobile: NDArray[np.floating],
    reference: NDArray[np.floating],
    *,
    weights: NDArray[np.floating] | None = None,
) -> float:
    """Return the minimum-RMSD over all rigid-body alignments.

    Convenience wrapper around :func:`superpose` for when you only want
    the RMSD value.
    """
    return superpose(mobile, reference, weights=weights).rmsd
