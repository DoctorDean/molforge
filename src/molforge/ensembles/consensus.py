"""Pick or build a representative pose from an ensemble.

Two strategies:

1. **Medoid** — pick the existing pose with the smallest summed
   (Boltzmann-weighted) RMSD to every other pose. This is a real pose
   from the ensemble, so it's chemically valid by construction. Best
   when "what's the representative binding mode?" is the question.

2. **Weighted mean** — synthesize a new pose by averaging the
   coordinates across the ensemble with the given weights. Returns a
   pose that may have chemically-invalid bond geometries (bond lengths
   shrink under averaging across diverse conformations). Useful only
   when the input ensemble has already been tight-clustered to similar
   conformations.

Default is medoid.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from molforge.ensembles.geometry import pairwise_rmsd

if TYPE_CHECKING:
    from collections.abc import Sequence

    from molforge.docking import Pose


def consensus_pose(
    poses: Sequence[Pose],
    *,
    weights: NDArray[np.floating] | None = None,
    method: str = "medoid",
    heavy_atoms_only: bool = True,
) -> Pose:
    """Pick or build a single pose representing the ensemble.

    Args:
        poses: Sequence of poses, all with the same atom ordering.
        weights: ``(n_poses,)`` array of weights. Default ``None`` =
            uniform. Typically from :func:`boltzmann_weights`. For
            ``method="medoid"``, weights bias which pose is chosen as
            representative (a high-weight outlier won't be picked
            unless it's also geometrically central). For
            ``method="mean"``, weights are used directly in the
            weighted average of coordinates.
        method: One of:

            - ``"medoid"`` (default): pick an actual pose from the
              ensemble that minimizes its summed weighted RMSD to the
              rest. Returns the original Pose object unchanged.
            - ``"mean"``: synthesize a new pose with coordinates that
              are the weighted average across the ensemble. The
              receptor and metadata are copied from the first pose;
              the score is set to the weighted average of input
              scores. Bond geometry is not guaranteed valid.

        heavy_atoms_only: see :func:`pairwise_rmsd`. Only affects
            the medoid choice (mean-averaging always uses all atoms).

    Returns:
        A single :class:`Pose`. For ``"medoid"`` this is one of the
        input poses, returned by reference. For ``"mean"`` it's a
        new deep-copied :class:`Pose` with averaged coordinates.

    Raises:
        ValueError: If ``poses`` is empty, ``method`` is unrecognized,
            or ``weights`` has the wrong length / doesn't sum to ~1.

    Example:
        >>> from molforge.ensembles import boltzmann_weights, consensus_pose
        >>> w = boltzmann_weights(docking.poses)
        >>> # Pick the most "central" high-affinity pose.
        >>> representative = consensus_pose(docking.poses, weights=w)
    """
    if not poses:
        raise ValueError("poses is empty")
    if method not in ("medoid", "mean"):
        raise ValueError(f"method must be 'medoid' or 'mean', got {method!r}")

    n = len(poses)
    if weights is None:
        w = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (n,):
            raise ValueError(f"weights has shape {w.shape}, expected ({n},)")
        if not np.isclose(w.sum(), 1.0, atol=1e-6):
            raise ValueError(f"weights must sum to 1.0, got {w.sum():.6f}")

    if method == "medoid":
        return _medoid_pose(poses, w, heavy_atoms_only=heavy_atoms_only)
    return _mean_pose(poses, w)


# ---------- internals ----------


def _medoid_pose(
    poses: Sequence[Pose],
    weights: NDArray[np.float64],
    *,
    heavy_atoms_only: bool,
) -> Pose:
    """Pose that minimizes its weighted summed RMSD to every other pose."""
    rmsd = pairwise_rmsd(poses, heavy_atoms_only=heavy_atoms_only)
    # For each i, compute sum_j w[j] * rmsd[i, j].
    weighted_sums = rmsd @ weights
    medoid_idx = int(weighted_sums.argmin())
    return poses[medoid_idx]


def _mean_pose(
    poses: Sequence[Pose],
    weights: NDArray[np.float64],
) -> Pose:
    """Synthesize a pose with weighted-average coordinates."""
    n = len(poses)
    # All poses must share atom count for a well-defined average.
    first_atom_array = poses[0].ligand.atom_array
    n_atoms = first_atom_array.coords.shape[0]
    for i, p in enumerate(poses):
        if p.ligand.atom_array.coords.shape[0] != n_atoms:
            raise ValueError(
                f"pose {i} has {p.ligand.atom_array.coords.shape[0]} atoms "
                f"but pose 0 has {n_atoms}; weighted-mean consensus requires "
                "a consistent atom ordering"
            )

    # Stack coords (n, n_atoms, 3) and average over axis 0 weighted by w.
    stacked = np.stack([p.ligand.atom_array.coords for p in poses], axis=0)  # (n, n_atoms, 3)
    averaged = np.tensordot(weights, stacked, axes=(0, 0))  # (n_atoms, 3)

    # Build a new pose: deepcopy the first pose's structure to get an
    # independent buffer, then overwrite the coords.
    new_pose = deepcopy(poses[0])
    new_pose.ligand.atom_array.coords[:] = averaged.astype(np.float32)
    # Score: weighted average of input scores.
    new_pose.score = float(np.dot(weights, [p.score for p in poses]))
    new_pose.rank = 0
    new_pose.rmsd_lb = None
    new_pose.rmsd_ub = None
    new_pose.metadata = {
        **(new_pose.metadata or {}),
        "consensus_method": "mean",
        "consensus_n_poses": n,
    }
    return new_pose
