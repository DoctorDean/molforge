"""Pairwise RMSD and diversity statistics for pose ensembles.

These functions take ligand poses as input — assumed to share an atom
ordering, as poses returned from one docking run normally do — and
return either an N×N distance matrix or summary statistics.

For symmetric ligands (e.g. benzene, p-substituted phenyls) the RMSD
reported here is an upper bound on the true symmetry-aware RMSD;
permutation-aware RMSD via RDKit's :func:`GetBestRMS` is a planned
future enhancement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Sequence

    from molforge.docking import Pose


def pairwise_rmsd(
    poses: Sequence[Pose],
    *,
    heavy_atoms_only: bool = True,
) -> NDArray[np.float32]:
    """Compute the N×N pairwise heavy-atom RMSD matrix over a pose ensemble.

    Args:
        poses: Sequence of :class:`molforge.docking.Pose` objects, all
            of which must have the same number of atoms in the same
            order. (Poses returned by a single docking run normally do.)
        heavy_atoms_only: If ``True`` (default), hydrogens are stripped
            before computing distances. Most docking engines run on
            heavy atoms anyway, so for Vina output this is a no-op;
            for poses from other sources it's a sensible default.

    Returns:
        A symmetric ``(n, n)`` float32 matrix with zero diagonal. Entry
        ``[i, j]`` is the RMSD between pose ``i`` and pose ``j`` in
        the input units (Å for biology).

    Raises:
        ValueError: If poses don't share an atom ordering (different
            heavy-atom counts) or if any pose has no atoms.

    Example:
        >>> from molforge.ensembles import pairwise_rmsd
        >>> rmsd_matrix = pairwise_rmsd(result.poses)
        >>> rmsd_matrix[0, 1]  # RMSD between poses 0 and 1
        2.34
    """
    coords = _stack_pose_coords(poses, heavy_atoms_only=heavy_atoms_only)
    n = coords.shape[0]
    out = np.zeros((n, n), dtype=np.float32)

    if n < 2:
        return out

    # Vectorized: for each i, compute distances to all j>i in one shot.
    # Memory: O(n_atoms * n) for the broadcast, fine for typical
    # docking outputs of ~20 poses × ~50 atoms.
    for i in range(n - 1):
        diff = coords[i + 1 :] - coords[i]  # (n-i-1, n_atoms, 3)
        sq = (diff * diff).sum(axis=(1, 2))
        rmsd_row = np.sqrt(sq / coords.shape[1])
        out[i, i + 1 :] = rmsd_row
        out[i + 1 :, i] = rmsd_row

    return out


def pose_diversity(
    poses: Sequence[Pose],
    *,
    heavy_atoms_only: bool = True,
) -> dict[str, float]:
    """Summary statistics over the pairwise RMSD distribution of an ensemble.

    Use as a quick diagnostic for "did the docking actually explore?"
    A run where every pose is within 0.5 Å of every other pose probably
    converged on one binding mode; a run with mean pairwise RMSD > 3 Å
    found multiple modes.

    Args:
        poses: Sequence of poses, as for :func:`pairwise_rmsd`.
        heavy_atoms_only: see :func:`pairwise_rmsd`.

    Returns:
        A dict with keys:

        - ``"min"``: minimum off-diagonal pairwise RMSD.
        - ``"max"``: maximum pairwise RMSD.
        - ``"mean"``: mean of the upper triangle (excluding diagonal).
        - ``"median"``: median of the upper triangle.
        - ``"std"``: standard deviation of the upper triangle.
        - ``"n_poses"``: number of poses in the ensemble.

        For an ensemble of size 1, all distance statistics are ``0.0``.

    Raises:
        ValueError: If poses don't share an atom ordering or any pose
            has no atoms.
    """
    rmsd = pairwise_rmsd(poses, heavy_atoms_only=heavy_atoms_only)
    n = rmsd.shape[0]

    if n < 2:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "n_poses": float(n),
        }

    # Upper triangle, excluding diagonal.
    iu = np.triu_indices(n, k=1)
    upper = rmsd[iu]
    return {
        "min": float(upper.min()),
        "max": float(upper.max()),
        "mean": float(upper.mean()),
        "median": float(np.median(upper)),
        "std": float(upper.std()),
        "n_poses": float(n),
    }


# ---------- internals ----------


def _stack_pose_coords(
    poses: Sequence[Pose],
    *,
    heavy_atoms_only: bool,
) -> NDArray[np.float32]:
    """Stack ligand coordinates from all poses into a (n_poses, n_atoms, 3) array.

    Verifies that all poses share an atom ordering.
    """
    if not poses:
        raise ValueError("poses is empty")

    per_pose: list[NDArray[np.float32]] = []
    for i, pose in enumerate(poses):
        arr = pose.ligand.atom_array
        if heavy_atoms_only:
            mask = arr.element != "H"
            coords = arr.coords[mask]
        else:
            coords = arr.coords

        if coords.shape[0] == 0:
            raise ValueError(
                f"pose {i} has no atoms after filtering (heavy_atoms_only={heavy_atoms_only})"
            )
        per_pose.append(np.asarray(coords, dtype=np.float32))

    n_atoms_first = per_pose[0].shape[0]
    for i, c in enumerate(per_pose[1:], start=1):
        if c.shape[0] != n_atoms_first:
            raise ValueError(
                f"pose {i} has {c.shape[0]} atoms but pose 0 has "
                f"{n_atoms_first}; ensemble RMSD requires a consistent "
                "atom ordering across poses (which is what single-engine "
                "docking output normally provides)"
            )

    return np.stack(per_pose, axis=0)
