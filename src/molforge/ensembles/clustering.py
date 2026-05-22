"""Hierarchical clustering of poses by pairwise RMSD.

Implements average-linkage agglomerative clustering directly in NumPy
to avoid a scipy dependency in the base install. For ensemble sizes
typical of docking output (5-100 poses), this is plenty fast — the
naive O(n³) algorithm runs in microseconds. For larger ensembles
(e.g. MD trajectory subsamples) scipy's optimized implementation would
be worth it; we can swap in `scipy.cluster.hierarchy.linkage` when
the user has the `[structure]` extra installed as a future
enhancement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from molforge.ensembles.geometry import pairwise_rmsd

if TYPE_CHECKING:
    from collections.abc import Sequence

    from molforge.docking import Pose


@dataclass
class PoseCluster:
    """A single cluster from hierarchical pose clustering.

    Attributes:
        members: 0-indexed indices into the original pose list.
        medoid: Index of the pose whose summed RMSD to the rest of
            the cluster is smallest (the cluster's representative).
        size: Number of poses in the cluster.
        mean_intra_rmsd: Mean within-cluster pairwise RMSD; ``0.0`` for
            singletons.
    """

    members: list[int]
    medoid: int
    size: int
    mean_intra_rmsd: float


@dataclass
class PoseClusteringResult:
    """Output of :func:`pose_clusters`.

    Attributes:
        labels: ``(n_poses,)`` int array of cluster IDs (0-indexed).
            ``labels[i]`` gives the cluster containing pose ``i``.
        clusters: List of :class:`PoseCluster` objects, sorted by
            cluster size (largest first), then by medoid index.
        cutoff: The RMSD cutoff (Å) used to define clusters.
        n_clusters: Number of distinct clusters.
        rmsd_matrix: The N×N pairwise RMSD matrix used internally.
            Exposed so callers can plot it without recomputing.
    """

    labels: NDArray[np.intp]
    clusters: list[PoseCluster] = field(default_factory=list)
    cutoff: float = 0.0
    n_clusters: int = 0
    rmsd_matrix: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float32)
    )


def pose_clusters(
    poses: Sequence[Pose],
    *,
    cutoff: float = 2.0,
    heavy_atoms_only: bool = True,
) -> PoseClusteringResult:
    """Hierarchical (average-linkage) clustering of poses by RMSD.

    Two poses end up in the same cluster if their average linkage to
    each other (= mean pairwise RMSD across the current cluster
    membership during agglomeration) stays below ``cutoff``.

    Args:
        poses: Sequence of poses to cluster. Must share an atom
            ordering — see :func:`pairwise_rmsd`.
        cutoff: RMSD threshold in Å (default 2.0, a common docking-
            community value for "same binding mode"). Larger ⇒ fewer,
            looser clusters; smaller ⇒ more, tighter clusters.
        heavy_atoms_only: see :func:`pairwise_rmsd`.

    Returns:
        A :class:`PoseClusteringResult` with cluster labels, the
        :class:`PoseCluster` objects (sorted largest-first), and the
        RMSD matrix used internally.

    Raises:
        ValueError: If ``cutoff <= 0``, ``poses`` is empty, or poses
            don't share an atom ordering.

    Example:
        >>> from molforge.ensembles import pose_clusters
        >>> result = pose_clusters(docking.poses, cutoff=2.0)
        >>> print(f"{result.n_clusters} distinct binding modes")
        >>> # Get the medoid of the biggest cluster:
        >>> biggest = result.clusters[0]
        >>> representative = docking.poses[biggest.medoid]
    """
    if cutoff <= 0:
        raise ValueError(f"cutoff must be > 0, got {cutoff}")
    if not poses:
        raise ValueError("poses is empty")

    rmsd = pairwise_rmsd(poses, heavy_atoms_only=heavy_atoms_only)
    n = rmsd.shape[0]

    labels = _agglomerative_average_linkage(rmsd, cutoff)
    clusters = _build_clusters(labels, rmsd)

    return PoseClusteringResult(
        labels=labels,
        clusters=clusters,
        cutoff=float(cutoff),
        n_clusters=len(clusters),
        rmsd_matrix=rmsd,
    )


# ---------- internals ----------


def _agglomerative_average_linkage(
    distances: NDArray[np.float32],
    cutoff: float,
) -> NDArray[np.intp]:
    """Average-linkage agglomerative clustering, returning cluster labels.

    Pure NumPy. O(n³) which is fine for n ≲ 200; the typical ensemble
    is n ≲ 20 from one docking run, so even n³ runs in microseconds.

    The algorithm:
        1. Start with n singleton clusters.
        2. Find the pair of clusters with the smallest average-linkage
           distance.
        3. If that distance is ≤ cutoff, merge them and update the
           distance to all remaining clusters (weighted average).
        4. Repeat until no pair is within cutoff.
        5. Re-label so cluster IDs are contiguous from 0.
    """
    n = distances.shape[0]
    # Each cluster is a list of member indices; start with singletons.
    members: list[list[int]] = [[i] for i in range(n)]
    # Working distance matrix; we'll shrink rows/cols as we merge.
    # Initially this is a copy of the input.
    d = distances.astype(np.float64).copy()
    np.fill_diagonal(d, np.inf)  # avoid self-merges

    # Track the current cluster index for each merged group; -1 means
    # this slot has been merged into another.
    active = list(range(n))

    while True:
        if len(active) < 2:
            break

        # Find the closest pair among active clusters.
        sub = d[np.ix_(active, active)]
        flat_idx = int(sub.argmin())
        i_local, j_local = divmod(flat_idx, sub.shape[1])
        if i_local == j_local:
            break

        min_dist = sub[i_local, j_local]
        if min_dist > cutoff:
            break

        i = active[i_local]
        j = active[j_local]
        if i > j:
            i, j = j, i  # canonicalize so the smaller index is i

        # Merge j into i: i absorbs j's members, j becomes inactive.
        ni = len(members[i])
        nj = len(members[j])
        members[i] = members[i] + members[j]
        members[j] = []

        # Update average-linkage distances from i to every other active
        # cluster k != i, j. The weighted-average formula:
        #     d(i+j, k) = (ni * d(i,k) + nj * d(j,k)) / (ni + nj)
        for k in active:
            if k in (i, j):
                continue
            new_d = (ni * d[i, k] + nj * d[j, k]) / (ni + nj)
            d[i, k] = new_d
            d[k, i] = new_d

        # Inactivate j by setting all its row/col distances to inf.
        d[j, :] = np.inf
        d[:, j] = np.inf
        active.remove(j)

    # Build labels: each pose gets the index of its containing cluster.
    labels = np.full(n, -1, dtype=np.intp)
    for cluster_id, member_list in enumerate(
        m for m in members if m  # filter out empty (merged-out) clusters
    ):
        for member in member_list:
            labels[member] = cluster_id
    return labels


def _build_clusters(
    labels: NDArray[np.intp],
    rmsd: NDArray[np.float32],
) -> list[PoseCluster]:
    """Convert raw labels into PoseCluster objects with medoids + intra-RMSDs.

    Returned list is sorted by size desc, then by medoid index asc, so
    output is deterministic for ties.
    """
    n_clusters = int(labels.max()) + 1 if len(labels) > 0 else 0
    clusters: list[PoseCluster] = []

    for cluster_id in range(n_clusters):
        members = sorted(int(i) for i in np.where(labels == cluster_id)[0])
        size = len(members)

        if size == 1:
            clusters.append(
                PoseCluster(
                    members=members,
                    medoid=members[0],
                    size=1,
                    mean_intra_rmsd=0.0,
                )
            )
            continue

        # Medoid = the member whose total RMSD to the rest is smallest.
        sub_rmsd = rmsd[np.ix_(members, members)]
        sum_dists = sub_rmsd.sum(axis=1)
        medoid_local = int(sum_dists.argmin())
        medoid = members[medoid_local]

        # Mean intra-cluster RMSD over the upper triangle.
        iu = np.triu_indices(size, k=1)
        mean_intra = float(sub_rmsd[iu].mean()) if size > 1 else 0.0

        clusters.append(
            PoseCluster(
                members=members,
                medoid=medoid,
                size=size,
                mean_intra_rmsd=mean_intra,
            )
        )

    # Sort: bigger first, ties broken by medoid index ascending.
    clusters.sort(key=lambda c: (-c.size, c.medoid))
    return clusters
