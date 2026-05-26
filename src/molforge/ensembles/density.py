"""Spatial densities for pose ensembles.

The fundamental operation: bin ligand atom positions across an
ensemble into a 3D grid. Each grid cell counts how often any ligand
atom landed there. Optionally weighted by Boltzmann (or other) weights
so high-affinity binding modes dominate.

The returned :class:`DensityGrid` can be:

- Saved as an OpenDX file for visualization in PyMOL or ChimeraX
  (writer not in v1; format is a one-page text spec, easy to add).
- Sliced for 2D heatmaps along x/y/z.
- Multiplied/added with other grids for differential analysis (which
  poses go *here* in cluster A vs. cluster B?).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Sequence

    from molforge.docking import Pose


@dataclass
class DensityGrid:
    """A 3D spatial density field over a regular grid.

    Attributes:
        density: ``(nx, ny, nz)`` float64 array of accumulated weight
            per grid cell. Cells are indexed by integer grid coordinate
            (not by Cartesian Å).
        origin: ``(3,)`` float32 Cartesian coordinate (Å) of the
            ``(0, 0, 0)`` corner of the grid.
        spacing: Edge length of one cubic grid cell, in Å.
        shape: ``(nx, ny, nz)`` integer shape of the grid (provided for
            convenience; equivalent to ``density.shape``).
        total_weight: Sum of all weights that landed inside the grid.
            Less than ``sum(weights)`` if atoms landed outside the
            bounding box.
    """

    density: NDArray[np.float64]
    origin: NDArray[np.float32]
    spacing: float
    shape: tuple[int, int, int]
    total_weight: float

    def coordinate_of(self, ijk: tuple[int, int, int]) -> NDArray[np.float32]:
        """Return the Cartesian coordinate of grid cell ``ijk``'s center.

        Args:
            ijk: 3-tuple of integer grid indices.

        Returns:
            ``(3,)`` float32 Cartesian coordinate in Å.
        """
        i, j, k = ijk
        return self.origin + self.spacing * (np.asarray([i, j, k], dtype=np.float32) + 0.5)


def binding_site_density(
    poses: Sequence[Pose],
    *,
    spacing: float = 1.0,
    padding: float = 4.0,
    weights: NDArray[np.floating] | None = None,
    heavy_atoms_only: bool = True,
    origin: NDArray[np.floating] | None = None,
    shape: tuple[int, int, int] | None = None,
) -> DensityGrid:
    """Compute a 3D spatial density of ligand atom positions across the ensemble.

    Each ligand atom in each pose contributes ``weights[i]`` (default
    uniform = ``1/n_poses``) to the grid cell containing it. Atoms
    landing outside the grid are silently discarded; the returned
    ``total_weight`` tells you how much weight made it in.

    By default the grid is auto-sized: it's the axis-aligned bounding
    box of all heavy ligand atoms across all poses, padded by
    ``padding`` Å on each side, with the requested ``spacing``. For
    differential / comparative analysis (where you want two ensembles
    on the same grid), pass explicit ``origin`` and ``shape``.

    Args:
        poses: Sequence of poses. Ligand atoms across all poses
            determine the bounding box by default.
        spacing: Cubic grid spacing in Å. Default ``1.0`` (resolves to
            something a bit finer than typical ligand atom-atom
            distances but coarser than a true density map).
        padding: Bounding-box padding in Å on each side, applied only
            when ``origin`` is not given. Default ``4.0``.
        weights: ``(n_poses,)`` array of weights. Default ``None`` =
            uniform ``1/n_poses`` (so total weight ≤ 1.0). Typically
            comes from :func:`boltzmann_weights`.
        heavy_atoms_only: see :func:`pairwise_rmsd`.
        origin: ``(3,)`` Cartesian coordinate of the grid's
            ``(0, 0, 0)`` corner. If provided, ``shape`` must also be
            provided, and the auto-bounding-box logic is bypassed.
        shape: ``(nx, ny, nz)`` explicit grid shape; required if and
            only if ``origin`` is provided.

    Returns:
        A :class:`DensityGrid`.

    Raises:
        ValueError: If ``spacing <= 0``, ``poses`` is empty, ``weights``
            has the wrong length, or only one of ``origin`` / ``shape``
            is provided.

    Example:
        >>> from molforge.ensembles import boltzmann_weights, binding_site_density
        >>> weights = boltzmann_weights(docking.poses)
        >>> grid = binding_site_density(docking.poses, spacing=0.5, weights=weights)
        >>> hot_spot = np.unravel_index(grid.density.argmax(), grid.shape)
        >>> grid.coordinate_of(hot_spot)  # the most-occupied point in Å
        array([12.3, 45.6, 78.9], dtype=float32)
    """
    if spacing <= 0:
        raise ValueError(f"spacing must be > 0, got {spacing}")
    if not poses:
        raise ValueError("poses is empty")
    if (origin is None) != (shape is None):
        raise ValueError(
            "origin and shape must both be provided together "
            "(for fixed-grid mode) or both be None (for auto-sizing)"
        )

    n = len(poses)
    if weights is None:
        w = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (n,):
            raise ValueError(f"weights has shape {w.shape}, expected ({n},)")

    # Collect heavy ligand coordinates per pose.
    coords_per_pose: list[NDArray[np.float32]] = []
    for pose in poses:
        arr = pose.ligand.atom_array
        if heavy_atoms_only:
            mask = arr.element != "H"
            coords_per_pose.append(np.asarray(arr.coords[mask], dtype=np.float32))
        else:
            coords_per_pose.append(np.asarray(arr.coords, dtype=np.float32))

    # Determine grid extent.
    if origin is None:
        all_coords = np.concatenate(coords_per_pose, axis=0)
        lo = all_coords.min(axis=0) - padding
        hi = all_coords.max(axis=0) + padding
        grid_origin = lo.astype(np.float32)
        grid_shape_arr = np.ceil((hi - lo) / spacing).astype(int)
        grid_shape: tuple[int, int, int] = (
            int(grid_shape_arr[0]),
            int(grid_shape_arr[1]),
            int(grid_shape_arr[2]),
        )
    else:
        grid_origin = np.asarray(origin, dtype=np.float32).ravel()
        if grid_origin.shape != (3,):
            raise ValueError(f"origin must have shape (3,), got {grid_origin.shape}")
        grid_shape = (int(shape[0]), int(shape[1]), int(shape[2]))  # type: ignore[index]

    density = np.zeros(grid_shape, dtype=np.float64)
    nx, ny, nz = grid_shape
    total_weight = 0.0

    # Per-atom contribution: weight[i] per atom in pose i means each
    # pose contributes weight[i] * n_atoms_i to the total. That's how
    # the user-readable "what's the density around the active site"
    # question is usually phrased — high atom count in a region means
    # high density, which is the desired semantic.
    for i, coords in enumerate(coords_per_pose):
        if coords.shape[0] == 0:
            continue
        # Convert Å → integer grid indices.
        idx = np.floor((coords - grid_origin) / spacing).astype(int)
        # Mask atoms inside the grid.
        in_bounds = (
            (idx[:, 0] >= 0)
            & (idx[:, 0] < nx)
            & (idx[:, 1] >= 0)
            & (idx[:, 1] < ny)
            & (idx[:, 2] >= 0)
            & (idx[:, 2] < nz)
        )
        idx = idx[in_bounds]
        if idx.shape[0] == 0:
            continue
        # Accumulate (vectorized).
        np.add.at(density, (idx[:, 0], idx[:, 1], idx[:, 2]), w[i])
        total_weight += float(w[i] * idx.shape[0])

    return DensityGrid(
        density=density,
        origin=grid_origin,
        spacing=float(spacing),
        shape=grid_shape,
        total_weight=total_weight,
    )
