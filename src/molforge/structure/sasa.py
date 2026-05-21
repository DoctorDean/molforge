"""Solvent-accessible surface area (Shrake-Rupley 1973).

Reference: Shrake, A. & Rupley, J. A. (1973) J. Mol. Biol. 79: 351-371.
"Environment and exposure to solvent of protein atoms. Lysozyme and
insulin."

The Shrake-Rupley algorithm places `n_sphere_points` test points on
the van-der-Waals + probe-radius sphere around each atom and counts
how many are *not* occluded by other atoms. The fraction of unoccluded
points times the sphere area gives the atom's accessible surface.

Default radii are the Bondi 1964 set adjusted for biomolecular use
(taken from NACCESS / FreeSASA). Probe radius defaults to 1.4 Å
(roughly the van-der-Waals radius of water).

Performance: O(n_atoms * n_sphere_points * n_neighbors). For typical
protein-sized inputs (~3000 atoms, 100 points, ~50 neighbors per atom)
this is around 1-2 seconds in pure NumPy. For high-throughput use,
route through FreeSASA's C implementation; this version exists for
dependency-free convenience.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from molforge.core import Protein


# Element -> van-der-Waals radius (Å). Based on Bondi 1964 with
# biomolecular adjustments per NACCESS.
_VDW_RADII: dict[str, float] = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "FE": 1.80,
    "ZN": 1.39,
    "MG": 1.73,
    "CA": 2.31,
    "NA": 2.27,
    "K": 2.75,
    "MN": 1.97,
    "CU": 1.40,
    "SE": 1.90,
}
_DEFAULT_RADIUS: float = 1.70  # carbon, sensible fallback


def _generate_sphere_points(n: int) -> NDArray[np.float64]:
    """Generate ``n`` (approximately) evenly-distributed points on the unit sphere.

    Uses the standard golden-spiral / Fibonacci method, which gives
    a very uniform distribution with no special cases at the poles.
    """
    if n < 1:
        raise ValueError(f"need at least 1 sphere point, got {n}")
    indices = np.arange(0, n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / n)
    theta = np.pi * (1.0 + 5.0**0.5) * indices  # golden angle
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)


def _vdw_radii(protein: Protein) -> NDArray[np.float64]:
    """Per-atom van-der-Waals radii in Å."""
    elements = protein.atom_array.element
    return np.array(
        [_VDW_RADII.get(str(e).upper(), _DEFAULT_RADIUS) for e in elements],
        dtype=np.float64,
    )


def sasa(
    protein: Protein,
    *,
    probe_radius: float = 1.4,
    n_sphere_points: int = 100,
) -> NDArray[np.float64]:
    """Per-atom solvent-accessible surface area (Shrake-Rupley).

    Args:
        protein: input structure.
        probe_radius: radius of the rolling probe in Å (water = 1.4).
        n_sphere_points: number of points sampled on each atom's
            extended sphere. More = more accurate, slower (linear).
            100 is the common default; 960 matches NACCESS exactly.

    Returns:
        ``(n_atoms,)`` float64 array of per-atom SASA values in Å².

    Notes:
        Hydrogens are typically excluded from SASA calculations in
        biomolecular contexts. This implementation includes them if
        present; pre-filter via ``protein.select(element=...)`` or
        ``include_hydrogens=False`` at parse time if you want them
        excluded.
    """
    arr = protein.atom_array
    n_atoms = len(arr)
    if n_atoms == 0:
        return np.zeros(0, dtype=np.float64)

    coords = arr.coords.astype(np.float64)
    radii = _vdw_radii(protein) + probe_radius
    sphere = _generate_sphere_points(n_sphere_points)
    n_pts = sphere.shape[0]

    out = np.zeros(n_atoms, dtype=np.float64)
    # For neighbor-finding, the worst-case interaction range is twice the
    # max extended radius.
    max_r = float(radii.max())

    for i in range(n_atoms):
        # Project sphere points to atom i's surface.
        test_points = coords[i] + radii[i] * sphere

        # Find neighbors within cutoff (vectorized).
        delta = coords - coords[i]
        d2 = (delta * delta).sum(axis=1)
        mask = (d2 > 0) & (d2 < (radii[i] + max_r) ** 2)
        # Tighter: only atoms whose own extended sphere could reach i's sphere.
        neighbor_idx = np.where(mask)[0]
        if neighbor_idx.size == 0:
            # All points are accessible
            sphere_area = 4.0 * np.pi * radii[i] ** 2
            out[i] = sphere_area
            continue
        nbr_coords = coords[neighbor_idx]
        nbr_radii = radii[neighbor_idx]

        # For each test point, check whether it lies inside any neighbor.
        # diffs: (n_pts, n_neighbors, 3)
        diffs = test_points[:, None, :] - nbr_coords[None, :, :]
        d2_pts = (diffs * diffs).sum(axis=-1)
        # A point is occluded if its distance² to any neighbor < that
        # neighbor's extended radius².
        occluded = (d2_pts < (nbr_radii**2)[None, :]).any(axis=1)
        n_visible = int(np.count_nonzero(~occluded))
        sphere_area = 4.0 * np.pi * radii[i] ** 2
        out[i] = sphere_area * (n_visible / n_pts)

    return out


def sasa_per_residue(
    protein: Protein,
    *,
    probe_radius: float = 1.4,
    n_sphere_points: int = 100,
) -> NDArray[np.float64]:
    """Per-residue SASA, summed across atoms in each residue.

    Args:
        protein: input structure.
        probe_radius: see :func:`sasa`.
        n_sphere_points: see :func:`sasa`.

    Returns:
        ``(n_residues,)`` float64 array, one SASA value per residue
        in array order.
    """
    per_atom = sasa(
        protein,
        probe_radius=probe_radius,
        n_sphere_points=n_sphere_points,
    )
    arr = protein.atom_array
    out: list[float] = []
    for sl in arr.iter_residue_slices():
        out.append(float(per_atom[sl].sum()))
    return np.asarray(out, dtype=np.float64)


def total_sasa(
    protein: Protein,
    *,
    probe_radius: float = 1.4,
    n_sphere_points: int = 100,
) -> float:
    """Total solvent-accessible surface area (Å²)."""
    return float(
        sasa(
            protein,
            probe_radius=probe_radius,
            n_sphere_points=n_sphere_points,
        ).sum()
    )
