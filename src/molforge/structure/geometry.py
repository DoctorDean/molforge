"""General geometric utilities: centroids, radius of gyration, bounding boxes.

These are small, vectorized helpers that operate on :class:`Protein`
objects (or raw NumPy arrays via the ``_raw`` variants).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from molforge.core import Protein


# Standard atomic masses (Da) for the elements that appear in proteins
# at non-trivial frequency. Anything else falls back to 12 (carbon).
_ATOMIC_MASS: dict[str, float] = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "P": 30.974,
    "S": 32.06,
    "CL": 35.45,
    "BR": 79.904,
    "I": 126.904,
    "FE": 55.845,
    "ZN": 65.38,
    "MG": 24.305,
    "CA": 40.078,
    "NA": 22.990,
    "K": 39.098,
    "MN": 54.938,
    "CU": 63.546,
    "SE": 78.971,
}


def _masses(protein: Protein) -> NDArray[np.float64]:
    """Per-atom mass array (Da)."""
    elements = protein.atom_array.element
    out = np.array(
        [_ATOMIC_MASS.get(str(e).upper(), 12.0) for e in elements],
        dtype=np.float64,
    )
    return out


def centroid(protein: Protein, *, mass_weighted: bool = False) -> NDArray[np.float64]:
    """Geometric (or mass-weighted) centroid of a structure.

    Args:
        protein: input structure.
        mass_weighted: If True, weight by atomic mass (i.e. compute the
            center of mass instead).

    Returns:
        ``(3,)`` float64 centroid coordinate.
    """
    coords = protein.atom_array.coords.astype(np.float64)
    if coords.shape[0] == 0:
        return np.zeros(3, dtype=np.float64)
    if mass_weighted:
        m = _masses(protein)
        return cast(
            "NDArray[np.float64]", (m[:, None] * coords).sum(axis=0) / m.sum()
        )
    return cast("NDArray[np.float64]", coords.mean(axis=0))


def center_of_mass(protein: Protein) -> NDArray[np.float64]:
    """Mass-weighted center of mass. Alias for ``centroid(mass_weighted=True)``."""
    return centroid(protein, mass_weighted=True)


def radius_of_gyration(protein: Protein, *, mass_weighted: bool = True) -> float:
    """Radius of gyration — RMS distance from atoms to the center of mass.

    A standard compactness metric: smaller Rg means more globular.

    Args:
        protein: input structure.
        mass_weighted: If True (default), use mass-weighted Rg.

    Returns:
        Radius of gyration in angstroms.
    """
    coords = protein.atom_array.coords.astype(np.float64)
    if coords.shape[0] == 0:
        return 0.0
    if mass_weighted:
        m = _masses(protein)
        com = (m[:, None] * coords).sum(axis=0) / m.sum()
        delta = coords - com
        return float(np.sqrt(((m[:, None] * (delta * delta)).sum()) / m.sum()))
    com = coords.mean(axis=0)
    delta = coords - com
    return float(np.sqrt((delta * delta).sum() / coords.shape[0]))


def bounding_box(protein: Protein) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Axis-aligned bounding box of a structure.

    Returns:
        ``(min_xyz, max_xyz)``, each a ``(3,)`` float64 array.
    """
    coords = protein.atom_array.coords.astype(np.float64)
    if coords.shape[0] == 0:
        z = np.zeros(3, dtype=np.float64)
        return z, z.copy()
    return coords.min(axis=0), coords.max(axis=0)


def translate(protein: Protein, vector: NDArray[np.floating]) -> None:
    """Translate ``protein`` in place by ``vector``.

    Mutates the underlying ``AtomArray.coords`` directly — both
    hierarchical and linear views reflect the change immediately.
    """
    v = np.asarray(vector, dtype=np.float32)
    if v.shape != (3,):
        raise ValueError(f"translation must be (3,), got {v.shape}")
    protein.atom_array.coords += v


def rotate(protein: Protein, rotation: NDArray[np.floating]) -> None:
    """Apply a 3x3 rotation in place around the origin.

    For a rotation around the centroid, translate to origin first, rotate,
    then translate back. Use :func:`center_at_origin` as a helper.
    """
    r = np.asarray(rotation, dtype=np.float32)
    if r.shape != (3, 3):
        raise ValueError(f"rotation must be (3, 3), got {r.shape}")
    coords = protein.atom_array.coords
    coords[:] = coords @ r.T


def center_at_origin(protein: Protein, *, mass_weighted: bool = False) -> None:
    """Translate the structure so its centroid is at the origin (in place)."""
    c = centroid(protein, mass_weighted=mass_weighted).astype(np.float32)
    protein.atom_array.coords -= c
