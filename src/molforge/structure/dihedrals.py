"""Backbone (and arbitrary) dihedral angles.

Conventions:
- Phi (φ): C(i-1) - N(i) - CA(i) - C(i). Undefined for the N-terminal
  residue.
- Psi (ψ): N(i) - CA(i) - C(i) - N(i+1). Undefined for the C-terminal
  residue.
- Omega (ω): CA(i-1) - C(i-1) - N(i) - CA(i). ~180° (trans) for almost
  all residues, ~0° (cis) for the rare cis-proline.

All angles are returned in **degrees**, in the standard range
``[-180, 180]``. Where an angle is undefined (chain termini, missing
backbone atoms), the corresponding entry is ``NaN``.

This module also provides a simplified Ramachandran *classifier*
(:func:`ramachandran_type`, :func:`classify_ramachandran`,
:func:`ramachandran_outliers`, :func:`ramachandran_favored_fraction`)
that labels each residue's (φ, ψ) as Favored / Allowed / Outlier — a
coarse backbone-quality gate. See its section below for the region
model and its limitations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from molforge.core import Protein


def dihedral(
    p1: NDArray[np.floating],
    p2: NDArray[np.floating],
    p3: NDArray[np.floating],
    p4: NDArray[np.floating],
) -> float:
    """Compute the dihedral angle (in degrees) between four 3D points.

    Args:
        p1: ``(3,)`` Cartesian coordinates of the first point.
        p2: ``(3,)`` Cartesian coordinates of the second point.
        p3: ``(3,)`` Cartesian coordinates of the third point.
        p4: ``(3,)`` Cartesian coordinates of the fourth point.

    Returns:
        Angle in degrees in ``[-180, 180]``. Uses the standard atan2
        formula which avoids the numerical issues of acos near
        ``±1`` and naturally captures the sign.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)
    p4 = np.asarray(p4, dtype=np.float64)
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    # Normalize b2 so the projection step is unit-scaled.
    b2_norm = np.linalg.norm(b2)
    if b2_norm < 1e-9:
        return float("nan")
    b2_u = b2 / b2_norm
    # Projection of b1 perpendicular to b2, and b3 perpendicular to b2.
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2_u)
    x = float(np.dot(n1, n2))
    y = float(np.dot(m1, n2))
    return float(np.degrees(np.arctan2(y, x)))


def dihedrals_batch(
    quartets: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Vectorized dihedral over an array of atom quartets.

    Args:
        quartets: ``(N, 4, 3)`` array of four-atom quartets.

    Returns:
        ``(N,)`` float64 array of dihedral angles in degrees.
    """
    q = np.asarray(quartets, dtype=np.float64)
    if q.ndim != 3 or q.shape[1:] != (4, 3):
        raise ValueError(f"expected (N, 4, 3), got {q.shape}")
    p1 = q[:, 0]
    p2 = q[:, 1]
    p3 = q[:, 2]
    p4 = q[:, 3]
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    b2_norms = np.linalg.norm(b2, axis=1, keepdims=True)
    valid = b2_norms[:, 0] > 1e-9
    out = np.full(q.shape[0], np.nan, dtype=np.float64)
    if not valid.any():
        return out
    b2_u = np.where(valid[:, None], b2 / np.maximum(b2_norms, 1e-12), 0.0)
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2_u)
    x = (n1 * n2).sum(axis=1)
    y = (m1 * n2).sum(axis=1)
    angles = np.degrees(np.arctan2(y, x))
    out[valid] = angles[valid]
    return out


def _backbone_coords(
    protein: Protein,
) -> tuple[
    NDArray[np.float64],  # N (n_res, 3)
    NDArray[np.float64],  # CA (n_res, 3)
    NDArray[np.float64],  # C (n_res, 3)
    NDArray[np.bool_],  # mask of residues with complete N/CA/C
    NDArray[np.bool_],  # chain-start mask (True for first residue of each chain)
]:
    """Pull backbone N/CA/C coords plus completeness/chain-break info."""
    arr = protein.atom_array
    slices = list(arr.iter_residue_slices())
    n_res = len(slices)
    n_xyz = np.zeros((n_res, 3), dtype=np.float64)
    ca_xyz = np.zeros((n_res, 3), dtype=np.float64)
    c_xyz = np.zeros((n_res, 3), dtype=np.float64)
    mask = np.zeros(n_res, dtype=bool)
    is_start = np.zeros(n_res, dtype=bool)
    prev_chain: str | None = None
    for i, sl in enumerate(slices):
        chain = str(arr.chain_id[sl.start])
        if chain != prev_chain:
            is_start[i] = True
            prev_chain = chain
        if str(arr.entity_type[sl.start]) != "protein":
            continue
        names = arr.atom_name[sl]
        coords = arr.coords[sl]
        idx_n = np.where(names == "N")[0]
        idx_ca = np.where(names == "CA")[0]
        idx_c = np.where(names == "C")[0]
        if not (idx_n.size and idx_ca.size and idx_c.size):
            continue
        n_xyz[i] = coords[idx_n[0]]
        ca_xyz[i] = coords[idx_ca[0]]
        c_xyz[i] = coords[idx_c[0]]
        mask[i] = True
    return n_xyz, ca_xyz, c_xyz, mask, is_start


def phi_psi_omega(
    protein: Protein,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Per-residue backbone dihedrals (φ, ψ, ω) in degrees.

    Args:
        protein: structure to analyze.

    Returns:
        Three ``(n_residues,)`` float64 arrays of φ, ψ, ω values in
        degrees. Entries where the angle is undefined (chain termini,
        missing backbone atoms) are ``NaN``.
    """
    n_xyz, ca_xyz, c_xyz, mask, is_start = _backbone_coords(protein)
    n_res = len(mask)
    phi = np.full(n_res, np.nan, dtype=np.float64)
    psi = np.full(n_res, np.nan, dtype=np.float64)
    omega = np.full(n_res, np.nan, dtype=np.float64)

    for i in range(n_res):
        if not mask[i]:
            continue
        # Phi: C(i-1) - N(i) - CA(i) - C(i)
        if i > 0 and mask[i - 1] and not is_start[i]:
            phi[i] = dihedral(c_xyz[i - 1], n_xyz[i], ca_xyz[i], c_xyz[i])
        # Psi: N(i) - CA(i) - C(i) - N(i+1)
        if i + 1 < n_res and mask[i + 1] and not is_start[i + 1]:
            psi[i] = dihedral(n_xyz[i], ca_xyz[i], c_xyz[i], n_xyz[i + 1])
        # Omega: CA(i-1) - C(i-1) - N(i) - CA(i)
        if i > 0 and mask[i - 1] and not is_start[i]:
            omega[i] = dihedral(ca_xyz[i - 1], c_xyz[i - 1], n_xyz[i], ca_xyz[i])

    return phi, psi, omega


def phi(protein: Protein) -> NDArray[np.float64]:
    """φ (phi) angles per residue, degrees, NaN where undefined."""
    return phi_psi_omega(protein)[0]


def psi(protein: Protein) -> NDArray[np.float64]:
    """ψ (psi) angles per residue, degrees, NaN where undefined."""
    return phi_psi_omega(protein)[1]


def omega(protein: Protein) -> NDArray[np.float64]:
    """ω (omega) angles per residue, degrees, NaN where undefined."""
    return phi_psi_omega(protein)[2]


def ramachandran(protein: Protein) -> NDArray[np.float64]:
    """Per-residue (φ, ψ) pairs for Ramachandran-plot construction.

    Args:
        protein: structure to analyze.

    Returns:
        ``(n_residues, 2)`` float64 array. Rows where either angle is
        undefined contain ``NaN``s.
    """
    phi_, psi_, _ = phi_psi_omega(protein)
    return np.stack([phi_, psi_], axis=1)


# ---------------------------------------------------------------------
# Ramachandran classification
# ---------------------------------------------------------------------
#
# A residue's (φ, ψ) pair falls in one of three regions of the
# Ramachandran plot: "Favored" (where the vast majority of well-refined
# residues sit), "Allowed" (rarer but physically fine), or "Outlier"
# (sparsely populated — a flag for a possible modelling error).
#
# A faithful classifier uses 2D probability contours estimated from tens
# of thousands of reference residues (e.g. MolProbity's Top8000 grids).
# molforge deliberately does not ship those data tables; instead it uses
# a *simplified region model* — unions of rectangles in (φ, ψ) space,
# tuned to the standard secondary-structure basins, with separate region
# sets for glycine (achiral, so its map is point-symmetric) and proline
# (φ pinned near −63° by the ring). Pre-proline is treated with the
# general regions.
#
# This catches gross outliers (a non-glycine residue in the left-handed
# or mirror-β regions, a proline with positive φ, etc.) and gives a
# useful "% favored" quality signal, but it is not a MolProbity-grade
# percentile classifier. For publication-quality validation, run a tool
# with the reference distributions.

RamachandranClass = Literal["Favored", "Allowed", "Outlier"]
RamachandranCategory = Literal["General", "Glycine", "Proline"]

# Each region is a rectangle (phi_min, phi_max, psi_min, psi_max) in
# degrees, with angles in [-180, 180]. Regions that straddle the ψ = ±180
# seam are split into two rectangles.
_Box = tuple[float, float, float, float]

_GENERAL_FAVORED: tuple[_Box, ...] = (
    (-135.0, -40.0, -70.0, 5.0),      # right-handed α-helix
    (-180.0, -40.0, 100.0, 180.0),    # β-sheet / polyproline-II
    (-180.0, -40.0, -180.0, -165.0),  # β-sheet (ψ wrap)
)
_GENERAL_ALLOWED: tuple[_Box, ...] = (
    (-165.0, -35.0, -100.0, 40.0),    # α-helix (broad)
    (-180.0, -35.0, 60.0, 180.0),     # β / PPII (broad)
    (-180.0, -35.0, -180.0, -150.0),  # β (broad, ψ wrap)
    (35.0, 80.0, 5.0, 85.0),          # left-handed α (rare but real)
)

# Proline: φ confined near −63° by the pyrrolidine ring; two ψ basins.
_PROLINE_FAVORED: tuple[_Box, ...] = (
    (-90.0, -35.0, -55.0, 5.0),       # α
    (-90.0, -35.0, 120.0, 180.0),     # PPII
)
_PROLINE_ALLOWED: tuple[_Box, ...] = (
    (-100.0, -30.0, -80.0, 30.0),
    (-100.0, -30.0, 100.0, 180.0),
)


def _reflect(boxes: tuple[_Box, ...]) -> tuple[_Box, ...]:
    """Point-reflect boxes through the origin (φ,ψ) → (−φ,−ψ)."""
    return tuple(
        (-pmax, -pmin, -smax, -smin) for (pmin, pmax, smin, smax) in boxes
    )


# Glycine is achiral: its Ramachandran map is symmetric under
# (φ, ψ) → (−φ, −ψ), so the region set is the general set unioned with
# its reflection.
_GLYCINE_FAVORED: tuple[_Box, ...] = _GENERAL_FAVORED + _reflect(_GENERAL_FAVORED)
_GLYCINE_ALLOWED: tuple[_Box, ...] = _GENERAL_ALLOWED + _reflect(_GENERAL_ALLOWED)

_REGIONS: dict[RamachandranCategory, tuple[tuple[_Box, ...], tuple[_Box, ...]]] = {
    "General": (_GENERAL_FAVORED, _GENERAL_ALLOWED),
    "Glycine": (_GLYCINE_FAVORED, _GLYCINE_ALLOWED),
    "Proline": (_PROLINE_FAVORED, _PROLINE_ALLOWED),
}


@dataclass(frozen=True)
class RamachandranResult:
    """Classification of one residue's backbone conformation.

    Attributes:
        residue: ``(chain_id, residue_id, residue_name)``.
        phi, psi: Backbone dihedrals in degrees.
        category: Which region set was used (``General``/``Glycine``/
            ``Proline``).
        classification: ``Favored``/``Allowed``/``Outlier``.
    """

    residue: tuple[str, int, str]
    phi: float
    psi: float
    category: RamachandranCategory
    classification: RamachandranClass


def _in_boxes(phi: float, psi: float, boxes: tuple[_Box, ...]) -> bool:
    for pmin, pmax, smin, smax in boxes:
        if pmin <= phi <= pmax and smin <= psi <= smax:
            return True
    return False


def _category_for(residue_name: str) -> RamachandranCategory:
    name = residue_name.upper()
    if name == "GLY":
        return "Glycine"
    if name == "PRO":
        return "Proline"
    return "General"


def ramachandran_type(
    phi: float,
    psi: float,
    *,
    category: RamachandranCategory = "General",
) -> RamachandranClass:
    """Classify a single (φ, ψ) pair as Favored / Allowed / Outlier.

    Args:
        phi: Backbone φ dihedral in degrees, in ``[-180, 180]``.
        psi: Backbone ψ dihedral in degrees, in ``[-180, 180]``.
        category: Region set to use — ``General`` (default), ``Glycine``
            (symmetric map), or ``Proline`` (restricted φ).

    Returns:
        The region the pair falls in.

    Raises:
        ValueError: If ``phi`` or ``psi`` is not finite.
    """
    if not (np.isfinite(phi) and np.isfinite(psi)):
        raise ValueError("phi and psi must be finite; got NaN/inf")
    favored, allowed = _REGIONS[category]
    if _in_boxes(phi, psi, favored):
        return "Favored"
    if _in_boxes(phi, psi, allowed):
        return "Allowed"
    return "Outlier"


def classify_ramachandran(protein: Protein) -> list[RamachandranResult]:
    """Classify every residue with a defined (φ, ψ) pair.

    Residues at chain termini or across chain breaks (where φ or ψ is
    undefined) are skipped, so the result holds one entry per residue
    that actually has a backbone conformation to judge.

    Args:
        protein: Structure to analyze.

    Returns:
        One :class:`RamachandranResult` per classifiable residue, in
        chain/sequence order.
    """
    arr = protein.atom_array
    slices = list(arr.iter_residue_slices())
    phi_, psi_, _ = phi_psi_omega(protein)

    results: list[RamachandranResult] = []
    for i, sl in enumerate(slices):
        phi_i = float(phi_[i])
        psi_i = float(psi_[i])
        if not (np.isfinite(phi_i) and np.isfinite(psi_i)):
            continue
        resname = str(arr.residue_name[sl.start])
        category = _category_for(resname)
        results.append(
            RamachandranResult(
                residue=(
                    str(arr.chain_id[sl.start]),
                    int(arr.residue_id[sl.start]),
                    resname,
                ),
                phi=phi_i,
                psi=psi_i,
                category=category,
                classification=ramachandran_type(phi_i, psi_i, category=category),
            )
        )
    return results


def ramachandran_outliers(protein: Protein) -> list[RamachandranResult]:
    """Just the residues classified as Ramachandran outliers."""
    return [r for r in classify_ramachandran(protein) if r.classification == "Outlier"]


def ramachandran_favored_fraction(protein: Protein) -> float:
    """Fraction of classifiable residues in the favored region.

    A quality metric in the spirit of MolProbity's "Ramachandran
    favored" percentage (here from the simplified region model). Returns
    ``1.0`` for a structure with no classifiable residues (nothing to
    fault).
    """
    results = classify_ramachandran(protein)
    if not results:
        return 1.0
    favored = sum(1 for r in results if r.classification == "Favored")
    return favored / len(results)
