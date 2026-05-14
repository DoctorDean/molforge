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
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
        p1, p2, p3, p4: ``(3,)`` Cartesian coordinates.

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
