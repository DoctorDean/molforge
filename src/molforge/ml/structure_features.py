"""Structure-level featurizers for ML models.

What's here:
    - :func:`pair_distances` — pairwise atom or residue distance map.
    - :func:`pair_distance_features` — distance binned into Gaussian RBFs
      (a standard featurization for distance-based GNNs).
    - :func:`pair_orientations` — backbone orientation features between
      pairs of residues (CA-CA vectors and angles).
    - :func:`local_environment` — per-residue local atomic environment
      summary (counts of atoms by element within a radius).
    - :func:`per_residue_features` — combined per-residue feature
      vectors that work as node features for protein GNNs.

All featurizers operate on a :class:`molforge.core.Protein` and return
NumPy float32 arrays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from molforge.core import Protein


def _ca_coords_and_labels(
    protein: Protein,
) -> tuple[NDArray[np.float32], list[tuple[str, int]]]:
    """Pull CA coordinates and (chain, residue_id) labels for protein residues."""
    arr = protein.atom_array
    coords: list[NDArray[np.float32]] = []
    labels: list[tuple[str, int]] = []
    for sl in arr.iter_residue_slices():
        if str(arr.entity_type[sl.start]) != "protein":
            continue
        names = arr.atom_name[sl]
        ca_idx = np.where(names == "CA")[0]
        if not ca_idx.size:
            continue
        coords.append(arr.coords[sl][ca_idx[0]])
        labels.append((str(arr.chain_id[sl.start]), int(arr.residue_id[sl.start])))
    return np.asarray(coords, dtype=np.float32), labels


def pair_distances(
    protein: Protein,
    *,
    atom_choice: str = "ca",
) -> NDArray[np.float32]:
    """Compute the residue-residue distance matrix.

    Convenience wrapper over :func:`molforge.structure.distance_map`
    that returns float32 for downstream tensor conversion.

    Args:
        protein: input structure.
        atom_choice: which atom defines residue position
            (``"ca"``, ``"cb"``, ``"heavy"``, ``"all"``).

    Returns:
        ``(n_res, n_res)`` float32 array of distances in Å.
    """
    from molforge.structure.contacts import distance_map

    return distance_map(protein, atom_choice=atom_choice)


def pair_distance_features(
    protein: Protein,
    *,
    n_bins: int = 16,
    d_min: float = 2.0,
    d_max: float = 22.0,
    atom_choice: str = "ca",
) -> NDArray[np.float32]:
    """Gaussian-radial-basis-function (RBF) encoding of pair distances.

    This is the standard featurization used in modern protein GNNs
    (Equivariant GNNs, GearNet, etc.). Each pair distance is expanded
    into ``n_bins`` Gaussian basis functions evenly spaced between
    ``d_min`` and ``d_max``, with sigma chosen so the bases overlap.

    Args:
        protein: input structure.
        n_bins: number of RBF basis functions. 16 is the typical default.
        d_min: lower end of the distance range covered by the basis (Å).
        d_max: upper end of the distance range covered by the basis (Å).
        atom_choice: anchor atom per residue.

    Returns:
        ``(n_res, n_res, n_bins)`` float32 array. The ``[i, j, k]`` entry
        is ``exp(-(d_ij - centers[k])^2 / (2 sigma^2))``.
    """
    d = pair_distances(protein, atom_choice=atom_choice)
    centers = np.linspace(d_min, d_max, n_bins, dtype=np.float32)
    # Width chosen so adjacent basis functions cross at half-max
    sigma = (d_max - d_min) / (n_bins - 1) if n_bins > 1 else 1.0
    diff = d[..., None] - centers[None, None, :]
    return np.exp(-(diff * diff) / (2.0 * sigma * sigma)).astype(np.float32)


def pair_orientations(
    protein: Protein,
) -> dict[str, NDArray[np.float32]]:
    """Backbone orientation features between every pair of residues.

    Computes for each residue pair (i, j):
      - ``direction``: unit vector from CA(i) to CA(j) in i's local frame
      - ``distance``: ``||CA(j) - CA(i)||``
      - ``cosine``: cosine of the angle between the CA(i)-CA(j) vector
        and residue i's local "forward" direction (CA(i+1) - CA(i-1)).
        Captures local orientation context.

    These are useful as edge features in equivariant-style protein GNNs.

    Returns:
        Dict with keys ``"direction"`` (``(n, n, 3)``), ``"distance"``
        (``(n, n)``), and ``"cosine"`` (``(n, n)``).
    """
    coords, _ = _ca_coords_and_labels(protein)
    n = coords.shape[0]
    if n == 0:
        return {
            "direction": np.zeros((0, 0, 3), dtype=np.float32),
            "distance": np.zeros((0, 0), dtype=np.float32),
            "cosine": np.zeros((0, 0), dtype=np.float32),
        }

    diff = coords[None, :, :] - coords[:, None, :]  # (n, n, 3)
    dist = np.linalg.norm(diff, axis=-1)
    # Avoid div-by-zero on the diagonal
    safe_dist = np.where(dist > 1e-6, dist, 1.0)
    direction = diff / safe_dist[..., None]

    # Per-residue forward direction: CA(i+1) - CA(i-1). Termini use a
    # single-sided difference.
    forward = np.zeros_like(coords)
    if n >= 2:
        forward[0] = coords[1] - coords[0]
        forward[-1] = coords[-1] - coords[-2]
    if n >= 3:
        forward[1:-1] = coords[2:] - coords[:-2]
    fwd_norm = np.linalg.norm(forward, axis=-1, keepdims=True)
    fwd_norm = np.where(fwd_norm > 1e-6, fwd_norm, 1.0)
    forward = forward / fwd_norm

    # cosine(theta) between (CA_j - CA_i) and i's forward direction.
    cosine = np.einsum("ijk,ik->ij", direction, forward).astype(np.float32)

    return {
        "direction": direction.astype(np.float32),
        "distance": dist.astype(np.float32),
        "cosine": cosine,
    }


def local_environment(
    protein: Protein,
    *,
    radius: float = 10.0,
) -> NDArray[np.float32]:
    """Per-residue local atomic environment counts.

    For each protein residue, count the atoms of each chemical element
    within ``radius`` Å of that residue's CA. This is a simple but
    effective featurization that captures local packing.

    Args:
        protein: input structure.
        radius: cutoff radius in Å (default 10).

    Returns:
        ``(n_res, 5)`` float32 array. Columns are counts of C, N, O,
        S, and "other" elements respectively.
    """
    arr = protein.atom_array
    ca_coords, _ = _ca_coords_and_labels(protein)
    n_res = ca_coords.shape[0]
    if n_res == 0:
        return np.zeros((0, 5), dtype=np.float32)

    all_coords = arr.coords.astype(np.float32)
    elements = arr.element

    # For each residue's CA, count atoms within radius by element bucket.
    element_idx_map = {"C": 0, "N": 1, "O": 2, "S": 3}
    out = np.zeros((n_res, 5), dtype=np.float32)
    radius_sq = radius * radius
    for i, ca in enumerate(ca_coords):
        diff = all_coords - ca
        dist_sq = (diff * diff).sum(axis=-1)
        within = dist_sq < radius_sq
        for j in np.where(within)[0]:
            el = str(elements[j]).upper()
            bucket = element_idx_map.get(el, 4)
            out[i, bucket] += 1.0
    return out


def per_residue_features(
    protein: Protein,
    *,
    include_sequence: bool = True,
    include_environment: bool = True,
    include_dssp: bool = True,
) -> NDArray[np.float32]:
    """Combined per-residue feature vectors suitable as GNN node features.

    Stacks (along the feature dimension):
      - One-hot residue identity (21 dims, if ``include_sequence``)
      - Local environment element counts (5 dims, if ``include_environment``)
      - DSSP 3-state one-hot (3 dims, if ``include_dssp``)

    Args:
        protein: input structure.
        include_sequence: include the one-hot residue identity block (21 dims).
        include_environment: include the local-environment block (5 dims).
        include_dssp: include the DSSP 3-state one-hot block (3 dims).

    Returns:
        ``(n_res, D)`` float32 array, where D depends on which blocks
        are included.
    """
    parts: list[NDArray[np.float32]] = []
    if include_sequence:
        from molforge.ml.sequence_features import one_hot

        seq = protein.sequence.replace("/", "")
        parts.append(one_hot(seq, include_unk=True))

    if include_environment:
        parts.append(local_environment(protein))

    if include_dssp:
        from molforge.structure import dssp_3state

        ss = dssp_3state(protein)
        # 3 columns for H/E/C
        ss_arr = np.zeros((len(ss), 3), dtype=np.float32)
        for i, c in enumerate(ss):
            if c == "H":
                ss_arr[i, 0] = 1.0
            elif c == "E":
                ss_arr[i, 1] = 1.0
            else:
                ss_arr[i, 2] = 1.0
        parts.append(ss_arr)

    if not parts:
        return np.zeros((0, 0), dtype=np.float32)

    # All parts should agree on leading dim. Trim to the shortest to be safe.
    n = min(p.shape[0] for p in parts)
    parts = [p[:n] for p in parts]
    return np.concatenate(parts, axis=-1).astype(np.float32)
