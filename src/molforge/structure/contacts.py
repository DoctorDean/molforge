"""Inter-residue contacts and distance maps.

A *contact* between two residues exists when any atom of one is within
``cutoff`` Å of any atom of the other. This is the standard definition
used in contact-map prediction benchmarks (CASP, CAMEO).

A *distance map* gives the minimum inter-atom distance between every
pair of residues — a continuous generalization of the contact map.

Both work on a single chain (default) or across all chains.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from molforge.core import Protein


AtomChoice = Literal["ca", "cb", "heavy", "all"]


def _residue_anchor_coords(
    protein: Protein,
    *,
    atom_choice: AtomChoice,
) -> tuple[NDArray[np.float32], list[tuple[str, int, str]]]:
    """Return one representative coordinate per residue plus residue labels.

    For ``atom_choice="ca"`` we take the CA atom. For ``"cb"`` we take CB
    (CA for Gly). For ``"heavy"`` and ``"all"`` we return the residue
    centroid; callers using these likely want :func:`distance_map_pairwise`
    instead since they need every atom.
    """
    arr = protein.atom_array
    labels: list[tuple[str, int, str]] = []
    out: list[NDArray[np.float32]] = []
    for sl in arr.iter_residue_slices():
        names = arr.atom_name[sl]
        coords = arr.coords[sl]
        if atom_choice == "ca":
            idx = np.where(names == "CA")[0]
            point = coords[idx[0]] if idx.size else coords.mean(axis=0)
        elif atom_choice == "cb":
            idx = np.where(names == "CB")[0]
            if idx.size:
                point = coords[idx[0]]
            else:
                ca_idx = np.where(names == "CA")[0]
                point = coords[ca_idx[0]] if ca_idx.size else coords.mean(axis=0)
        else:
            point = coords.mean(axis=0)
        out.append(point)
        labels.append(
            (
                str(arr.chain_id[sl.start]),
                int(arr.residue_id[sl.start]),
                str(arr.insertion_code[sl.start]),
            )
        )
    return np.asarray(out, dtype=np.float32), labels


def distance_map(
    protein: Protein,
    *,
    atom_choice: AtomChoice = "ca",
) -> NDArray[np.float32]:
    """Compute a residue-by-residue distance map.

    Args:
        protein: structure to analyze.
        atom_choice: per-residue representative point — ``"ca"``,
            ``"cb"``, or the residue centroid (``"heavy"`` / ``"all"``).

    Returns:
        ``(n_res, n_res)`` float32 array of pairwise Euclidean distances
        between the representative points.
    """
    coords, _ = _residue_anchor_coords(protein, atom_choice=atom_choice)
    # Pairwise distance: ||a - b||
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt((diff * diff).sum(axis=-1))
    return d.astype(np.float32)


def contact_map(
    protein: Protein,
    *,
    cutoff: float = 8.0,
    atom_choice: AtomChoice = "cb",
    exclude_neighbors: int = 0,
) -> NDArray[np.bool_]:
    """Binary contact map at ``cutoff`` Å.

    Args:
        protein: structure to analyze.
        cutoff: distance below which residues are in contact (default
            8.0 Å, the CASP standard for CB-CB).
        atom_choice: which atom defines the residue position — defaults
            to ``"cb"`` (the field standard); use ``"ca"`` for Gly-heavy
            structures.
        exclude_neighbors: Set the diagonal band of width
            ``2*exclude_neighbors + 1`` to False. Useful for ignoring
            trivial sequential contacts; pass 4 to remove the ±4 band
            (helix neighbors) for instance.

    Returns:
        ``(n_res, n_res)`` boolean array; entry ``[i, j]`` is True if
        residue ``i`` and ``j`` are in contact.
    """
    d = distance_map(protein, atom_choice=atom_choice)
    contacts = (d < cutoff) & (d > 0)  # exclude self-contacts
    if exclude_neighbors > 0:
        for k in range(-exclude_neighbors, exclude_neighbors + 1):
            np.fill_diagonal(contacts[max(0, -k) :, max(0, k) :], False)
    return contacts


def residue_contacts(
    protein: Protein,
    *,
    cutoff: float = 5.0,
    chain_a: str | None = None,
    chain_b: str | None = None,
) -> list[tuple[tuple[str, int], tuple[str, int], float]]:
    """List inter-residue contacts at the all-atom level.

    Unlike :func:`contact_map`, this enumerates contacts as triples of
    ``((chain_a, resid_a), (chain_b, resid_b), distance)`` and uses the
    "any atom within cutoff" definition.

    Args:
        protein: structure to analyze.
        cutoff: distance threshold in Å (default 5.0).
        chain_a: If both ``chain_a`` and ``chain_b`` are given, only
            return contacts between those two chains (useful for
            interface analysis).
        chain_b: see ``chain_a``.

    Returns:
        Sorted list of contact tuples.
    """
    arr = protein.atom_array
    # Group atoms by residue
    residues: list[tuple[tuple[str, int], slice]] = []
    for sl in arr.iter_residue_slices():
        key = (str(arr.chain_id[sl.start]), int(arr.residue_id[sl.start]))
        residues.append((key, sl))

    if chain_a is not None and chain_b is not None:
        residues_a = [r for r in residues if r[0][0] == chain_a]
        residues_b = [r for r in residues if r[0][0] == chain_b]
    else:
        residues_a = residues
        residues_b = residues

    contacts: list[tuple[tuple[str, int], tuple[str, int], float]] = []
    coords = arr.coords
    for ka, sl_a in residues_a:
        coords_a = coords[sl_a]
        for kb, sl_b in residues_b:
            if ka >= kb:  # avoid duplicates and self
                continue
            coords_b = coords[sl_b]
            # Minimum pairwise distance between atom sets
            diff = coords_a[:, None, :] - coords_b[None, :, :]
            d = np.sqrt((diff * diff).sum(axis=-1))
            min_d = float(d.min())
            if min_d < cutoff:
                contacts.append((ka, kb, min_d))
    contacts.sort(key=lambda c: c[2])
    return contacts
