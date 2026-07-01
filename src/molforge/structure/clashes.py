"""Steric clash detection for structure-quality validation.

A *steric clash* is a pair of non-bonded atoms whose van der Waals
shells overlap by more than a tolerance — the single most common
signature of a physically implausible structure, and a cheap gate to
put in front of folding / docking output before spending compute
downstream.

Definition
----------

For two atoms with van der Waals radii ``r_i`` and ``r_j`` a distance
``d`` apart, the *overlap* is::

    overlap = (r_i + r_j) - d

The pair is a clash when ``overlap >= tolerance``. The default
tolerance of 0.4 Å matches the MolProbity convention (an "all-atom
contact" overlap of 0.4 Å), so :func:`clash_score` reports clashes
per 1000 atoms in the same spirit as MolProbity's clashscore.

Bonded exclusion without a topology
-----------------------------------

Covalently bonded atoms sit *well* inside each other's van der Waals
sum, so they would swamp the output with false positives. molforge does
not carry a bond graph, so one is *inferred from geometry*: two atoms
are bonded when they lie within the sum of their covalent radii plus a
small tolerance. Any pair separated by at most ``bonded_separation``
bonds in that inferred graph is excluded (default 3, i.e. 1-2, 1-3 and
1-4 neighbours). This transparently handles intra-residue bonds, the
peptide bond, disulfides and bonds inside a bound ligand, without any
per-residue special-casing.

The inference is distance-based, so a badly distorted input can in
principle mis-assign a bond; set ``bonded_separation=0`` to disable
exclusion entirely and see every raw overlap. Hydrogens are ignored by
default (folding output frequently lacks them).

Example::

    from molforge.structure import find_clashes, clash_score

    bad = find_clashes(model)
    if bad:
        print(f"{len(bad)} clashes; worst overlap {bad[0].overlap:.2f} Å")
    print("clashscore:", clash_score(model))
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from molforge.core import Protein

# Van der Waals radii in Å. Bondi (1964) for the biological main-group
# elements, extended with the Mantina et al. (2009) consistent set for
# the alkali/alkaline-earth ions and a handful of transition metals
# commonly seen as cofactors. Unknown elements fall back to
# DEFAULT_VDW_RADIUS.
VDW_RADII: dict[str, float] = {
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
    "SE": 1.90,
    "SI": 2.10,
    "B": 1.92,
    "AS": 1.85,
    "LI": 1.82,
    "NA": 2.27,
    "K": 2.75,
    "MG": 1.73,
    "CA": 2.31,
    "ZN": 1.39,
    "MN": 2.05,
    "FE": 2.04,
    "CU": 1.40,
    "CO": 2.00,
    "NI": 1.63,
}

# Single-bond covalent radii in Å (Cordero et al. 2008), used only to
# *infer* connectivity for bonded exclusion. Unknown elements fall back
# to DEFAULT_COVALENT_RADIUS.
COVALENT_RADII: dict[str, float] = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "CL": 1.02,
    "BR": 1.20,
    "I": 1.39,
    "SE": 1.20,
    "SI": 1.11,
    "B": 0.84,
    "AS": 1.19,
    "LI": 1.28,
    "NA": 1.66,
    "K": 2.03,
    "MG": 1.41,
    "CA": 1.76,
    "ZN": 1.22,
    "MN": 1.39,
    "FE": 1.32,
    "CU": 1.32,
    "CO": 1.26,
    "NI": 1.24,
}

#: Fallback radius (Å) for elements absent from :data:`VDW_RADII`.
DEFAULT_VDW_RADIUS: float = 1.70

#: Fallback radius (Å) for elements absent from :data:`COVALENT_RADII`.
DEFAULT_COVALENT_RADIUS: float = 0.77

#: Default overlap tolerance (Å) — the MolProbity clash threshold.
DEFAULT_TOLERANCE: float = 0.4

#: Slack (Å) added to the covalent-radii sum when inferring bonds.
_BOND_TOLERANCE: float = 0.45

#: Minimum distance (Å) for an inferred bond (guards against duplicate
#: atoms at identical coordinates being called "bonded").
_MIN_BOND_DISTANCE: float = 0.4


@dataclass(frozen=True)
class Clash:
    """A single clashing pair of atoms.

    Attributes:
        atom_i, atom_j: Global atom indices into ``protein.atom_array``
            (``atom_i < atom_j``).
        element_i, element_j: Element symbols (upper-cased).
        distance: Inter-atom distance in Å.
        vdw_sum: Sum of the two van der Waals radii in Å.
        overlap: ``vdw_sum - distance`` in Å (>= tolerance).
        residue_i, residue_j: ``(chain_id, residue_id, residue_name)``
            for each atom's residue.
    """

    atom_i: int
    atom_j: int
    element_i: str
    element_j: str
    distance: float
    vdw_sum: float
    overlap: float
    residue_i: tuple[str, int, str]
    residue_j: tuple[str, int, str]


def _radii_for(
    elements: NDArray[np.str_],
    table: dict[str, float],
    fallback: float,
) -> NDArray[np.float64]:
    """Map element symbols to radii from ``table`` with a fallback."""
    return np.array(
        [table.get(str(e).upper(), fallback) for e in elements],
        dtype=np.float64,
    )


def _residue_indices(
    protein: Protein,
) -> tuple[NDArray[np.int64], list[tuple[str, int, str]]]:
    """Per-atom residue index plus a ``(chain, resid, resname)`` label."""
    arr = protein.atom_array
    res_index = np.empty(len(arr), dtype=np.int64)
    labels: list[tuple[str, int, str]] = []
    for r, sl in enumerate(arr.iter_residue_slices()):
        res_index[sl] = r
        labels.append(
            (
                str(arr.chain_id[sl.start]),
                int(arr.residue_id[sl.start]),
                str(arr.residue_name[sl.start]),
            )
        )
    return res_index, labels


def _row_block(m: int) -> int:
    """Row-block size that bounds a chunked (block, m) matrix to ~64 MB."""
    return max(1, min(m, int(64_000_000 // (8 * m))))


def _infer_adjacency(
    coords: NDArray[np.float64],
    covalent: NDArray[np.float64],
) -> list[set[int]]:
    """Infer covalent bonds from geometry → adjacency list.

    Two atoms are bonded when ``_MIN_BOND_DISTANCE < d <= cov_i + cov_j
    + _BOND_TOLERANCE``. Chunked to keep memory bounded on large inputs.
    """
    m = coords.shape[0]
    adjacency: list[set[int]] = [set() for _ in range(m)]
    block = _row_block(m)
    for start in range(0, m, block):
        end = min(start + block, m)
        diff = coords[start:end, None, :] - coords[None, :, :]
        dist = np.sqrt((diff * diff).sum(axis=-1))  # (b, m)
        bond_cut = covalent[start:end, None] + covalent[None, :] + _BOND_TOLERANCE
        local_i = np.arange(start, end)[:, None]
        local_j = np.arange(m)[None, :]
        mask = (dist <= bond_cut) & (dist > _MIN_BOND_DISTANCE) & (local_j > local_i)
        rows, cols = np.where(mask)
        for a, j in zip(rows.tolist(), cols.tolist(), strict=True):
            i = start + a
            adjacency[i].add(j)
            adjacency[j].add(i)
    return adjacency


def _within_k_bonds(adjacency: list[set[int]], k: int) -> list[set[int]]:
    """For each atom, the set of atoms reachable in ``1..k`` bonds."""
    m = len(adjacency)
    reachable: list[set[int]] = [set() for _ in range(m)]
    if k <= 0:
        return reachable
    for src in range(m):
        seen = {src}
        frontier = deque([(src, 0)])
        while frontier:
            node, depth = frontier.popleft()
            if depth == k:
                continue
            for nb in adjacency[node]:
                if nb not in seen:
                    seen.add(nb)
                    reachable[src].add(nb)
                    frontier.append((nb, depth + 1))
    return reachable


def find_clashes(
    protein: Protein,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    bonded_separation: int = 3,
    include_hydrogens: bool = False,
) -> list[Clash]:
    """Find all steric clashes in ``protein``.

    A pair of atoms clashes when their van der Waals shells overlap by
    at least ``tolerance`` Å and they are more than ``bonded_separation``
    inferred covalent bonds apart (see the module docstring).

    Args:
        protein: The structure to check.
        tolerance: Minimum overlap (Å) to count as a clash. Smaller
            values flag more (softer) contacts.
        bonded_separation: Exclude atom pairs separated by at most this
            many inferred bonds (default 3 → 1-2, 1-3, 1-4 neighbours).
            Set to 0 to disable bonded exclusion.
        include_hydrogens: Consider hydrogen atoms too. Off by default
            since folding output usually omits them.

    Returns:
        Clashes sorted worst-first (largest overlap). Empty when the
        structure is clean or has fewer than two eligible atoms.
    """
    arr = protein.atom_array
    n = len(arr)
    if n < 2:
        return []

    elements_all = np.array([str(e).upper() for e in arr.element])
    keep = np.ones(n, dtype=bool)
    if not include_hydrogens:
        keep &= elements_all != "H"
    global_idx = np.where(keep)[0]
    if global_idx.size < 2:
        return []

    coords = np.asarray(arr.coords, dtype=np.float64)[global_idx]
    elements = elements_all[global_idx]
    vdw = _radii_for(elements, VDW_RADII, DEFAULT_VDW_RADIUS)
    covalent = _radii_for(elements, COVALENT_RADII, DEFAULT_COVALENT_RADIUS)
    res_index_all, res_labels = _residue_indices(protein)
    res_index = res_index_all[global_idx]

    m = global_idx.size
    excluded = (
        _within_k_bonds(_infer_adjacency(coords, covalent), bonded_separation)
        if bonded_separation > 0
        else [set[int]() for _ in range(m)]
    )

    clashes: list[Clash] = []
    block = _row_block(m)
    for start in range(0, m, block):
        end = min(start + block, m)
        sub = coords[start:end]
        diff = sub[:, None, :] - coords[None, :, :]
        dist = np.sqrt((diff * diff).sum(axis=-1))  # (b, m)
        vdw_sum = vdw[start:end, None] + vdw[None, :]  # (b, m)
        overlap = vdw_sum - dist  # (b, m)

        local_i = np.arange(start, end)[:, None]
        local_j = np.arange(m)[None, :]
        mask = (overlap >= tolerance) & (local_j > local_i)
        rows, cols = np.where(mask)
        for a, lj in zip(rows.tolist(), cols.tolist(), strict=True):
            li = start + a
            if lj in excluded[li]:
                continue
            clashes.append(
                Clash(
                    atom_i=int(global_idx[li]),
                    atom_j=int(global_idx[lj]),
                    element_i=str(elements[li]),
                    element_j=str(elements[lj]),
                    distance=float(dist[a, lj]),
                    vdw_sum=float(vdw_sum[a, lj]),
                    overlap=float(overlap[a, lj]),
                    residue_i=res_labels[int(res_index[li])],
                    residue_j=res_labels[int(res_index[lj])],
                )
            )

    clashes.sort(key=lambda c: c.overlap, reverse=True)
    return clashes


def _n_considered(protein: Protein, *, include_hydrogens: bool) -> int:
    """Count atoms that :func:`find_clashes` would consider."""
    arr = protein.atom_array
    if include_hydrogens:
        return len(arr)
    elements = np.array([str(e).upper() for e in arr.element])
    return int(np.count_nonzero(elements != "H"))


def clash_score(
    protein: Protein,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    bonded_separation: int = 3,
    include_hydrogens: bool = False,
) -> float:
    """Clashes per 1000 atoms — a single-number quality gate.

    Lower is better; a well-refined structure scores near zero. The
    denominator is the number of atoms actually considered (heavy atoms
    unless ``include_hydrogens`` is set), so the score is comparable
    across structures of different sizes. Returns 0.0 for an empty
    structure.
    """
    n = _n_considered(protein, include_hydrogens=include_hydrogens)
    if n == 0:
        return 0.0
    clashes = find_clashes(
        protein,
        tolerance=tolerance,
        bonded_separation=bonded_separation,
        include_hydrogens=include_hydrogens,
    )
    return 1000.0 * len(clashes) / n


def has_clashes(
    protein: Protein,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    bonded_separation: int = 3,
    include_hydrogens: bool = False,
) -> bool:
    """Whether ``protein`` has at least one steric clash."""
    return bool(
        find_clashes(
            protein,
            tolerance=tolerance,
            bonded_separation=bonded_separation,
            include_hydrogens=include_hydrogens,
        )
    )
