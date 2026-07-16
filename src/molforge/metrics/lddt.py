"""lDDT (local Distance Difference Test) — alignment-free structure quality.

Reference: Mariani, V. et al. (2013) "lDDT: A local superposition-free
score for comparing protein structures and models using distance
difference tests." *Bioinformatics* 29: 2722-2728.

lDDT measures the fraction of pairwise inter-atom distances that are
preserved between a model and a reference structure, *without
requiring superposition*. This makes it ideal for:

  - Per-residue confidence scores (what AlphaFold's pLDDT and
    ESMFold's pLDDT estimate without seeing the reference).
  - Local quality assessment on flexible / multi-domain structures
    where a single global alignment is misleading.

Algorithm:
  1. Enumerate all pairs of atoms in the reference that are within
     ``inclusion_radius`` (default 15 Å) *and belong to different
     residues*. These are the "reference distances" the model is graded
     against. (Intra-residue pairs are excluded — they're fixed by
     stereochemistry, not by the fold.)
  2. For each reference pair, compute the absolute difference between
     the reference distance and the corresponding model distance.
  3. Count the pair as "preserved" if the difference is below at least
     one of the tolerance thresholds (default 0.5/1/2/4 Å — matching
     the lDDT paper). The score per pair is the fraction of thresholds
     it passes (0, 0.25, 0.5, 0.75, or 1.0).
  4. Average over the pairs involving each residue (per-residue scores);
     the global score is the mean of the per-residue scores.

``atom_set`` selects which atoms are compared:

  - ``"ca"`` (default) — one Cα per residue. Fast, and what most folding
    benchmarks report. With one atom per residue every pair is
    inter-residue, so the intra-residue exclusion is a no-op here.
  - ``"heavy"`` — all non-hydrogen atoms (the canonical all-atom lDDT).
    Sensitive to side-chain placement, so it is stricter and lower than
    the Cα score for the same model. Requires the model and reference to
    contain the *same heavy atoms in the same order* (as predicted-vs-
    native of one sequence normally do).

The result is a value in ``[0, 1]``; higher = more accurate local
geometry. For AlphaFold-quality predictions, lDDT > 0.7 is typical for
ordered regions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from molforge.core import Protein

_DEFAULT_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)

AtomSet = Literal["ca", "heavy"]


def lddt(
    model: Protein,
    reference: Protein,
    *,
    atom_set: AtomSet = "ca",
    inclusion_radius: float = 15.0,
    thresholds: tuple[float, ...] = _DEFAULT_THRESHOLDS,
) -> float:
    """Global lDDT score (alignment-free).

    Args:
        model: Predicted structure.
        reference: Native / target structure.
        atom_set: ``"ca"`` (default, one Cα per residue) or ``"heavy"``
            (all non-hydrogen atoms — the canonical all-atom lDDT, which
            also grades side-chain placement).
        inclusion_radius: Reference-distance cutoff in Å. Only pairs
            of atoms within this distance in the *reference* are
            included in the score.
        thresholds: Tolerance thresholds. Default ``(0.5, 1, 2, 4)``
            matches the lDDT paper.

    Returns:
        lDDT score in ``[0, 1]`` (the mean of the per-residue scores).
        Higher = better.
    """
    per_residue = lddt_per_residue(
        model,
        reference,
        atom_set=atom_set,
        inclusion_radius=inclusion_radius,
        thresholds=thresholds,
    )
    # Drop NaN entries (residues with no included pairs) before averaging.
    valid = per_residue[~np.isnan(per_residue)]
    if valid.size == 0:
        return 0.0
    return float(valid.mean())


def lddt_per_residue(
    model: Protein,
    reference: Protein,
    *,
    atom_set: AtomSet = "ca",
    inclusion_radius: float = 15.0,
    thresholds: tuple[float, ...] = _DEFAULT_THRESHOLDS,
) -> NDArray[np.float32]:
    """Per-residue lDDT (the per-residue confidence pLDDT estimates).

    Args:
        model: Predicted structure.
        reference: Native / target structure.
        atom_set: see :func:`lddt`. For ``"heavy"`` a residue's score
            aggregates every non-hydrogen atom it contains.
        inclusion_radius: see :func:`lddt`.
        thresholds: see :func:`lddt`.

    Returns:
        ``(n_residues,)`` float32 array. Residues with no pair partners
        within ``inclusion_radius`` get ``NaN``.

    Raises:
        ValueError: If the model and reference don't present matched atoms
            (equal Cα counts for ``"ca"``; equal heavy-atom counts in the
            same order for ``"heavy"``), or ``atom_set`` is unrecognized.

    Note:
        The comparison is dense (``O(k²)`` in the atom count ``k``), so
        ``atom_set="heavy"`` on very large structures is memory-heavy.
    """
    if atom_set == "ca":
        from molforge.metrics.tm import _ca_coords

        m_coords = _ca_coords(model)
        r_coords = _ca_coords(reference)
        if m_coords.shape != r_coords.shape:
            raise ValueError(
                f"lDDT requires matched residue lists: model has "
                f"{m_coords.shape[0]} CAs, reference has {r_coords.shape[0]}."
            )
        n = m_coords.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        # One atom per residue: each atom is its own residue group.
        atom_residue = np.arange(n)
        n_residues = n
    elif atom_set == "heavy":
        m_coords, _, _ = _heavy_atoms(model)
        r_coords, atom_residue, n_residues = _heavy_atoms(reference)
        if m_coords.shape != r_coords.shape:
            raise ValueError(
                f"all-atom lDDT requires matched heavy atoms in the same order: "
                f"model has {m_coords.shape[0]}, reference has {r_coords.shape[0]}."
            )
        if n_residues == 0:
            return np.zeros(0, dtype=np.float32)
    else:
        raise ValueError(f"unknown atom_set {atom_set!r}; expected 'ca' or 'heavy'.")

    return _lddt_core(
        m_coords,
        r_coords,
        atom_residue,
        n_residues,
        inclusion_radius=inclusion_radius,
        thresholds=thresholds,
    )


# ---------- internals ----------


def _heavy_atoms(protein: Protein) -> tuple[NDArray[np.float64], NDArray[np.int_], int]:
    """Heavy-atom coords, their residue-group index, and the residue count.

    Atoms are returned in residue order; ``atom_residue[k]`` is the 0-based
    ordinal of the residue that atom ``k`` belongs to. Only protein residues
    with at least one non-hydrogen atom are included.
    """
    arr = protein.atom_array
    coords_parts: list[NDArray[np.float64]] = []
    residue_parts: list[NDArray[np.int_]] = []
    r = 0
    for sl in arr.iter_residue_slices():
        if str(arr.entity_type[sl.start]) != "protein":
            continue
        heavy = np.where(arr.element[sl] != "H")[0]
        if heavy.size == 0:
            continue
        coords_parts.append(arr.coords[sl][heavy].astype(np.float64))
        residue_parts.append(np.full(heavy.size, r, dtype=np.int_))
        r += 1
    if not coords_parts:
        return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.int_), 0
    return np.concatenate(coords_parts), np.concatenate(residue_parts), r


def _lddt_core(
    m_coords: NDArray[np.float64],
    r_coords: NDArray[np.float64],
    atom_residue: NDArray[np.int_],
    n_residues: int,
    *,
    inclusion_radius: float,
    thresholds: tuple[float, ...],
) -> NDArray[np.float32]:
    """Per-residue lDDT over pre-selected, residue-labelled atoms.

    ``atom_residue`` maps each of the ``k`` atoms to a residue ordinal in
    ``[0, n_residues)``. Pairs are considered when they're within
    ``inclusion_radius`` in the reference and belong to *different*
    residues (a no-op for the one-atom-per-residue Cα case).
    """
    ref_dist = np.linalg.norm(r_coords[None, :, :] - r_coords[:, None, :], axis=-1)
    mod_dist = np.linalg.norm(m_coords[None, :, :] - m_coords[:, None, :], axis=-1)

    inter_residue = atom_residue[:, None] != atom_residue[None, :]
    mask = (ref_dist < inclusion_radius) & (ref_dist > 0) & inter_residue

    diff = np.abs(ref_dist - mod_dist)
    threshold_pass = np.stack([(diff < t).astype(np.float64) for t in thresholds], axis=-1)
    pair_score = threshold_pass.mean(axis=-1)  # (k, k)

    out = np.full(n_residues, np.nan, dtype=np.float32)
    for res in range(n_residues):
        atoms = np.where(atom_residue == res)[0]
        sub_mask = mask[atoms]  # (n_atoms_in_res, k)
        if sub_mask.any():
            out[res] = float(pair_score[atoms][sub_mask].mean())
    return out
