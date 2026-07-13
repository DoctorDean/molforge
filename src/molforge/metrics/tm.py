"""TM-score (Template Modeling score) for protein structure comparison.

Reference: Zhang, Y. & Skolnick, J. (2004) "Scoring function for automated
assessment of protein structure template quality." *Proteins* 57: 702-710.

TM-score measures the structural similarity between two protein
structures, length-normalized so values are comparable across proteins
of different sizes. The formula is::

    TM = (1/L_target) * max over alignments of sum_i 1 / (1 + (d_i / d0)^2)

where ``L_target`` is the length of the *reference* (target) structure,
``d_i`` is the distance between aligned residue i in the model and its
counterpart in the reference after optimal superposition, and ``d0``
is a length-dependent scaling factor designed to make scores
comparable across sizes.

TM-score interpretation (Zhang & Skolnick 2005):
  - **< 0.17** — random structural similarity
  - **0.17 - 0.5** — uncertain (similarity is possible but not assured)
  - **> 0.5** — generally the same fold
  - **> 0.85** — essentially the same structure

Caveats vs. the reference TM-align binary:
  - The full TM-align algorithm searches for the **optimal sequence
    alignment** between two structures. molforge's implementation
    assumes the **input residues are already in correspondence** —
    same number, same order. For comparing two structures of the
    same sequence (e.g. predicted vs. native, two homology models),
    this is exactly what you want.
  - For comparing structures with different sequences or lengths,
    you need a structural-alignment step first; see TM-align proper
    (https://zhanggroup.org/TM-align/).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from molforge.structure.superposition import superpose

if TYPE_CHECKING:
    from collections.abc import Callable

    from molforge.core import Protein


# TM-score and GDT are both defined as the *maximum* of a residue sum over
# rigid-body superpositions. A single Kabsch fit only minimizes RMSD, which
# is dominated by the most-displaced residues and systematically
# underestimates both metrics when part of the structure is displaced (a
# hinge or domain motion). The search below follows the TM-score / LGA
# heuristic: seed superpositions on contiguous fragments, then iteratively
# re-superpose on the residues that currently fit, keeping the best score.
_MAX_REFINE_ITERS = 20


def _d0(length: int) -> float:
    """Length-dependent normalization factor from Zhang & Skolnick 2004.

    For L >= 21:  d0 = 1.24 * (L - 15)^(1/3) - 1.8
    For L < 21:   d0 = 0.5 (floor used in the reference implementation)
    """
    if length < 21:
        return 0.5
    return float(1.24 * (length - 15) ** (1.0 / 3.0) - 1.8)


def _ca_coords(protein: Protein) -> np.ndarray:
    """Return ``(n_residues, 3)`` CA coordinates for protein residues."""
    arr = protein.atom_array
    out = []
    for sl in arr.iter_residue_slices():
        if str(arr.entity_type[sl.start]) != "protein":
            continue
        names = arr.atom_name[sl]
        ca_idx = np.where(names == "CA")[0]
        if not ca_idx.size:
            continue
        out.append(arr.coords[sl][ca_idx[0]])
    return np.asarray(out, dtype=np.float64)


def _seed_lengths(length: int) -> list[int]:
    """Initial fragment lengths for the superposition search.

    The full length plus successively halved windows down to a short
    floor. Shorter seeds let the search lock onto a well-superposable
    sub-domain and disregard a displaced remainder.
    """
    lengths: list[int] = []
    f = length
    while f > 4:
        lengths.append(f)
        f //= 2
    lengths.append(min(4, length))
    return sorted({min(x, length) for x in lengths}, reverse=True)


def _optimal_superposition_score(
    m_coords: np.ndarray,
    r_coords: np.ndarray,
    *,
    select_cutoff: float,
    score_fn: Callable[[np.ndarray], float],
) -> float:
    """Maximize ``score_fn`` over fragment-seeded superpositions.

    For each seed fragment (a contiguous residue window) the model is
    superposed onto the reference on that fragment, then iteratively
    re-superposed on the residues currently within ``select_cutoff`` Å.
    ``score_fn`` (a function of the per-residue CA distance array) is
    evaluated at every superposition visited, and the maximum is returned.
    The full-length seed's first fit is the plain Kabsch superposition, so
    the result is always >= the single-Kabsch value.
    """
    length = r_coords.shape[0]
    best = 0.0
    for l_ini in _seed_lengths(length):
        step = max(1, l_ini // 2)
        for start in range(0, length - l_ini + 1, step):
            sel = np.arange(start, start + l_ini)
            for _ in range(_MAX_REFINE_ITERS):
                sp = superpose(m_coords[sel], r_coords[sel])
                aligned = (sp.rotation @ m_coords.T).T + sp.translation
                distances = np.linalg.norm(aligned - r_coords, axis=1)
                best = max(best, score_fn(distances))
                nxt = np.where(distances < select_cutoff)[0]
                if nxt.size < 3 or np.array_equal(nxt, sel):
                    break  # too few to superpose, or converged to a fixed point
                sel = nxt
    return best


def tm_score(
    model: Protein,
    reference: Protein,
    *,
    normalize_by: str = "reference",
) -> float:
    """Compute TM-score between two CA-aligned structures.

    Args:
        model: The model (predicted / candidate) structure.
        reference: The reference (target / native) structure.
        normalize_by: Length used to compute ``d0`` and as the
            denominator in the TM formula:

            - ``"reference"`` (default) — match the reference's length.
              Use this for "how good is the prediction relative to the
              target".
            - ``"model"`` — match the model's length. Use when you want
              "how much of the model agrees with the reference".
            - ``"shorter"`` / ``"longer"`` — the convention in some
              papers; uses min/max of the two lengths.

    Returns:
        TM-score in ``[0, 1]``. Higher is better.

    Raises:
        ValueError: If the structures don't have equal CA counts.
    """
    m_coords = _ca_coords(model)
    r_coords = _ca_coords(reference)
    if m_coords.shape != r_coords.shape:
        raise ValueError(
            f"TM-score requires matched residue lists: model has "
            f"{m_coords.shape[0]} CAs, reference has {r_coords.shape[0]}. "
            "For sequence-mismatched structures, perform a structural "
            "alignment first or use the TM-align reference implementation."
        )
    if m_coords.shape[0] < 3:
        raise ValueError(f"TM-score requires at least 3 residues, got {m_coords.shape[0]}")

    if normalize_by == "reference":
        norm_length = r_coords.shape[0]
    elif normalize_by == "model":
        norm_length = m_coords.shape[0]
    elif normalize_by == "shorter":
        norm_length = min(m_coords.shape[0], r_coords.shape[0])
    elif normalize_by == "longer":
        norm_length = max(m_coords.shape[0], r_coords.shape[0])
    else:
        raise ValueError(
            f"unknown normalize_by {normalize_by!r}; "
            "expected 'reference', 'model', 'shorter', or 'longer'"
        )

    d0 = _d0(norm_length)

    # TM-score is the MAXIMUM of the residue sum over superpositions, not
    # its value at the RMSD-minimizing (Kabsch) fit. Search fragment-seeded
    # superpositions; the selection cutoff follows the reference program's
    # d0_search (d0 clamped to [4.5, 8.0] Å).
    d0_search = min(max(d0, 4.5), 8.0)

    def _tm(distances: np.ndarray) -> float:
        return float((1.0 / (1.0 + (distances / d0) ** 2)).sum() / norm_length)

    return _optimal_superposition_score(
        m_coords, r_coords, select_cutoff=d0_search, score_fn=_tm
    )
