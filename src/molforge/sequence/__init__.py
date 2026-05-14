"""Sequence-level operations: alignment, mutations, composition, properties.

What's here:
    - :func:`align` / :func:`needleman_wunsch` / :func:`smith_waterman`
      — pairwise alignment with affine gaps and BLOSUM62/PAM250 matrices.
    - :func:`identity` — convenience for "what's the sequence identity"
    - :class:`Mutation`, :func:`apply_mutation`, :func:`apply_mutations`,
      :func:`mutate_protein` — point mutations on sequences and Proteins.
    - :func:`composition`, :func:`molecular_weight`, :func:`gravy`,
      :func:`aromaticity`, :func:`length` — sequence properties.

For sequence I/O see :mod:`molforge.io` (FASTA).
"""

from __future__ import annotations

from molforge.sequence.alignment import (
    Alignment,
    align,
    identity,
    needleman_wunsch,
    smith_waterman,
)
from molforge.sequence.composition import (
    aromaticity,
    composition,
    gravy,
    length,
    molecular_weight,
)
from molforge.sequence.matrices import (
    BLOSUM62,
    PAM250,
    available_matrices,
    get_matrix,
)
from molforge.sequence.mutations import (
    Mutation,
    apply_mutation,
    apply_mutations,
    mutate_protein,
    parse_mutations,
)

__all__ = [  # noqa: RUF022 — grouped by concern
    # Alignment
    "align",
    "needleman_wunsch",
    "smith_waterman",
    "identity",
    "Alignment",
    # Substitution matrices
    "BLOSUM62",
    "PAM250",
    "get_matrix",
    "available_matrices",
    # Mutations
    "Mutation",
    "apply_mutation",
    "apply_mutations",
    "parse_mutations",
    "mutate_protein",
    # Composition / properties
    "composition",
    "molecular_weight",
    "gravy",
    "aromaticity",
    "length",
]
