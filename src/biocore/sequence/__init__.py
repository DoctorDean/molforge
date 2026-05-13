"""Sequence-level operations: alignment, mutations, composition.

Examples:
    >>> from molforge.sequence import align, mutate
    >>> align("MKTV", "MKAV")
    >>> mutate(protein, chain="A", position=42, to="ALA")
"""

from __future__ import annotations

__all__ = ["align", "composition", "mutate"]


def align(seq1: str, seq2: str) -> object:
    """Pairwise sequence alignment. TODO: implement."""
    raise NotImplementedError


def mutate(protein: object, chain: str, position: int, to: str) -> object:
    """Apply a point mutation at ``chain:position`` to residue ``to``. TODO."""
    raise NotImplementedError


def composition(sequence: str) -> dict[str, int]:
    """Return the amino-acid composition of ``sequence``. TODO."""
    raise NotImplementedError
