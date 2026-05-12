"""Chain — a polypeptide (or polynucleotide) chain within a protein."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from biocore.core.protein import Protein
    from biocore.core.residue import Residue


@dataclass
class Chain:
    """A chain within a protein structure.

    Attributes:
        chain_id: Single-character chain identifier (e.g. ``"A"``).
        residues: List of :class:`Residue` objects in N-to-C order.
        parent: Backreference to the containing :class:`Protein`.
    """

    chain_id: str
    residues: list[Residue] = field(default_factory=list)
    parent: Protein | None = None

    @property
    def sequence(self) -> str:
        """One-letter amino-acid sequence for this chain."""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.residues)
