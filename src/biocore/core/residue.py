"""Residue — an amino acid (or other monomer) within a chain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from biocore.core.atom import Atom
    from biocore.core.chain import Chain


@dataclass
class Residue:
    """A residue (typically an amino acid) within a chain.

    Attributes:
        name: Three-letter residue code (e.g. ``"ALA"``, ``"GLY"``).
        seq_id: Author-assigned sequence number.
        insertion_code: PDB insertion code, ``""`` if none.
        atoms: List of :class:`Atom` objects in this residue.
        parent: Backreference to the containing :class:`Chain`.
    """

    name: str
    seq_id: int
    insertion_code: str = ""
    atoms: list[Atom] = field(default_factory=list)
    parent: Chain | None = None

    @property
    def one_letter(self) -> str:
        """Return the one-letter amino acid code for this residue.

        Raises:
            KeyError: If the residue name has no canonical one-letter mapping.
        """
        raise NotImplementedError

    def __getitem__(self, atom_name: str) -> Atom:
        """Look up an atom by name (e.g. ``residue["CA"]``)."""
        raise NotImplementedError
