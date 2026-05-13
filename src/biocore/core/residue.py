"""Residue — a view onto a contiguous group of atoms in an :class:`AtomArray`.

A residue corresponds to a unique
``(chain_id, residue_id, insertion_code, model_id)`` tuple, materialized
as a slice ``[start, end)`` into the parent array.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from molforge.core.atom_array import AtomArray
    from molforge.core.chain import Chain

from molforge.core.atom import Atom
from molforge.core.constants import is_standard_amino_acid, is_water, three_to_one


class Residue:
    """View over a residue's atoms inside an :class:`AtomArray`."""

    __slots__ = ("_array", "_end", "_parent", "_start")

    def __init__(
        self,
        array: AtomArray,
        start: int,
        end: int,
        *,
        parent: Chain | None = None,
    ) -> None:
        if not 0 <= start < end <= len(array):
            raise IndexError(
                f"invalid residue slice [{start}, {end}) for array of length {len(array)}"
            )
        self._array = array
        self._start = int(start)
        self._end = int(end)
        self._parent = parent

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        """Three-letter residue code (e.g. ``"ALA"``)."""
        return str(self._array.residue_name[self._start])

    @property
    def seq_id(self) -> int:
        """Author-assigned residue sequence number."""
        return int(self._array.residue_id[self._start])

    @property
    def insertion_code(self) -> str:
        return str(self._array.insertion_code[self._start])

    @property
    def chain_id(self) -> str:
        return str(self._array.chain_id[self._start])

    @property
    def model_id(self) -> int:
        return int(self._array.model_id[self._start])

    @property
    def entity_type(self) -> str:
        """e.g. ``"protein"``, ``"dna"``, ``"rna"``, ``"ligand"``, ``"water"``, ``"ion"``."""
        return str(self._array.entity_type[self._start])

    @property
    def parent(self) -> Chain | None:
        return self._parent

    # ------------------------------------------------------------------
    # Atom access
    # ------------------------------------------------------------------
    @property
    def atoms(self) -> list[Atom]:
        """All atoms in this residue, as :class:`Atom` views."""
        return [Atom(self._array, i, parent=self) for i in range(self._start, self._end)]

    def __iter__(self):  # type: ignore[no-untyped-def]
        for i in range(self._start, self._end):
            yield Atom(self._array, i, parent=self)

    def __len__(self) -> int:
        return self._end - self._start

    def __getitem__(self, atom_name: str) -> Atom:
        """Look up an atom by name (e.g. ``residue["CA"]``).

        Raises:
            KeyError: If no atom with that name exists in this residue.
        """
        names = self._array.atom_name[self._start : self._end]
        matches = np.nonzero(names == atom_name)[0]
        if matches.size == 0:
            raise KeyError(f"residue {self.name} {self.seq_id} has no atom named {atom_name!r}")
        if matches.size > 1:
            # Pick the highest-occupancy match (handles altlocs).
            occ = self._array.occupancy[self._start : self._end][matches]
            best = matches[int(np.argmax(occ))]
            return Atom(self._array, self._start + int(best), parent=self)
        return Atom(self._array, self._start + int(matches[0]), parent=self)

    def has_atom(self, atom_name: str) -> bool:
        names = self._array.atom_name[self._start : self._end]
        return bool(np.any(names == atom_name))

    @property
    def coords(self) -> NDArray[np.float32]:
        """All atom coordinates for this residue, shape ``(n_atoms, 3)``."""
        return self._array.coords[self._start : self._end]

    @property
    def slice(self) -> slice:
        """Underlying array slice covering this residue's atoms."""
        return slice(self._start, self._end)

    # ------------------------------------------------------------------
    # Derived / convenience
    # ------------------------------------------------------------------
    @property
    def one_letter(self) -> str:
        """Single-letter amino-acid (or nucleotide) code; ``"X"`` for unknown."""
        return three_to_one(self.name)

    @property
    def is_standard_amino_acid(self) -> bool:
        return is_standard_amino_acid(self.name)

    @property
    def is_water(self) -> bool:
        return is_water(self.name)

    @property
    def is_hetero(self) -> bool:
        """True if any atom in this residue is a HETATM record."""
        return bool(np.any(self._array.record_type[self._start : self._end] == "HETATM"))

    def __repr__(self) -> str:
        return (
            f"Residue(name={self.name!r}, seq_id={self.seq_id}, "
            f"chain={self.chain_id!r}, n_atoms={len(self)})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Residue):
            return NotImplemented
        return (
            self._array is other._array and self._start == other._start and self._end == other._end
        )

    def __hash__(self) -> int:
        return hash((id(self._array), self._start, self._end))
