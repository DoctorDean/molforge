"""Atom — a lightweight view onto a single atom in an :class:`AtomArray`.

An ``Atom`` does not own its data. It holds a reference to the parent
``AtomArray`` and an integer index. All property accesses read from the
underlying arrays; all mutations write through.

This makes ``Atom`` cheap to create (no copy) and guarantees consistency
with the linear view.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from molforge.core.atom_array import AtomArray
    from molforge.core.residue import Residue

from molforge.core.constants import PROTEIN_BACKBONE_ATOMS


class Atom:
    """View of a single atom in an :class:`AtomArray`.

    Attributes are read/written through to the underlying array, so
    mutating an ``Atom`` mutates the source-of-truth representation.
    """

    __slots__ = ("_array", "_index", "_parent")

    def __init__(
        self,
        array: AtomArray,
        index: int,
        *,
        parent: Residue | None = None,
    ) -> None:
        if not 0 <= index < len(array):
            raise IndexError(f"index {index} out of bounds for array of length {len(array)}")
        self._array = array
        self._index = index
        self._parent = parent

    # ------------------------------------------------------------------
    # Identity / context
    # ------------------------------------------------------------------
    @property
    def index(self) -> int:
        """The atom's index into the parent :class:`AtomArray`."""
        return self._index

    @property
    def parent(self) -> Residue | None:
        """The containing residue, if known."""
        return self._parent

    # ------------------------------------------------------------------
    # Field accessors (read/write through to the array)
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return str(self._array.atom_name[self._index])

    @name.setter
    def name(self, value: str) -> None:
        self._array.atom_name[self._index] = value

    @property
    def element(self) -> str:
        return str(self._array.element[self._index])

    @element.setter
    def element(self, value: str) -> None:
        self._array.element[self._index] = value

    @property
    def coord(self) -> NDArray[np.float32]:
        """The atom's 3-D coordinate as a (3,) float32 view (mutable)."""
        return self._array.coords[self._index]

    @coord.setter
    def coord(self, value: NDArray[np.float32]) -> None:
        self._array.coords[self._index] = value

    @property
    def b_factor(self) -> float:
        return float(self._array.b_factor[self._index])

    @b_factor.setter
    def b_factor(self, value: float) -> None:
        self._array.b_factor[self._index] = value

    @property
    def occupancy(self) -> float:
        return float(self._array.occupancy[self._index])

    @occupancy.setter
    def occupancy(self, value: float) -> None:
        self._array.occupancy[self._index] = value

    @property
    def charge(self) -> float:
        return float(self._array.charge[self._index])

    @charge.setter
    def charge(self, value: float) -> None:
        self._array.charge[self._index] = value

    @property
    def serial(self) -> int:
        return int(self._array.serial[self._index])

    @serial.setter
    def serial(self, value: int) -> None:
        self._array.serial[self._index] = value

    @property
    def altloc(self) -> str:
        return str(self._array.altloc[self._index])

    @altloc.setter
    def altloc(self, value: str) -> None:
        self._array.altloc[self._index] = value

    @property
    def record_type(self) -> str:
        return str(self._array.record_type[self._index])

    # ------------------------------------------------------------------
    # Derived / convenience
    # ------------------------------------------------------------------
    @property
    def is_backbone(self) -> bool:
        """True if this is a standard protein backbone atom (N, CA, C, O, OXT)."""
        return self.name in PROTEIN_BACKBONE_ATOMS

    @property
    def is_hetero(self) -> bool:
        """True if this atom comes from a HETATM record."""
        return self.record_type == "HETATM"

    def __repr__(self) -> str:
        return f"Atom(name={self.name!r}, element={self.element!r}, index={self._index})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Atom):
            return NotImplemented
        return self._array is other._array and self._index == other._index

    def __hash__(self) -> int:
        return hash((id(self._array), self._index))
