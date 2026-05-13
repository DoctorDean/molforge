"""Chain — a view over the residues of a single chain.

A chain corresponds to a contiguous slice of an :class:`AtomArray` sharing
the same ``(chain_id, model_id)``. Residue boundaries within the chain
are resolved on demand from the underlying array's
:attr:`~molforge.core.atom_array.AtomArray.residue_starts` cache.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from molforge.core.atom_array import AtomArray
    from molforge.core.protein import Protein

from molforge.core.residue import Residue


class Chain:
    """View over a chain's atoms inside an :class:`AtomArray`."""

    __slots__ = ("_array", "_end", "_parent", "_start")

    def __init__(
        self,
        array: AtomArray,
        start: int,
        end: int,
        *,
        parent: Protein | None = None,
    ) -> None:
        if not 0 <= start < end <= len(array):
            raise IndexError(
                f"invalid chain slice [{start}, {end}) for array of length {len(array)}"
            )
        self._array = array
        self._start = int(start)
        self._end = int(end)
        self._parent = parent

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    @property
    def chain_id(self) -> str:
        return str(self._array.chain_id[self._start])

    @property
    def model_id(self) -> int:
        return int(self._array.model_id[self._start])

    @property
    def parent(self) -> Protein | None:
        return self._parent

    # ------------------------------------------------------------------
    # Residue access
    # ------------------------------------------------------------------
    def _residue_slice_bounds(self) -> NDArray[np.int32]:
        """Return the global residue-start indices that fall within this chain."""
        starts = self._array.residue_starts
        mask = (starts >= self._start) & (starts < self._end)
        return starts[mask]

    @property
    def residues(self) -> list[Residue]:
        """All residues in this chain, in N-to-C order."""
        bounds = self._residue_slice_bounds()
        out: list[Residue] = []
        for i, s in enumerate(bounds):
            e = int(bounds[i + 1]) if i + 1 < len(bounds) else self._end
            out.append(Residue(self._array, int(s), e, parent=self))
        return out

    def __iter__(self):  # type: ignore[no-untyped-def]
        bounds = self._residue_slice_bounds()
        for i, s in enumerate(bounds):
            e = int(bounds[i + 1]) if i + 1 < len(bounds) else self._end
            yield Residue(self._array, int(s), e, parent=self)

    def __len__(self) -> int:
        return int(self._residue_slice_bounds().shape[0])

    def __getitem__(self, key: int | tuple[int, str]) -> Residue:
        """Look up a residue.

        - ``chain[42]`` returns the residue with ``seq_id == 42`` (no insertion code).
        - ``chain[(42, "A")]`` returns the residue with ``seq_id == 42`` and
          ``insertion_code == "A"``.

        Raises:
            KeyError: If no matching residue exists.
        """
        if isinstance(key, tuple):
            seq_id, ins = key
        else:
            seq_id, ins = key, ""
        for res in self:
            if res.seq_id == seq_id and res.insertion_code == ins:
                return res
        raise KeyError(f"chain {self.chain_id!r} has no residue {seq_id}{ins or ''}")

    @property
    def sequence(self) -> str:
        """One-letter sequence for this chain (standard AAs + non-canonical mappings).

        Non-amino-acid residues (ligands, water, ions) are skipped.
        Unknown residues become ``"X"``.
        """
        out: list[str] = []
        for res in self:
            if res.entity_type not in {"protein", "dna", "rna"}:
                continue
            out.append(res.one_letter)
        return "".join(out)

    @property
    def n_atoms(self) -> int:
        return self._end - self._start

    @property
    def n_residues(self) -> int:
        return len(self)

    @property
    def coords(self) -> NDArray[np.float32]:
        """All atom coordinates for this chain, shape ``(n_atoms, 3)``."""
        return self._array.coords[self._start : self._end]

    @property
    def slice(self) -> slice:
        return slice(self._start, self._end)

    def __repr__(self) -> str:
        return f"Chain(chain_id={self.chain_id!r}, n_residues={len(self)}, n_atoms={self.n_atoms})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Chain):
            return NotImplemented
        return (
            self._array is other._array and self._start == other._start and self._end == other._end
        )

    def __hash__(self) -> int:
        return hash((id(self._array), self._start, self._end))
