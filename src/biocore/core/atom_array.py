"""AtomArray — the canonical, flat, NumPy-backed representation.

This is the *source of truth* for a protein's atomic data. The hierarchical
classes (`Atom`, `Residue`, `Chain`, `Protein`) are thin accessors that hold
an :class:`AtomArray` reference plus index slices into it.

Design notes:
- All per-atom fields are parallel NumPy arrays of length ``N``.
- Coordinates are ``(N, 3)`` float32 — float32 is enough for any single PDB
  and halves the memory of float64. Promote on demand if you need it.
- String fields use NumPy unicode dtypes (``"U1"``, ``"U3"``, ``"U4"``)
  rather than Python ``object`` arrays — keeps memory predictable and
  enables vectorized comparisons.
- The class exposes a residue-/chain-boundary index (`_chain_starts`,
  `_residue_starts`) that's built lazily and cached. Any mutation that
  changes residue/chain identity invalidates the cache.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Iterable

# Public type aliases for users.
FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int32]
StrArray = NDArray[np.str_]
BoolArray = NDArray[np.bool_]


# Field schema — single source of truth for column names, dtypes, and defaults.
# Used by AtomArray constructors and IO routines so adding a column is a
# one-line change here.
_FIELD_SCHEMA: dict[str, tuple[str, object]] = {
    # name              dtype       default
    "coords": ("float32", 0.0),  # special-cased: shape (N, 3)
    "element": ("U2", ""),
    "atom_name": ("U4", ""),
    "residue_name": ("U3", ""),
    "residue_id": ("int32", 0),
    "insertion_code": ("U1", ""),
    "chain_id": ("U4", ""),  # 4 chars to support mmCIF auth_asym_id
    "b_factor": ("float32", 0.0),
    "occupancy": ("float32", 1.0),
    "charge": ("float32", 0.0),
    "serial": ("int32", 0),
    "record_type": ("U6", "ATOM"),  # "ATOM" or "HETATM"
    "entity_type": ("U8", "protein"),  # protein, dna, rna, ligand, water, ion, other
    "altloc": ("U1", ""),
    "model_id": ("int32", 0),  # for multi-model NMR / trajectories
}

ATOM_FIELDS: tuple[str, ...] = tuple(_FIELD_SCHEMA.keys())


class AtomArray:
    """Flat, NumPy-backed array of atoms.

    This is the canonical representation; hierarchical views read from
    these arrays. All per-atom fields have shape ``(N,)`` except
    ``coords`` which has shape ``(N, 3)``.

    Example:
        >>> aa = AtomArray.empty(3)
        >>> aa.element[:] = ["C", "N", "O"]
        >>> aa.coords[0] = [1.0, 2.0, 3.0]
        >>> len(aa)
        3
    """

    __slots__ = (
        "_chain_starts_cache",
        "_residue_starts_cache",
        "altloc",
        "atom_name",
        "b_factor",
        "chain_id",
        "charge",
        "coords",
        "element",
        "entity_type",
        "insertion_code",
        "model_id",
        "occupancy",
        "record_type",
        "residue_id",
        "residue_name",
        "serial",
    )

    coords: FloatArray
    element: StrArray
    atom_name: StrArray
    residue_name: StrArray
    residue_id: IntArray
    insertion_code: StrArray
    chain_id: StrArray
    b_factor: FloatArray
    occupancy: FloatArray
    charge: FloatArray
    serial: IntArray
    record_type: StrArray
    entity_type: StrArray
    altloc: StrArray
    model_id: IntArray

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, n: int = 0) -> None:
        """Create an empty array of ``n`` atoms, all fields at default values."""
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        self.coords = np.zeros((n, 3), dtype=np.float32)
        for field, (dtype, default) in _FIELD_SCHEMA.items():
            if field == "coords":
                continue
            arr = np.empty(n, dtype=dtype)
            arr[:] = default
            object.__setattr__(self, field, arr)
        self._chain_starts_cache = None
        self._residue_starts_cache = None

    @classmethod
    def empty(cls, n: int) -> AtomArray:
        """Alias for ``AtomArray(n)`` — more readable at call sites."""
        return cls(n)

    @classmethod
    def from_dict(cls, data: dict[str, NDArray]) -> AtomArray:
        """Construct from a dict of equal-length arrays.

        Args:
            data: Mapping field-name -> array. Must include ``coords``;
                missing fields are filled with their schema defaults.

        Raises:
            KeyError: If ``coords`` is missing.
            ValueError: If array lengths disagree.
        """
        if "coords" not in data:
            raise KeyError("`coords` is required to construct an AtomArray")
        n = data["coords"].shape[0]
        for name, arr in data.items():
            if name == "coords":
                continue
            if arr.shape[0] != n:
                raise ValueError(f"Field {name!r} has length {arr.shape[0]}, expected {n}")
        out = cls(n)
        for name, arr in data.items():
            if name not in _FIELD_SCHEMA:
                raise KeyError(f"Unknown field {name!r}; valid: {ATOM_FIELDS}")
            setattr(out, name, np.asarray(arr))
        out._invalidate_cache()
        return out

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return int(self.coords.shape[0])

    def __repr__(self) -> str:
        return f"AtomArray(n_atoms={len(self)})"

    def __getitem__(self, key: int | slice | NDArray) -> AtomArray:
        """Slice or fancy-index the array; returns a new AtomArray (copy)."""
        if isinstance(key, int):
            # Single-atom selection still returns an AtomArray of length 1
            # so the API stays uniform. Use `.to_atom(i)` for a hierarchical Atom view.
            key = slice(key, key + 1)
        out = AtomArray(0)
        out.coords = np.ascontiguousarray(self.coords[key])
        for field in ATOM_FIELDS:
            if field == "coords":
                continue
            setattr(out, field, np.ascontiguousarray(getattr(self, field)[key]))
        out._invalidate_cache()
        return out

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------
    def _invalidate_cache(self) -> None:
        """Drop the residue / chain boundary caches. Call after mutations."""
        object.__setattr__(self, "_chain_starts_cache", None)
        object.__setattr__(self, "_residue_starts_cache", None)

    def append(self, other: AtomArray) -> AtomArray:
        """Return a new array with ``other`` concatenated after this one."""
        if not isinstance(other, AtomArray):
            raise TypeError(f"expected AtomArray, got {type(other).__name__}")
        out = AtomArray(0)
        out.coords = np.concatenate([self.coords, other.coords], axis=0)
        for field in ATOM_FIELDS:
            if field == "coords":
                continue
            out_arr = np.concatenate([getattr(self, field), getattr(other, field)])
            setattr(out, field, out_arr)
        out._invalidate_cache()
        return out

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def select(self, mask: BoolArray) -> AtomArray:
        """Return a new AtomArray containing only atoms where ``mask`` is True.

        Args:
            mask: Boolean array of length ``len(self)``.
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (len(self),):
            raise ValueError(f"mask shape {mask.shape} does not match atom count ({len(self)},)")
        return self[mask]

    def where(self, **filters: object) -> BoolArray:
        """Build a boolean mask from equality filters on any field.

        Example:
            >>> mask = aa.where(chain_id="A", atom_name="CA")
            >>> ca_atoms = aa.select(mask)
        """
        mask = np.ones(len(self), dtype=bool)
        for field, value in filters.items():
            if field not in _FIELD_SCHEMA:
                raise KeyError(f"Unknown field {field!r}; valid: {ATOM_FIELDS}")
            arr = getattr(self, field)
            if isinstance(value, (list, tuple, set, np.ndarray)):
                mask &= np.isin(arr, list(value))
            else:
                mask &= arr == value
        return mask

    # ------------------------------------------------------------------
    # Boundary indices (chains / residues)
    # ------------------------------------------------------------------
    @property
    def chain_starts(self) -> IntArray:
        """Indices of the first atom of each chain, in order.

        A chain boundary is any change in ``chain_id`` or ``model_id``.
        """
        if self._chain_starts_cache is None:
            object.__setattr__(self, "_chain_starts_cache", self._compute_chain_starts())
        return self._chain_starts_cache  # type: ignore[return-value]

    @property
    def residue_starts(self) -> IntArray:
        """Indices of the first atom of each residue, in order.

        A residue boundary is any change in
        ``(chain_id, residue_id, insertion_code, model_id)``.
        """
        if self._residue_starts_cache is None:
            object.__setattr__(self, "_residue_starts_cache", self._compute_residue_starts())
        return self._residue_starts_cache  # type: ignore[return-value]

    def _compute_chain_starts(self) -> IntArray:
        n = len(self)
        if n == 0:
            return np.empty(0, dtype=np.int32)
        chain_change = np.empty(n, dtype=bool)
        chain_change[0] = True
        chain_change[1:] = (self.chain_id[1:] != self.chain_id[:-1]) | (
            self.model_id[1:] != self.model_id[:-1]
        )
        return np.nonzero(chain_change)[0].astype(np.int32)

    def _compute_residue_starts(self) -> IntArray:
        n = len(self)
        if n == 0:
            return np.empty(0, dtype=np.int32)
        res_change = np.empty(n, dtype=bool)
        res_change[0] = True
        res_change[1:] = (
            (self.residue_id[1:] != self.residue_id[:-1])
            | (self.chain_id[1:] != self.chain_id[:-1])
            | (self.insertion_code[1:] != self.insertion_code[:-1])
            | (self.model_id[1:] != self.model_id[:-1])
        )
        return np.nonzero(res_change)[0].astype(np.int32)

    @property
    def n_atoms(self) -> int:
        return len(self)

    @property
    def n_residues(self) -> int:
        return int(self.residue_starts.shape[0])

    @property
    def n_chains(self) -> int:
        return int(self.chain_starts.shape[0])

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------
    def iter_residue_slices(self) -> Iterable[slice]:
        """Yield a ``slice`` for each residue's atoms (in array order)."""
        starts = self.residue_starts
        n = len(self)
        for i, s in enumerate(starts):
            e = int(starts[i + 1]) if i + 1 < len(starts) else n
            yield slice(int(s), e)

    def iter_chain_slices(self) -> Iterable[slice]:
        """Yield a ``slice`` for each chain's atoms (in array order)."""
        starts = self.chain_starts
        n = len(self)
        for i, s in enumerate(starts):
            e = int(starts[i + 1]) if i + 1 < len(starts) else n
            yield slice(int(s), e)
