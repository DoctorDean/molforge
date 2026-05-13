"""Protein — top-level container holding an :class:`AtomArray` plus metadata.

``Protein`` is the entry point to the hierarchy: it owns the canonical
``AtomArray`` and exposes chain / residue / atom views over it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Iterable

from molforge.core.atom_array import AtomArray
from molforge.core.chain import Chain


class Protein:
    """A protein (or protein complex) structure.

    ``Protein`` owns a single :class:`AtomArray` (``atom_array``) which is
    the canonical data store. Hierarchical accessors (``chains``,
    ``residues``, etc.) read from it.

    Args:
        atom_array: The flat array of atoms backing this protein. If
            omitted, an empty array is used.
        name: Optional identifier (e.g. PDB ID).
        metadata: Free-form key/value metadata (resolution, header, ...).
    """

    __slots__ = ("atom_array", "metadata", "name")

    def __init__(
        self,
        atom_array: AtomArray | None = None,
        *,
        name: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.atom_array = atom_array if atom_array is not None else AtomArray(0)
        self.name = name
        self.metadata: dict[str, Any] = dict(metadata) if metadata else {}

    # ------------------------------------------------------------------
    # Hierarchical access
    # ------------------------------------------------------------------
    @property
    def chains(self) -> list[Chain]:
        """All chains in this protein, in array order."""
        starts = self.atom_array.chain_starts
        n = len(self.atom_array)
        out: list[Chain] = []
        for i, s in enumerate(starts):
            e = int(starts[i + 1]) if i + 1 < len(starts) else n
            out.append(Chain(self.atom_array, int(s), e, parent=self))
        return out

    def __iter__(self) -> Iterable[Chain]:
        return iter(self.chains)

    def __len__(self) -> int:
        """Number of chains."""
        return self.n_chains

    def __getitem__(self, chain_id: str) -> Chain:
        """Look up a chain by id (e.g. ``protein["A"]``).

        If multiple chains share an id (e.g. across NMR models), the
        first match is returned. Use :meth:`get_chain` for explicit
        ``(chain_id, model_id)`` lookup.
        """
        for ch in self.chains:
            if ch.chain_id == chain_id:
                return ch
        raise KeyError(f"protein has no chain with id {chain_id!r}")

    def get_chain(self, chain_id: str, model_id: int = 0) -> Chain:
        """Look up a chain by ``(chain_id, model_id)``."""
        for ch in self.chains:
            if ch.chain_id == chain_id and ch.model_id == model_id:
                return ch
        raise KeyError(f"protein has no chain {chain_id!r} in model {model_id}")

    # ------------------------------------------------------------------
    # Linear / array views
    # ------------------------------------------------------------------
    @property
    def coords(self) -> NDArray[np.float32]:
        """All atom coordinates, shape ``(n_atoms, 3)``."""
        return self.atom_array.coords

    @property
    def sequence(self) -> str:
        """Concatenated one-letter sequence across all protein/nucleic chains.

        Chains are joined with ``"/"`` to make boundaries visible.
        Non-polymer chains (ligand, water, ion) are skipped.
        """
        return "/".join(ch.sequence for ch in self.chains if ch.sequence)

    def sequences(self) -> dict[str, str]:
        """Per-chain one-letter sequences keyed by ``chain_id``."""
        return {ch.chain_id: ch.sequence for ch in self.chains if ch.sequence}

    # ------------------------------------------------------------------
    # Counts / shape
    # ------------------------------------------------------------------
    @property
    def n_atoms(self) -> int:
        return len(self.atom_array)

    @property
    def n_residues(self) -> int:
        return self.atom_array.n_residues

    @property
    def n_chains(self) -> int:
        return self.atom_array.n_chains

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------
    def select(self, **filters: object) -> Protein:
        """Return a new ``Protein`` containing only atoms matching the filters.

        Filters are forwarded to :meth:`AtomArray.where`.

        Example:
            >>> # Keep only chain A, protein atoms
            >>> sub = protein.select(chain_id="A", entity_type="protein")
        """
        mask = self.atom_array.where(**filters)
        return Protein(
            self.atom_array.select(mask),
            name=self.name,
            metadata=dict(self.metadata),
        )

    def protein_only(self) -> Protein:
        """Return a new ``Protein`` containing only polymer protein atoms.

        Drops ligands, waters, ions, and nucleic acids.
        """
        return self.select(entity_type="protein")

    def remove_water(self) -> Protein:
        """Return a new ``Protein`` with all water atoms removed."""
        mask = self.atom_array.entity_type != "water"
        return Protein(
            self.atom_array.select(mask),
            name=self.name,
            metadata=dict(self.metadata),
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"Protein(name={self.name!r}, n_chains={self.n_chains}, "
            f"n_residues={self.n_residues}, n_atoms={self.n_atoms})"
        )
