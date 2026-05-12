"""Protein — the top-level structural container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from biocore.core.atom_array import AtomArray
    from biocore.core.chain import Chain


@dataclass
class Protein:
    """Top-level container for a protein (or protein complex) structure.

    A `Protein` is the entry point to the hierarchical data model. It also
    exposes a flat, NumPy-backed :class:`AtomArray` view for ML and
    vectorized analysis.

    Attributes:
        name: Optional identifier (e.g. PDB ID).
        chains: List of :class:`Chain` objects.
        metadata: Free-form metadata dict (resolution, method, header, ...).
    """

    name: str = ""
    chains: list[Chain] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def atom_array(self) -> AtomArray:
        """Return a flat, NumPy-backed view of every atom.

        The view is kept consistent with the hierarchical representation.
        """
        raise NotImplementedError

    @property
    def sequence(self) -> str:
        """Concatenated one-letter sequence across all chains."""
        raise NotImplementedError

    def __getitem__(self, chain_id: str) -> Chain:
        """Look up a chain by id (e.g. ``protein["A"]``)."""
        raise NotImplementedError
