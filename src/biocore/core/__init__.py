"""Core data model: hierarchical and linear views of protein structure.

The hierarchy is:

    Protein -> Chain -> Residue -> Atom

A `Protein` additionally exposes:

- `atom_array`: a flat, NumPy-backed view of every atom (linear view).
- `sequence`:   the one-letter amino-acid sequence per chain.

Both views are kept consistent; mutating one updates the other.
"""

from __future__ import annotations

from biocore.core.atom import Atom
from biocore.core.atom_array import AtomArray
from biocore.core.chain import Chain
from biocore.core.protein import Protein
from biocore.core.residue import Residue

__all__ = ["Atom", "AtomArray", "Chain", "Protein", "Residue"]
