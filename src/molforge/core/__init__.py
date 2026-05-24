"""Core data model: hierarchical and linear views of protein structure.

The :class:`AtomArray` is the *canonical* representation — a flat,
NumPy-backed array of all atoms. The hierarchical classes
(:class:`Protein`, :class:`Chain`, :class:`Residue`, :class:`Atom`) are
lightweight views that read and write through to the array.

Typical usage:

    >>> from molforge.core import Protein, AtomArray
    >>> protein = Protein(atom_array=AtomArray(0), name="example")
    >>> protein.n_atoms
    0
"""

from __future__ import annotations

from molforge.core.atom import Atom
from molforge.core.atom_array import (
    ATOM_FIELDS,
    AtomArray,
    BoolArray,
    FloatArray,
    IntArray,
    StrArray,
)
from molforge.core.chain import Chain
from molforge.core.constants import (
    NUCLEOTIDE_TO_ONE,
    ONE_TO_THREE,
    PROTEIN_BACKBONE_ATOMS,
    THREE_TO_ONE,
    is_ion,
    is_standard_amino_acid,
    is_water,
    three_to_one,
)
from molforge.core.metadata_keys import ProteinMetadata
from molforge.core.protein import Protein
from molforge.core.residue import Residue

__all__ = [  # noqa: RUF022 — grouped by concept, not alphabetical
    # Hierarchical
    "Atom",
    "Residue",
    "Chain",
    "Protein",
    # Linear
    "AtomArray",
    "ATOM_FIELDS",
    # Type aliases
    "BoolArray",
    "FloatArray",
    "IntArray",
    "StrArray",
    # Metadata vocabulary (string constants live in
    # molforge.core.metadata_keys; the TypedDict is re-exported here)
    "ProteinMetadata",
    # Constants & helpers
    "THREE_TO_ONE",
    "ONE_TO_THREE",
    "NUCLEOTIDE_TO_ONE",
    "PROTEIN_BACKBONE_ATOMS",
    "three_to_one",
    "is_standard_amino_acid",
    "is_water",
    "is_ion",
]
