"""Cheminformatics operations on :class:`~molforge.core.Molecule`.

Where :mod:`molforge.core` holds the small-molecule *type* and
:mod:`molforge.io` reads molecules from files, this package holds the
chemistry *operations* — starting with standardization (cleaning) for
consistent, deduplicable structures. Everything here is RDKit-backed and
lazy: importing :mod:`molforge.chem` never pulls RDKit in, and an operation
without RDKit raises :class:`~molforge.core.RDKitNotInstalledError`.
"""

from __future__ import annotations

from molforge.chem.dataset import MoleculeDataset
from molforge.chem.descriptors import DESCRIPTOR_NAMES, molecule_descriptors
from molforge.chem.quality import is_valid, unique
from molforge.chem.standardize import (
    canonical_tautomer,
    cleanup,
    largest_fragment,
    neutralize,
    standardize,
)

__all__ = [
    "DESCRIPTOR_NAMES",
    "MoleculeDataset",
    "canonical_tautomer",
    "cleanup",
    "is_valid",
    "largest_fragment",
    "molecule_descriptors",
    "neutralize",
    "standardize",
    "unique",
]
