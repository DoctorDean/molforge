"""AtomArray — a flat, NumPy-backed view of a protein's atoms.

Where the hierarchical model (`Protein` / `Chain` / `Residue` / `Atom`) is
ideal for structural reasoning, the linear view is ideal for ML featurizers,
SIMD-friendly analysis, and zero-copy interop with NumPy/PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AtomArray:
    """Flat array view over all atoms in a structure.

    All arrays have the same first dimension ``N`` (the total atom count).

    Attributes:
        coords: ``(N, 3)`` float32 Cartesian coordinates in angstroms.
        element: ``(N,)`` element symbols.
        atom_name: ``(N,)`` PDB atom names.
        residue_name: ``(N,)`` 3-letter residue codes.
        residue_id: ``(N,)`` integer residue sequence numbers.
        chain_id: ``(N,)`` chain identifiers.
        b_factor: ``(N,)`` float32 temperature factors.
        occupancy: ``(N,)`` float32 occupancies.
    """

    coords: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    element: np.ndarray = field(default_factory=lambda: np.empty(0, dtype="U2"))
    atom_name: np.ndarray = field(default_factory=lambda: np.empty(0, dtype="U4"))
    residue_name: np.ndarray = field(default_factory=lambda: np.empty(0, dtype="U3"))
    residue_id: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    chain_id: np.ndarray = field(default_factory=lambda: np.empty(0, dtype="U1"))
    b_factor: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    occupancy: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))

    def __len__(self) -> int:
        return int(self.coords.shape[0])

    def select(self, mask: np.ndarray) -> AtomArray:
        """Return a new ``AtomArray`` containing only atoms where ``mask`` is True."""
        raise NotImplementedError
