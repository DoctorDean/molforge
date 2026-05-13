"""Structural analysis: RMSD, SASA, contacts, secondary structure, geometry."""

from __future__ import annotations

__all__ = ["contacts", "dssp", "rmsd", "sasa"]


def rmsd(a: object, b: object, *, align: bool = True) -> float:
    """Root-mean-square deviation between two structures (optionally superposed). TODO."""
    raise NotImplementedError


def sasa(protein: object) -> object:
    """Solvent-accessible surface area per atom or per residue. TODO."""
    raise NotImplementedError


def contacts(protein: object, cutoff: float = 5.0) -> object:
    """Inter-residue contact map at the given distance cutoff. TODO."""
    raise NotImplementedError


def dssp(protein: object) -> object:
    """DSSP secondary-structure assignment. TODO."""
    raise NotImplementedError
