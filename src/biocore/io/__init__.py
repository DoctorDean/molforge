"""File I/O: PDB, mmCIF, FASTA, MOL2, SDF, and trajectory formats.

The top-level :func:`load` and :func:`save` dispatch on file extension
to the appropriate parser/writer.
"""

from __future__ import annotations

__all__ = ["fetch", "load", "save"]


def load(path: str, *, format: str | None = None) -> object:
    """Load a structure from a file. Format is auto-detected if not given. TODO."""
    raise NotImplementedError


def save(structure: object, path: str, *, format: str | None = None) -> None:
    """Save a structure to a file. TODO."""
    raise NotImplementedError


def fetch(pdb_id: str, *, source: str = "rcsb") -> object:
    """Fetch a structure by PDB ID from a remote source. TODO."""
    raise NotImplementedError
