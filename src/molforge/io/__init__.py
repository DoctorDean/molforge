"""File I/O for molforge.

This subpackage provides parsers and writers for the file formats you'll
encounter across structural-biology workflows. The top-level entry
points are :func:`load`, :func:`save`, and :func:`fetch`, which dispatch
to the appropriate handler based on the file extension.

Supported formats:

- **PDB** (``.pdb``, ``.ent``) — full read/write, the universal default.
- **mmCIF / PDBx** (``.cif``, ``.mmcif``) — read/write planned;
  recommended for structures with >99,999 atoms (PDB's hard limit).
- **FASTA** (``.fasta``, ``.fa``, ``.faa``, ``.fna``) — sequence read/write.
- **PDBQT** (``.pdbqt``) — AutoDock Vina format; read/write planned.
- **PQR** (``.pqr``) — APBS / PDB2PQR with explicit charges and radii;
  read/write planned.
- **SDF** (``.sdf``, ``.mol``) and **MOL2** (``.mol2``) — small-molecule
  exchange; read/write planned via RDKit.

Convenience helpers:

- :func:`fetch` — pull a structure by PDB ID from RCSB or AlphaFold.
- :func:`load_alphafold` — load an AlphaFold prediction, exposing pLDDT
  as a first-class field rather than buried in B-factor.

Example:
    >>> import molforge as mf
    >>> protein = mf.load("1ubq.pdb")
    >>> mf.save(protein, "1ubq_clean.pdb")
"""

from __future__ import annotations

from molforge.io.dispatch import fetch, load, save
from molforge.io.fasta import (
    FastaRecord,
    read_fasta,
    read_fasta_string,
    write_fasta,
    write_fasta_string,
)
from molforge.io.pdb import (
    PDBParseError,
    PDBWriteError,
    read_pdb,
    read_pdb_string,
    write_pdb,
    write_pdb_string,
)
from molforge.io.pdb_alphafold import is_alphafold_pdb, load_alphafold

__all__ = [  # noqa: RUF022 — grouped by format, not alphabetical
    # Top-level dispatch
    "load",
    "save",
    "fetch",
    # PDB
    "read_pdb",
    "read_pdb_string",
    "write_pdb",
    "write_pdb_string",
    "PDBParseError",
    "PDBWriteError",
    # FASTA
    "read_fasta",
    "read_fasta_string",
    "write_fasta",
    "write_fasta_string",
    "FastaRecord",
    # AlphaFold helpers
    "load_alphafold",
    "is_alphafold_pdb",
]
