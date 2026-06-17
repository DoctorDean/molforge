"""File I/O for molforge.

This subpackage provides parsers and writers for the file formats you'll
encounter across structural-biology workflows. The top-level entry
points are :func:`load`, :func:`save`, and :func:`fetch`, which dispatch
to the appropriate handler based on the file extension.

Supported formats:

- **PDB** (``.pdb``, ``.ent``) — full read/write, the universal default.
- **mmCIF / PDBx** (``.cif``, ``.mmcif``) — full read/write; recommended
  for structures with >99,999 atoms (PDB's hard limit).
- **FASTA** (``.fasta``, ``.fa``, ``.faa``, ``.fna``) — sequence read/write.
- **SDF** (``.sdf``, ``.mol``) — small-molecule exchange; full read/write
  of V2000 (coordinates, elements, title, property block). V3000 is not
  yet supported.
- **MOL2** (``.mol2``) — Tripos small-molecule exchange; full read/write
  of the ATOM section (coordinates, elements via Tripos type prefix,
  atom names, partial charges, substructure info).
- **PDBQT** (``.pdbqt``) — AutoDock / Vina format; full read/write of
  ATOM records with per-atom partial charges and AutoDock atom types,
  reusing the PDB reader for the leading columns. ROOT / BRANCH /
  TORSDOF rotatable-bond markers are read-tolerated; round-tripping
  preserves coordinates, charges, and types.
- **PQR** (``.pqr``) — APBS / PDB2PQR with explicit per-atom charges
  and radii. The leading PDB-compatible columns are parsed as
  fixed-position; the charge and radius are whitespace-split from the
  trailing fields (PQR is not strictly fixed-column past the
  coordinates). Radii are attached to ``protein.metadata["radii"]``.

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
from molforge.io.mmcif import (
    CIFParseError,
    CIFWriteError,
    read_cif,
    read_cif_string,
    write_cif,
    write_cif_string,
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
    # mmCIF
    "read_cif",
    "read_cif_string",
    "write_cif",
    "write_cif_string",
    "CIFParseError",
    "CIFWriteError",
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
