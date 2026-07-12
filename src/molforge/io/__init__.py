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

For MD trajectories:

- :func:`read_trajectory` / :func:`iter_trajectory` /
  :func:`write_trajectory` — eager and streaming I/O for binary MD
  trajectories (``.xtc``, ``.trr``, ``.dcd``, ``.nc``, ``.h5``, plus
  multi-MODEL PDB). Trajectories are kept off the :func:`load` /
  :func:`save` dispatcher because they need an explicit ``topology``
  argument and return :class:`molforge.md.Trajectory` rather than
  :class:`molforge.core.Protein`.

Convenience helpers:

- :func:`fetch` / :func:`fetch_many` — pull one or many structures by PDB ID
  from RCSB or AlphaFold.
- :func:`search_rcsb` — full-text search the RCSB PDB for entry IDs, ready to
  hand to :func:`fetch_many`.
- :func:`load_alphafold` — load an AlphaFold prediction, exposing pLDDT
  as a first-class field rather than buried in B-factor.

Example:
    >>> import molforge as mf
    >>> protein = mf.load("1ubq.pdb")
    >>> mf.save(protein, "1ubq_clean.pdb")
"""

from __future__ import annotations

from molforge.io.dispatch import fetch, fetch_many, load, save
from molforge.io.rcsb_search import search_rcsb
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
from molforge.io.molecules import iter_molecules, iter_smiles, read_molecules, read_smiles
from molforge.io.trajectory import (
    iter_trajectory,
    read_trajectory,
    write_trajectory,
)

__all__ = [  # noqa: RUF022 — grouped by format, not alphabetical
    # Top-level dispatch
    "load",
    "save",
    "fetch",
    "fetch_many",
    "search_rcsb",
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
    # Trajectory I/O
    "read_trajectory",
    "iter_trajectory",
    "write_trajectory",
    # Small-molecule ingestion (chemistry-aware, RDKit-backed)
    "read_molecules",
    "read_smiles",
    "iter_molecules",
    "iter_smiles",
]
