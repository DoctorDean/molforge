"""Reference constants for the core data model.

These tables are intentionally small and self-contained — no external data
files, no I/O. They cover the 20 canonical amino acids plus the most common
non-canonical residues you'll see in real PDB files.
"""

from __future__ import annotations

from typing import Final

# Canonical 20 amino acids — three-letter <-> one-letter.
THREE_TO_ONE: Final[dict[str, str]] = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLU": "E",
    "GLN": "Q",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

ONE_TO_THREE: Final[dict[str, str]] = {v: k for k, v in THREE_TO_ONE.items()}

# Common non-canonical / modified residues that map to canonical AAs.
# Source: RCSB component dictionary; extend as needed.
NON_CANONICAL_TO_ONE: Final[dict[str, str]] = {
    "MSE": "M",  # selenomethionine
    "SEC": "U",  # selenocysteine
    "PYL": "O",  # pyrrolysine
    "HSD": "H",
    "HSE": "H",
    "HSP": "H",  # CHARMM histidine variants
    "HIE": "H",
    "HID": "H",
    "HIP": "H",  # AMBER histidine variants
    "CYX": "C",
    "CYM": "C",  # disulfide / deprotonated Cys
    "ASH": "D",
    "GLH": "E",  # protonated Asp / Glu
    "LYN": "K",  # neutral Lys
    "TPO": "T",  # phosphothreonine
    "SEP": "S",  # phosphoserine
    "PTR": "Y",  # phosphotyrosine
}

# Standard nucleotides (DNA + RNA), three-letter -> one-letter.
NUCLEOTIDE_TO_ONE: Final[dict[str, str]] = {
    "DA": "A",
    "DT": "T",
    "DG": "G",
    "DC": "C",
    "DI": "I",
    "A": "A",
    "U": "U",
    "G": "G",
    "C": "C",
    "I": "I",
}

# Common waters and ions — handy for filtering.
WATER_RESIDUES: Final[frozenset[str]] = frozenset({"HOH", "WAT", "TIP", "TIP3", "TIP4", "H2O"})
ION_RESIDUES: Final[frozenset[str]] = frozenset(
    {
        "NA",
        "K",
        "CL",
        "MG",
        "CA",
        "ZN",
        "FE",
        "MN",
        "CU",
        "NI",
        "CO",
        "BR",
        "F",
        "I",
        "SO4",
        "PO4",
    }
)

# Standard backbone atoms for proteins.
PROTEIN_BACKBONE_ATOMS: Final[frozenset[str]] = frozenset({"N", "CA", "C", "O", "OXT"})
# Standard backbone atoms for nucleic acids.
NUCLEIC_BACKBONE_ATOMS: Final[frozenset[str]] = frozenset(
    {
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "C3'",
        "O3'",
        "C2'",
        "C1'",
        "O4'",
    }
)


def three_to_one(resname: str, *, unknown: str = "X") -> str:
    """Convert a 3-letter residue name to one-letter code.

    Falls back to `unknown` for residues outside the canonical and known
    non-canonical tables. Nucleotides are handled too.
    """
    key = resname.strip().upper()
    if key in THREE_TO_ONE:
        return THREE_TO_ONE[key]
    if key in NON_CANONICAL_TO_ONE:
        return NON_CANONICAL_TO_ONE[key]
    if key in NUCLEOTIDE_TO_ONE:
        return NUCLEOTIDE_TO_ONE[key]
    return unknown


def is_standard_amino_acid(resname: str) -> bool:
    """Return True for the 20 canonical amino acids."""
    return resname.strip().upper() in THREE_TO_ONE


def is_water(resname: str) -> bool:
    """Return True for common water residue names."""
    return resname.strip().upper() in WATER_RESIDUES


def is_ion(resname: str) -> bool:
    """Return True for common monatomic / small ion residue names."""
    return resname.strip().upper() in ION_RESIDUES
