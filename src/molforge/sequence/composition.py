"""Composition and basic property calculations for protein sequences.

Quick stats: amino-acid composition, molecular weight, theoretical pI,
hydrophobicity / GRAVY score, instability index.

These are sequence-level — they don't need a 3D structure. For
structure-dependent properties (SASA, contacts, secondary structure)
see :mod:`molforge.structure`.
"""

from __future__ import annotations

from typing import Final

from molforge.core.constants import ONE_TO_THREE

# Monoisotopic average masses of amino acids in Da (NIST values).
# These are the *residue* masses (free amino acid minus a water).
_RESIDUE_MASS: Final[dict[str, float]] = {
    "A": 71.0788,
    "R": 156.1875,
    "N": 114.1038,
    "D": 115.0886,
    "C": 103.1388,
    "E": 129.1155,
    "Q": 128.1307,
    "G": 57.0519,
    "H": 137.1411,
    "I": 113.1594,
    "L": 113.1594,
    "K": 128.1741,
    "M": 131.1926,
    "F": 147.1766,
    "P": 97.1167,
    "S": 87.0782,
    "T": 101.1051,
    "W": 186.2132,
    "Y": 163.1760,
    "V": 99.1326,
}
# H2O mass to add for terminal hydrogen/hydroxyl on a polypeptide.
_WATER_MASS: Final[float] = 18.01528

# Kyte-Doolittle hydrophobicity scale (Kyte & Doolittle 1982).
_KD_HYDROPHOBICITY: Final[dict[str, float]] = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}


def composition(sequence: str, *, as_fraction: bool = False) -> dict[str, float]:
    """Per-residue composition of a sequence.

    Args:
        sequence: One-letter amino-acid sequence (case-insensitive).
        as_fraction: If True, return fractions (sum to 1). Default False
            returns raw counts.

    Returns:
        Dict mapping residue code to count (or fraction).
    """
    s = sequence.upper()
    out: dict[str, float] = dict.fromkeys(ONE_TO_THREE, 0)
    for c in s:
        if c in out:
            out[c] += 1
    if as_fraction:
        total = sum(out.values())
        if total > 0:
            out = {k: v / total for k, v in out.items()}
    return out


def length(sequence: str) -> int:
    """Number of standard amino acids in a sequence."""
    return sum(1 for c in sequence.upper() if c in ONE_TO_THREE)


def molecular_weight(sequence: str) -> float:
    """Approximate molecular weight in Da.

    Sums per-residue monoisotopic masses plus one water for the terminal
    H/OH. Non-standard residues are ignored.
    """
    s = sequence.upper()
    total = sum(_RESIDUE_MASS.get(c, 0.0) for c in s)
    if total == 0:
        return 0.0
    return total + _WATER_MASS


def gravy(sequence: str) -> float:
    """Grand average of hydropathy (GRAVY) — Kyte-Doolittle 1982.

    Higher = more hydrophobic. Soluble globular proteins typically have
    GRAVY between -2 and +2.
    """
    s = sequence.upper()
    vals = [_KD_HYDROPHOBICITY[c] for c in s if c in _KD_HYDROPHOBICITY]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def aromaticity(sequence: str) -> float:
    """Fraction of aromatic residues (F, W, Y).

    A quick proxy for things like protein UV absorption.
    """
    n = length(sequence)
    if n == 0:
        return 0.0
    s = sequence.upper()
    return sum(1 for c in s if c in "FWY") / n
