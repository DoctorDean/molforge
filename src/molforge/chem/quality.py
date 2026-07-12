"""Assess and deduplicate small molecules before modelling.

Ingestion and standardization produce molecules; before scoring, docking,
or training you usually want to keep only the ones that are chemically
sound and drop repeats. :func:`is_valid` reports whether a molecule passes
RDKit sanitization, and :func:`unique` removes duplicates by structural
identity (InChIKey or SMILES), keeping the first occurrence so input order
is preserved. Both are RDKit-backed (via :mod:`molforge.core._rdkit`) and
lazy — calling one without RDKit raises
:class:`~molforge.core.RDKitNotInstalledError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from molforge.core import Molecule
from molforge.core import _rdkit

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

__all__ = [
    "is_valid",
    "unique",
]


def is_valid(molecule: Molecule) -> bool:
    """Whether ``molecule`` passes RDKit sanitization.

    Sanitization (valence, aromaticity, kekulization) runs on a copy, so the
    molecule is never mutated. A structure RDKit rejects — a pentavalent
    carbon, an unkekulizable ring — is reported as invalid rather than
    raising, so this reads as a predicate you can filter a set on.

    Args:
        molecule: The molecule to check.

    Returns:
        ``True`` if the molecule sanitizes cleanly, ``False`` otherwise.

    Raises:
        RDKitNotInstalledError: If RDKit isn't installed.
    """
    return _rdkit.sanitize_ok(molecule.to_rdkit())


def unique(molecules: Iterable[Molecule], *, key: str = "inchikey") -> list[Molecule]:
    """Deduplicate molecules by structural identity, keeping the first seen.

    Args:
        molecules: The molecules to deduplicate.
        key: Which identity to compare on — ``"inchikey"`` (the default, a
            stable structural hash) or ``"smiles"`` (canonical isomeric
            SMILES). InChIKey is the safer default; SMILES is there for when
            InChI generation is unavailable or undesirable.

    Returns:
        A new list with duplicates removed, preserving input order and
        keeping the first molecule of each identity.

    Raises:
        ValueError: If ``key`` is neither ``"inchikey"`` nor ``"smiles"``.
        RDKitNotInstalledError: If RDKit isn't installed.
    """
    _check_key(key)
    return list(_iter_unique(molecules, key=key))


def _check_key(key: str) -> None:
    """Validate a dedup ``key``, raising ``ValueError`` if unsupported."""
    if key not in ("inchikey", "smiles"):
        raise ValueError(f"key must be 'inchikey' or 'smiles', got {key!r}")


def _iter_unique(molecules: Iterable[Molecule], *, key: str) -> Iterator[Molecule]:
    """Stream molecules, skipping repeats by ``key`` (assumes ``key`` valid).

    The lazy core shared by :func:`unique` and
    :meth:`molforge.chem.MoleculeDataset.dedup` — keeps the first occurrence
    of each identity, holding only the set of seen identities in memory.
    """
    seen: set[str] = set()
    for molecule in molecules:
        identity = molecule.inchikey if key == "inchikey" else molecule.smiles
        if identity in seen:
            continue
        seen.add(identity)
        yield molecule
