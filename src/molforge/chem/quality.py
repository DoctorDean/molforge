"""Assess and deduplicate small molecules before modelling.

Ingestion and standardization produce molecules; before scoring, docking,
or training you usually want to keep only the ones that are chemically
sound and drop repeats. :func:`is_valid` reports whether a molecule passes
RDKit sanitization, and :func:`unique` removes duplicates by structural
identity (InChIKey, SMILES, or Bemis-Murcko scaffold), keeping the first
occurrence so input order is preserved. Both are RDKit-backed (via
:mod:`molforge.core._rdkit`) and lazy — calling one without RDKit raises
:class:`~molforge.core.RDKitNotInstalledError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from molforge.core import Molecule, _rdkit

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

__all__ = [
    "is_valid",
    "unique",
]

#: How each ``key`` turns a molecule into the string identity dedup compares
#: on. ``"scaffold"`` collapses a whole chemical series to one representative,
#: where the two structural keys only collapse exact repeats.
_IDENTITIES: dict[str, Callable[[Molecule], str]] = {
    "inchikey": lambda m: m.inchikey,
    "scaffold": lambda m: m.scaffold_smiles,
    "smiles": lambda m: m.smiles,
}


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
            stable structural hash), ``"smiles"`` (canonical isomeric
            SMILES), or ``"scaffold"`` (Bemis-Murcko scaffold SMILES).
            InChIKey is the safer default; SMILES is there for when InChI
            generation is unavailable or undesirable. ``"scaffold"`` dedups
            at the level of the *chemical series* rather than the exact
            structure — one representative per scaffold — and treats all
            acyclic molecules as a single (empty) scaffold.

    Returns:
        A new list with duplicates removed, preserving input order and
        keeping the first molecule of each identity.

    Raises:
        ValueError: If ``key`` isn't one of the supported identities.
        RDKitNotInstalledError: If RDKit isn't installed.
    """
    _check_key(key)
    return list(_iter_unique(molecules, key=key))


def _check_key(key: str) -> None:
    """Validate a dedup ``key``, raising ``ValueError`` if unsupported."""
    if key not in _IDENTITIES:
        options = ", ".join(repr(k) for k in sorted(_IDENTITIES))
        raise ValueError(f"key must be one of {options}, got {key!r}")


def _iter_unique(molecules: Iterable[Molecule], *, key: str) -> Iterator[Molecule]:
    """Stream molecules, skipping repeats by ``key`` (assumes ``key`` valid).

    The lazy core shared by :func:`unique` and
    :meth:`molforge.chem.MoleculeDataset.dedup` — keeps the first occurrence
    of each identity, holding only the set of seen identities in memory.
    """
    identity_of = _IDENTITIES[key]
    seen: set[str] = set()
    for molecule in molecules:
        identity = identity_of(molecule)
        if identity in seen:
            continue
        seen.add(identity)
        yield molecule
