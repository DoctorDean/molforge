"""Ingest small molecules from ChEMBL.

:func:`fetch_chembl` pulls a compound from the ChEMBL REST API by ID and
builds a chemistry-aware :class:`~molforge.core.Molecule` from its canonical
SMILES; :func:`fetch_chembl_many` does a set. Networking is standard-library
only (:mod:`urllib`); building the molecule is RDKit-backed (lazy), so a
missing RDKit raises :class:`~molforge.core.RDKitNotInstalledError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from molforge.core import Molecule

if TYPE_CHECKING:
    from collections.abc import Iterable

_MOLECULE_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule"

__all__ = ["fetch_chembl", "fetch_chembl_many"]


def fetch_chembl(chembl_id: str, *, timeout: float = 30.0, sanitize: bool = True) -> Molecule:
    """Fetch one compound from ChEMBL by ID as a :class:`Molecule`.

    Downloads the entry from the ChEMBL REST API and builds a molecule from
    its canonical SMILES.

    Args:
        chembl_id: A ChEMBL molecule ID, e.g. ``"CHEMBL25"``.
        timeout: Network timeout in seconds.
        sanitize: Run RDKit sanitization when parsing the SMILES.

    Returns:
        A :class:`Molecule` whose ``name`` is ChEMBL's preferred name (or the
        ID when there's none), with ``metadata["source"] == "chembl"`` and the
        ``chembl_id`` recorded.

    Raises:
        ValueError: If ``chembl_id`` is empty, or the entry carries no
            small-molecule structure (e.g. a biotherapeutic).
        OSError: If the download fails — network error, timeout, or a non-2xx
            response (a 404 for an unknown ID).
        RDKitNotInstalledError: If RDKit isn't installed.

    Example:
        >>> from molforge.io import fetch_chembl
        >>> aspirin = fetch_chembl("CHEMBL25")
    """
    import json
    import urllib.error
    import urllib.request

    if not chembl_id or not chembl_id.strip():
        raise ValueError("chembl_id must be a non-empty string")
    chembl_id = chembl_id.strip()

    url = f"{_MOLECULE_URL}/{chembl_id}.json"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        raise OSError(
            f"ChEMBL fetch failed: returned HTTP {e.code} for {chembl_id!r}. "
            "Check that the ID exists."
        ) from e
    except urllib.error.URLError as e:
        raise OSError(
            f"ChEMBL fetch failed: could not reach ChEMBL ({e.reason}). "
            "Check your network connection."
        ) from e

    return _molecule_from_record(json.loads(body), chembl_id, sanitize=sanitize)


def _molecule_from_record(record: dict[str, Any], chembl_id: str, *, sanitize: bool) -> Molecule:
    """Build a Molecule from a ChEMBL molecule record's canonical SMILES."""
    structures = record.get("molecule_structures") or {}
    smiles = structures.get("canonical_smiles")
    if not smiles:
        raise ValueError(
            f"ChEMBL entry {chembl_id!r} has no small-molecule structure "
            "(no canonical SMILES); it may be a biotherapeutic."
        )
    name = record.get("pref_name") or chembl_id
    return Molecule.from_smiles(
        smiles,
        name=name,
        sanitize=sanitize,
        metadata={
            "source": "chembl",
            "chembl_id": record.get("molecule_chembl_id", chembl_id),
        },
    )


def fetch_chembl_many(
    chembl_ids: Iterable[str],
    *,
    timeout: float = 30.0,
    sanitize: bool = True,
    on_error: str = "raise",
) -> list[Molecule]:
    """Fetch several ChEMBL compounds by ID, one :func:`fetch_chembl` per ID.

    Args:
        chembl_ids: The ChEMBL IDs to fetch, in the order you want them back.
        timeout: Per-download network timeout in seconds.
        sanitize: Run RDKit sanitization when parsing each SMILES.
        on_error: ``"raise"`` (default) stops at the first ID that fails;
            ``"skip"`` drops IDs that fail — a download error or an entry with
            no small-molecule structure — and returns the rest.

    Returns:
        The fetched molecules, in input order (minus any dropped under
        ``on_error="skip"``).

    Raises:
        ValueError: If ``on_error`` is not ``"raise"`` or ``"skip"``.
        OSError: On a download failure when ``on_error="raise"``.
        RDKitNotInstalledError: If RDKit isn't installed.
    """
    if on_error not in ("raise", "skip"):
        raise ValueError(f"on_error must be 'raise' or 'skip', got {on_error!r}")

    molecules: list[Molecule] = []
    for chembl_id in chembl_ids:
        try:
            molecules.append(fetch_chembl(chembl_id, timeout=timeout, sanitize=sanitize))
        except (OSError, ValueError):
            if on_error == "raise":
                raise
    return molecules
