"""Ingest small molecules from the PDB Chemical Component Dictionary.

Every ligand in the PDB is identified by a short Chemical Component
Dictionary code — ``STI`` for imatinib, ``NAD``, ``ATP``. That code is what
experimental datasets actually record, so a workflow that reads "this holo
structure binds ``STI`` at this site" needs a way to turn the code into a
molecule it can co-fold, describe, or match decoys against.

:func:`fetch_ccd` does exactly that, and :func:`fetch_ccd_many` does a set.
Both mirror the :func:`~molforge.io.fetch_chembl` family: networking is
standard-library only (:mod:`urllib`), and building the molecule is
RDKit-backed (lazy), so a missing RDKit raises
:class:`~molforge.core.RDKitNotInstalledError`.

Coordinates come from RCSB's per-component SDF, which carries the
*idealized* geometry — computed for the component in isolation rather than
observed in any one structure. That is the right starting point for docking
or conformer generation; for a ligand as actually bound, fetch the holo
structure itself with :func:`~molforge.io.fetch` and slice it out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from molforge.core import Molecule, _rdkit

if TYPE_CHECKING:
    from collections.abc import Iterable

_LIGAND_URL = "https://files.rcsb.org/ligands/download"

__all__ = ["fetch_ccd", "fetch_ccd_many"]


def fetch_ccd(code: str, *, timeout: float = 30.0, sanitize: bool = True) -> Molecule:
    """Fetch one PDB chemical component by CCD code as a :class:`Molecule`.

    Downloads the component's idealized SDF from RCSB and parses it with its
    chemistry intact — bonds, formal charges, aromaticity, stereochemistry,
    and the 3D conformer.

    Args:
        code: A Chemical Component Dictionary code, e.g. ``"STI"``.
            Case-insensitive; RCSB serves these uppercase.
        timeout: Network timeout in seconds.
        sanitize: Run RDKit sanitization when parsing the SDF. Metal-
            coordinated components (``HEM``, ``B12``) encode coordination as
            ordinary bonds, which RDKit rejects on valence grounds — pass
            ``sanitize=False`` to read those, accepting a molecule whose
            aromaticity and valences haven't been normalized.

    Returns:
        A :class:`Molecule` with a 3D conformer, whose ``name`` is the CCD
        code, with ``metadata["source"] == "rcsb-ccd"`` and the ``ccd_code``
        recorded.

    Raises:
        ValueError: If ``code`` is empty or isn't alphanumeric, or if RDKit
            can't parse what RCSB returned.
        OSError: If the download fails — network error, timeout, or a
            non-2xx response (a 404 for an unknown code).
        RDKitNotInstalledError: If RDKit isn't installed.

    Example:
        >>> from molforge.io import fetch_ccd
        >>> imatinib = fetch_ccd("STI")
        >>> imatinib.name
        'STI'
    """
    import urllib.error
    import urllib.request

    code = _validate_code(code)

    url = f"{_LIGAND_URL}/{code}_ideal.sdf"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            sdf_text = response.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        raise OSError(
            f"CCD fetch failed: RCSB returned HTTP {e.code} for {code!r}. "
            "Check that the component code exists."
        ) from e
    except urllib.error.URLError as e:
        raise OSError(
            f"CCD fetch failed: could not reach RCSB ({e.reason}). Check your network connection."
        ) from e

    return _molecule_from_sdf(sdf_text, code, sanitize=sanitize)


def _validate_code(code: str) -> str:
    """Normalize a CCD code to the uppercase form RCSB serves.

    Codes are alphanumeric and short — three characters classically, five
    since the PDB began extending the space. The length isn't enforced (the
    server is the authority on which codes exist), but a code carrying a
    path separator or whitespace is a caller bug worth catching before it
    becomes a URL.
    """
    if not code or not code.strip():
        raise ValueError("code must be a non-empty CCD code, e.g. 'STI'")
    code = code.strip().upper()
    if not code.isalnum():
        raise ValueError(
            f"CCD codes are alphanumeric, got {code!r}. "
            "Pass the component code alone, e.g. 'STI' or 'NAD'."
        )
    return code


def _molecule_from_sdf(sdf_text: str, code: str, *, sanitize: bool) -> Molecule:
    """Build a Molecule from a single-record CCD SDF."""
    try:
        mol = _rdkit.mol_from_molblock(sdf_text, sanitize=sanitize)
    except ValueError as e:
        hint = (
            " Components that coordinate a metal (HEM, B12) encode the coordination as "
            "ordinary bonds, which fails RDKit's valence model; retry with sanitize=False."
            if sanitize
            else ""
        )
        raise ValueError(f"RDKit could not parse the SDF RCSB returned for {code!r}.{hint}") from e
    return Molecule.from_rdkit(
        mol,
        name=code,
        metadata={"source": "rcsb-ccd", "ccd_code": code},
    )


def fetch_ccd_many(
    codes: Iterable[str],
    *,
    timeout: float = 30.0,
    sanitize: bool = True,
    on_error: str = "raise",
) -> list[Molecule]:
    """Fetch several chemical components, one :func:`fetch_ccd` per code.

    Args:
        codes: The CCD codes to fetch, in the order you want them back.
        timeout: Per-download network timeout in seconds.
        sanitize: Run RDKit sanitization when parsing each SDF.
        on_error: ``"raise"`` (default) stops at the first code that fails;
            ``"skip"`` drops codes that fail — a download error, or an SDF
            RDKit won't sanitize — and returns the rest. ``"skip"`` is the
            useful mode for a ligand set assembled from a benchmark, where
            one obsolete code shouldn't lose the other ninety-nine.

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
    for code in codes:
        try:
            molecules.append(fetch_ccd(code, timeout=timeout, sanitize=sanitize))
        except (OSError, ValueError):
            if on_error == "raise":
                raise
    return molecules
