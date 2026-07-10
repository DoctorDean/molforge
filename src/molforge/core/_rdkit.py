"""Lazy-import shim for the RDKit-backed :class:`~molforge.core.Molecule`.

RDKit is never imported when :mod:`molforge.core` is imported — only when a
chemistry operation actually runs. Every entry point here raises
:class:`RDKitNotInstalledError` with the same install hint, so the error
story is consistent with the rest of the package (``MDEngineNotInstalledError``
and friends).

:class:`~molforge.core.Molecule` calls these functions through the module
(``from molforge.core import _rdkit`` then ``_rdkit.to_smiles(...)``) rather
than importing the names, so tests can substitute a fake backend at this
boundary without RDKit installed.
"""

from __future__ import annotations

from typing import Any


class RDKitNotInstalledError(ImportError):
    """Raised when a chemistry operation needs RDKit but it isn't installed."""


_INSTALL_HINT = (
    "Chemistry-aware Molecule features require RDKit. Install with:\n"
    "    pip install 'molforge[chem]'\n"
    "or directly:\n"
    "    pip install rdkit"
)


def _chem() -> Any:
    """Return ``rdkit.Chem`` or raise :class:`RDKitNotInstalledError`."""
    try:
        from rdkit import Chem
    except ImportError as e:  # pragma: no cover - exercised via the public API
        raise RDKitNotInstalledError(f"{_INSTALL_HINT}\nUnderlying error: {e}") from e
    return Chem


def mol_from_smiles(smiles: str, *, sanitize: bool = True) -> Any:
    """Parse a SMILES string into an RDKit ``Mol``.

    Raises:
        RDKitNotInstalledError: If RDKit isn't installed.
        ValueError: If RDKit can't parse ``smiles``.
    """
    chem = _chem()
    mol = chem.MolFromSmiles(smiles, sanitize=sanitize)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    return mol


def to_smiles(mol: Any, *, canonical: bool = True, isomeric: bool = True) -> str:
    """Canonical SMILES for ``mol`` (isomeric by default)."""
    chem = _chem()
    return str(chem.MolToSmiles(mol, canonical=canonical, isomericSmiles=isomeric))


def to_inchi(mol: Any) -> str:
    """Standard InChI for ``mol``."""
    chem = _chem()
    return str(chem.MolToInchi(mol))


def to_inchikey(mol: Any) -> str:
    """Standard InChIKey for ``mol`` (a stable structural hash)."""
    chem = _chem()
    return str(chem.MolToInchiKey(mol))


def formula(mol: Any) -> str:
    """Hill-system molecular formula, e.g. ``"C2H6O"``."""
    _chem()
    from rdkit.Chem import rdMolDescriptors

    return str(rdMolDescriptors.CalcMolFormula(mol))


def molecular_weight(mol: Any) -> float:
    """Average molecular weight in g/mol."""
    _chem()
    from rdkit.Chem import Descriptors

    return float(Descriptors.MolWt(mol))


def formal_charge(mol: Any) -> int:
    """Net formal charge (sum over atoms)."""
    chem = _chem()
    return int(chem.GetFormalCharge(mol))


def read_sdf_records(
    path: str, *, sanitize: bool = True, remove_hs: bool = False
) -> list[tuple[Any, str]]:
    """Read an SDF file into ``(mol, name)`` pairs, chemistry preserved.

    Uses RDKit's ``SDMolSupplier``, so bonds, formal charges, aromaticity,
    stereochemistry, and any 3D coordinates survive — unlike the
    coordinate-only :func:`molforge.io.read_sdf`. Records RDKit can't parse
    are skipped rather than raising, so one bad entry doesn't sink a bulk
    read.

    Raises:
        RDKitNotInstalledError: If RDKit isn't installed.
    """
    chem = _chem()
    supplier = chem.SDMolSupplier(str(path), sanitize=sanitize, removeHs=remove_hs)
    records: list[tuple[Any, str]] = []
    for mol in supplier:
        if mol is None:
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        records.append((mol, name))
    return records
