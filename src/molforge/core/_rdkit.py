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

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator


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


def logp(mol: Any) -> float:
    """Crippen octanol-water partition coefficient (MolLogP)."""
    _chem()
    from rdkit.Chem import Descriptors

    return float(Descriptors.MolLogP(mol))


def tpsa(mol: Any) -> float:
    """Topological polar surface area in Å²."""
    _chem()
    from rdkit.Chem import Descriptors

    return float(Descriptors.TPSA(mol))


def num_h_donors(mol: Any) -> int:
    """Number of hydrogen-bond donors (Lipinski definition)."""
    _chem()
    from rdkit.Chem import Descriptors

    return int(Descriptors.NumHDonors(mol))


def num_h_acceptors(mol: Any) -> int:
    """Number of hydrogen-bond acceptors (Lipinski definition)."""
    _chem()
    from rdkit.Chem import Descriptors

    return int(Descriptors.NumHAcceptors(mol))


def num_rotatable_bonds(mol: Any) -> int:
    """Number of rotatable bonds."""
    _chem()
    from rdkit.Chem import Descriptors

    return int(Descriptors.NumRotatableBonds(mol))


def _standardize_mod() -> Any:
    """Return ``rdkit.Chem.MolStandardize.rdMolStandardize`` or raise."""
    _chem()  # clean error first if RDKit is absent
    from rdkit.Chem.MolStandardize import rdMolStandardize

    return rdMolStandardize


def cleanup(mol: Any) -> Any:
    """RDKit ``Cleanup``: sanitize, normalize functional groups, reionize."""
    return _standardize_mod().Cleanup(mol)


def largest_fragment(mol: Any) -> Any:
    """Keep the largest organic fragment (strips salts/solvents)."""
    return _standardize_mod().FragmentParent(mol)


def uncharge(mol: Any) -> Any:
    """Neutralize where chemically reasonable."""
    return _standardize_mod().Uncharger().uncharge(mol)


def canonical_tautomer(mol: Any) -> Any:
    """Pick RDKit's canonical tautomer."""
    return _standardize_mod().TautomerEnumerator().Canonicalize(mol)


def sanitize_ok(mol: Any) -> bool:
    """Whether ``mol`` passes RDKit sanitization, checked on a copy.

    Sanitization runs on ``Chem.Mol(mol)`` so ``mol`` is never mutated. A
    structure RDKit rejects returns ``False`` rather than raising, which
    makes this a validity predicate. RDKit being absent still raises
    :class:`RDKitNotInstalledError`, consistent with the rest of the shim.
    """
    chem = _chem()
    try:
        chem.SanitizeMol(chem.Mol(mol))
    except Exception:  # any RDKit sanitize failure means the structure is invalid
        return False
    return True


def iter_sdf_records(
    path: str, *, sanitize: bool = True, remove_hs: bool = False
) -> Iterator[tuple[Any, str]]:
    """Stream an SDF file as ``(mol, name)`` pairs, one record at a time.

    Uses RDKit's ``ForwardSDMolSupplier`` over an open file handle, so a
    large multi-record SDF is parsed lazily and never fully materialized —
    the streaming counterpart to :func:`read_sdf_records`. Chemistry (bonds,
    formal charges, aromaticity, stereochemistry, any 3D coordinates) is
    preserved, and records RDKit can't parse are skipped rather than raising
    so one bad entry doesn't sink the stream.

    Yields:
        ``(mol, name)`` per parsable record, where ``name`` is the SDF title
        (the ``_Name`` property) or ``""``.

    Raises:
        RDKitNotInstalledError: If RDKit isn't installed.
    """
    chem = _chem()  # raise cleanly before opening the file
    with open(path, "rb") as handle:
        supplier = chem.ForwardSDMolSupplier(handle, sanitize=sanitize, removeHs=remove_hs)
        for mol in supplier:
            if mol is None:
                continue
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
            yield (mol, name)


def read_sdf_records(
    path: str, *, sanitize: bool = True, remove_hs: bool = False
) -> list[tuple[Any, str]]:
    """Read an SDF file into ``(mol, name)`` pairs, chemistry preserved.

    The eager counterpart to :func:`iter_sdf_records` — materializes the
    whole file. Bonds, formal charges, aromaticity, stereochemistry, and any
    3D coordinates survive, unlike the coordinate-only
    :func:`molforge.io.read_sdf`; unparsable records are skipped.

    Raises:
        RDKitNotInstalledError: If RDKit isn't installed.
    """
    return list(iter_sdf_records(path, sanitize=sanitize, remove_hs=remove_hs))


def has_conformer(mol: Any) -> bool:
    """Whether ``mol`` carries at least one 3D conformer."""
    _chem()  # clean error first if RDKit is absent
    return bool(mol.GetNumConformers())


def embed_conformer(mol: Any, *, seed: int, add_hs: bool) -> Any:
    """Return a copy of ``mol`` with a freshly generated ETKDG 3D conformer.

    ``add_hs`` adds explicit hydrogens before embedding (RDKit's recommended
    recipe for realistic geometry); the returned mol then carries them.

    Raises:
        RDKitNotInstalledError: If RDKit isn't installed.
        ValueError: If RDKit can't embed a conformer for this molecule.
    """
    chem = _chem()
    from rdkit.Chem import AllChem

    work = chem.AddHs(mol) if add_hs else chem.Mol(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(work, params) < 0:
        raise ValueError("RDKit could not generate a 3D conformer for this molecule")
    return work


def conformer_atoms(mol: Any) -> tuple[list[str], list[int], Any]:
    """Per-atom element symbols, formal charges, and ``(N, 3)`` coordinates.

    Coordinates come from ``mol``'s (first) conformer; element and charge are
    read in the same atom order, so the three line up positionally.

    Raises:
        RDKitNotInstalledError: If RDKit isn't installed.
    """
    _chem()
    conformer = mol.GetConformer()
    coords = conformer.GetPositions()
    elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
    charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    return elements, charges, coords


def mol_from_atoms(
    elements: list[str], coords: Any, *, charge: int, perceive_bond_orders: bool
) -> Any:
    """Build an RDKit ``Mol`` from element symbols and 3D coordinates.

    Connectivity — and, when ``perceive_bond_orders`` is set, bond orders —
    are inferred from geometry with RDKit's ``rdDetermineBonds``, the reverse
    of reading coordinates out of a mol. ``charge`` is the molecule's net
    formal charge, which bond-order perception needs to get valences right.
    Intended for small molecules, not whole polymers.

    Raises:
        RDKitNotInstalledError: If RDKit isn't installed.
        ValueError: If RDKit can't perceive bonds from the geometry.
    """
    chem = _chem()
    from rdkit.Chem import rdDetermineBonds
    from rdkit.Geometry import Point3D

    editable = chem.RWMol()
    for element in elements:
        editable.AddAtom(chem.Atom(element))
    conformer = chem.Conformer(len(elements))
    for i, position in enumerate(coords):
        x, y, z = position
        conformer.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    editable.AddConformer(conformer, assignId=True)
    mol = editable.GetMol()
    try:
        if perceive_bond_orders:
            rdDetermineBonds.DetermineBonds(mol, charge=charge)
        else:
            rdDetermineBonds.DetermineConnectivity(mol)
    except (ValueError, RuntimeError) as e:
        raise ValueError(f"RDKit could not perceive bonds from geometry: {e}") from e
    return mol
