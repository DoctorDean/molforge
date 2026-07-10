"""Tests for :class:`molforge.core.Molecule`.

RDKit isn't installed in this environment, which lets us test two things
directly: the *real* not-installed path (unmocked), and the molecule's own
delegation logic against a fake backend — the shim functions in
``molforge.core._rdkit`` are monkeypatched to return canned values from a
fake RDKit ``Mol``.
"""

from __future__ import annotations

import pytest

from molforge.core import Molecule, RDKitNotInstalledError
from molforge.core import _rdkit


class _FakeMol:
    """Minimal stand-in for an RDKit ``Mol`` (only the methods used)."""

    def __init__(self, n_atoms: int = 9, n_heavy: int = 3) -> None:
        self._n = n_atoms
        self._heavy = n_heavy

    def GetNumAtoms(self) -> int:
        return self._n

    def GetNumHeavyAtoms(self) -> int:
        return self._heavy


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Substitute canned ethanol values at the _rdkit shim boundary."""
    monkeypatch.setattr(_rdkit, "to_smiles", lambda mol, **k: "CCO")
    monkeypatch.setattr(_rdkit, "to_inchi", lambda mol: "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3")
    monkeypatch.setattr(_rdkit, "to_inchikey", lambda mol: "LFQSCWFLJHTTHZ-UHFFFAOYSA-N")
    monkeypatch.setattr(_rdkit, "formula", lambda mol: "C2H6O")
    monkeypatch.setattr(_rdkit, "molecular_weight", lambda mol: 46.07)
    monkeypatch.setattr(_rdkit, "formal_charge", lambda mol: 0)


class TestConstruction:
    def test_from_rdkit_shares_the_mol(self) -> None:
        mol = _FakeMol()
        m = Molecule.from_rdkit(mol, name="ethanol")
        assert m.to_rdkit() is mol  # shared, not copied
        assert m.name == "ethanol"

    def test_none_mol_rejected(self) -> None:
        with pytest.raises(ValueError, match="requires an RDKit Mol"):
            Molecule(None)

    def test_metadata_is_copied(self) -> None:
        src = {"source": "chembl"}
        m = Molecule.from_rdkit(_FakeMol(), metadata=src)
        m.metadata["extra"] = 1
        assert src == {"source": "chembl"}  # not aliased

    def test_from_smiles_wraps_parsed_mol(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mol = _FakeMol()
        monkeypatch.setattr(_rdkit, "mol_from_smiles", lambda s, **k: mol)
        m = Molecule.from_smiles("CCO", name="ethanol")
        assert m.to_rdkit() is mol
        assert m.name == "ethanol"


class TestProperties:
    def test_identity_properties(self, fake_backend: None) -> None:
        m = Molecule.from_rdkit(_FakeMol())
        assert m.smiles == "CCO"
        assert m.inchikey == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
        assert m.inchi.startswith("InChI=")
        assert m.formula == "C2H6O"
        assert m.molecular_weight == pytest.approx(46.07)
        assert m.formal_charge == 0

    def test_atom_counts_from_mol(self) -> None:
        m = Molecule.from_rdkit(_FakeMol(n_atoms=9, n_heavy=3))
        assert m.n_atoms == 9
        assert m.n_heavy_atoms == 3

    def test_repr(self) -> None:
        assert repr(Molecule.from_rdkit(_FakeMol(n_atoms=9))) == "Molecule(n_atoms=9)"
        m = Molecule.from_rdkit(_FakeMol(n_atoms=9), name="ethanol")
        assert repr(m) == "Molecule(n_atoms=9 name='ethanol')"


class TestRDKitAbsent:
    """With RDKit genuinely absent, chemistry entry points raise cleanly."""

    def test_from_smiles_raises(self) -> None:
        with pytest.raises(RDKitNotInstalledError, match="require RDKit"):
            Molecule.from_smiles("CCO")

    def test_property_raises(self) -> None:
        # a molecule can be *wrapped* around a fake mol without RDKit, but a
        # chemistry property that hits the real backend raises
        m = Molecule.from_rdkit(_FakeMol())
        with pytest.raises(RDKitNotInstalledError):
            _ = m.smiles

    def test_error_is_importerror(self) -> None:
        assert issubclass(RDKitNotInstalledError, ImportError)
