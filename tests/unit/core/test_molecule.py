"""Tests for :class:`molforge.core.Molecule`.

RDKit isn't installed in this environment, which lets us test two things
directly: the *real* not-installed path (unmocked), and the molecule's own
delegation logic against a fake backend — the shim functions in
``molforge.core._rdkit`` are monkeypatched to return canned values from a
fake RDKit ``Mol``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from molforge.core import AtomArray, Molecule, RDKitNotInstalledError, _rdkit


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


class TestToAtomArray:
    """Bridge to AtomArray: use existing coords, or embed on command."""

    # (elements, formal_charges, coords) — integer coords are float32-exact.
    ETHANOL = (
        ["C", "C", "O"],
        [0, 0, 0],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]],
    )

    def test_uses_existing_conformer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_rdkit, "has_conformer", lambda mol: True)
        monkeypatch.setattr(_rdkit, "conformer_atoms", lambda mol: self.ETHANOL)
        monkeypatch.setattr(
            _rdkit, "embed_conformer", lambda *a, **k: pytest.fail("must not embed")
        )
        aa = Molecule.from_rdkit(_FakeMol(), name="ethanol").to_atom_array()
        assert len(aa) == 3
        assert list(aa.element) == ["C", "C", "O"]
        assert list(aa.atom_name) == ["C1", "C2", "O1"]  # per-element numbering
        assert aa.coords[2].tolist() == [2.0, 1.0, 0.0]
        assert list(aa.serial) == [1, 2, 3]
        assert set(aa.record_type) == {"HETATM"}
        assert set(aa.entity_type) == {"ligand"}
        assert set(aa.residue_name) == {"LIG"}

    def test_carries_formal_charges(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_rdkit, "has_conformer", lambda mol: True)
        monkeypatch.setattr(_rdkit, "conformer_atoms", lambda mol: (["N"], [1], [[0.0, 0.0, 0.0]]))
        aa = Molecule.from_rdkit(_FakeMol()).to_atom_array()
        assert aa.charge[0] == 1.0

    def test_embeds_on_command_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        got: dict[str, object] = {}

        def fake_embed(mol: object, *, seed: int, add_hs: bool) -> str:
            got["seed"] = seed
            got["add_hs"] = add_hs
            return "EMBEDDED"

        def fake_atoms(mol: object) -> tuple[list[str], list[int], list[list[float]]]:
            got["read_from"] = mol
            return (["C"], [0], [[0.0, 0.0, 0.0]])

        monkeypatch.setattr(_rdkit, "has_conformer", lambda mol: False)
        monkeypatch.setattr(_rdkit, "embed_conformer", fake_embed)
        monkeypatch.setattr(_rdkit, "conformer_atoms", fake_atoms)

        aa = Molecule.from_rdkit(_FakeMol()).to_atom_array(embed=True, add_hydrogens=True, seed=7)
        assert got["read_from"] == "EMBEDDED"  # coords read off the embedded mol
        assert got["seed"] == 7 and got["add_hs"] is True
        assert len(aa) == 1

    def test_missing_conformer_without_embed_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_rdkit, "has_conformer", lambda mol: False)
        with pytest.raises(ValueError, match="no 3D coordinates"):
            Molecule.from_rdkit(_FakeMol()).to_atom_array()


class TestFromAtomArray:
    """Reverse bridge: build a Molecule from element + coords by perceiving bonds."""

    @staticmethod
    def _aa() -> AtomArray:
        return AtomArray.from_dict(
            {
                "coords": np.array(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float32
                ),
                "element": np.array(["C", "C", "O"], dtype="U2"),
            }
        )

    def test_builds_molecule_from_elements(self, monkeypatch: pytest.MonkeyPatch) -> None:
        got: dict[str, object] = {}

        def fake(
            elements: list[str], coords: Any, *, charge: int, perceive_bond_orders: bool
        ) -> _FakeMol:
            got["elements"] = elements
            got["charge"] = charge
            got["orders"] = perceive_bond_orders
            got["coords_shape"] = coords.shape
            return _FakeMol()

        monkeypatch.setattr(_rdkit, "mol_from_atoms", fake)
        m = Molecule.from_atom_array(self._aa(), name="lig")
        assert isinstance(m, Molecule)
        assert m.name == "lig"
        assert got["elements"] == ["C", "C", "O"]  # numpy strs coerced to plain str
        assert got["charge"] == 0
        assert got["orders"] is True
        assert got["coords_shape"] == (3, 3)

    def test_threads_charge_and_connectivity_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        got: dict[str, object] = {}

        def fake(
            elements: list[str], coords: Any, *, charge: int, perceive_bond_orders: bool
        ) -> _FakeMol:
            got["charge"] = charge
            got["orders"] = perceive_bond_orders
            return _FakeMol()

        monkeypatch.setattr(_rdkit, "mol_from_atoms", fake)
        Molecule.from_atom_array(self._aa(), charge=-1, perceive_bond_orders=False)
        assert got == {"charge": -1, "orders": False}

    def test_metadata_preserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_rdkit, "mol_from_atoms", lambda *a, **k: _FakeMol())
        m = Molecule.from_atom_array(self._aa(), metadata={"source": "1abc"})
        assert m.metadata["source"] == "1abc"


class TestRDKitAbsent:
    """With RDKit genuinely absent, chemistry entry points raise cleanly."""

    def test_from_smiles_raises(self) -> None:
        with pytest.raises(RDKitNotInstalledError, match="require RDKit"):
            Molecule.from_smiles("CCO")

    def test_to_atom_array_raises(self) -> None:
        with pytest.raises(RDKitNotInstalledError):
            Molecule.from_rdkit(_FakeMol()).to_atom_array()

    def test_from_atom_array_raises(self) -> None:
        aa = AtomArray.from_dict(
            {
                "coords": np.zeros((1, 3), dtype=np.float32),
                "element": np.array(["C"], dtype="U2"),
            }
        )
        with pytest.raises(RDKitNotInstalledError):
            Molecule.from_atom_array(aa)

    def test_property_raises(self) -> None:
        # a molecule can be *wrapped* around a fake mol without RDKit, but a
        # chemistry property that hits the real backend raises
        m = Molecule.from_rdkit(_FakeMol())
        with pytest.raises(RDKitNotInstalledError):
            _ = m.smiles

    def test_error_is_importerror(self) -> None:
        assert issubclass(RDKitNotInstalledError, ImportError)
