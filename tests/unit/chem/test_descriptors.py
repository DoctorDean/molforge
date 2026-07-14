"""Tests for :func:`molforge.chem.molecule_descriptors`.

Atom-count descriptors come straight off the (fake) mol; the RDKit-backed
descriptors go through the ``molforge.core._rdkit`` shim, so those are
monkeypatched. The genuine not-installed path is checked too, and a real
RDKit reference check runs where RDKit is installed.
"""

from __future__ import annotations

import pytest

from molforge.chem import DESCRIPTOR_NAMES, molecule_descriptors
from molforge.core import Molecule, RDKitNotInstalledError, _rdkit


class _FakeMol:
    def GetNumAtoms(self) -> int:
        return 9

    def GetNumHeavyAtoms(self) -> int:
        return 5


def _mol() -> Molecule:
    return Molecule.from_rdkit(_FakeMol())


@pytest.fixture
def fake_props(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_rdkit, "molecular_weight", lambda m: 180.0)
    monkeypatch.setattr(_rdkit, "formal_charge", lambda m: -1)
    monkeypatch.setattr(_rdkit, "logp", lambda m: 6.0)
    monkeypatch.setattr(_rdkit, "tpsa", lambda m: 40.0)
    monkeypatch.setattr(_rdkit, "num_h_donors", lambda m: 2)
    monkeypatch.setattr(_rdkit, "num_h_acceptors", lambda m: 3)
    monkeypatch.setattr(_rdkit, "num_rotatable_bonds", lambda m: 4)


def test_names_vocabulary() -> None:
    assert sorted(DESCRIPTOR_NAMES) == [
        "formal_charge",
        "lipinski_violations",
        "logp",
        "molecular_weight",
        "n_atoms",
        "n_h_acceptors",
        "n_h_donors",
        "n_heavy_atoms",
        "n_rotatable_bonds",
        "tpsa",
    ]


def test_all_descriptors(fake_props: None) -> None:
    assert molecule_descriptors(_mol()) == {
        "molecular_weight": 180.0,
        "formal_charge": -1,
        "n_atoms": 9,
        "n_heavy_atoms": 5,
        "logp": 6.0,
        "tpsa": 40.0,
        "n_h_donors": 2,
        "n_h_acceptors": 3,
        "n_rotatable_bonds": 4,
        # MW 180 ok, logP 6 > 5 (1), donors 2 ok, acceptors 3 ok -> 1 violation.
        "lipinski_violations": 1,
    }


def test_named_subset(fake_props: None) -> None:
    assert molecule_descriptors(_mol(), names=["n_heavy_atoms", "tpsa"]) == {
        "n_heavy_atoms": 5,
        "tpsa": 40.0,
    }


def test_subset_computes_only_requested() -> None:
    # molecular_weight is NOT monkeypatched; asking only for n_atoms must not
    # touch it (else RDKit-absent would raise).
    assert molecule_descriptors(_mol(), names=["n_atoms"]) == {"n_atoms": 9}


@pytest.mark.parametrize(
    "mw, lp, hbd, hba, expected",
    [
        (300.0, 2.0, 1, 4, 0),  # clean
        (600.0, 2.0, 1, 4, 1),  # MW only
        (600.0, 6.0, 8, 12, 4),  # all four
        (500.0, 5.0, 5, 10, 0),  # exactly at the limits -> not violations
    ],
)
def test_lipinski_violations(
    monkeypatch: pytest.MonkeyPatch, mw: float, lp: float, hbd: int, hba: int, expected: int
) -> None:
    monkeypatch.setattr(_rdkit, "molecular_weight", lambda m: mw)
    monkeypatch.setattr(_rdkit, "logp", lambda m: lp)
    monkeypatch.setattr(_rdkit, "num_h_donors", lambda m: hbd)
    monkeypatch.setattr(_rdkit, "num_h_acceptors", lambda m: hba)
    assert molecule_descriptors(_mol(), names=["lipinski_violations"]) == {
        "lipinski_violations": expected
    }


def test_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="unknown descriptor"):
        molecule_descriptors(_mol(), names=["not_a_descriptor"])


def test_rdkit_absent_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # An RDKit-backed descriptor raising must propagate through
    # molecule_descriptors. Simulate the absent-RDKit failure so the test is
    # deterministic whether or not RDKit happens to be installed.
    def _boom(_mol: object) -> float:
        raise RDKitNotInstalledError("RDKit is not installed")

    monkeypatch.setattr(_rdkit, "molecular_weight", _boom)
    with pytest.raises(RDKitNotInstalledError):
        molecule_descriptors(_mol())


class TestRealRDKit:
    """Reference values from a real RDKit install (skips without it)."""

    def test_ethanol_descriptors(self) -> None:
        pytest.importorskip("rdkit")
        mol = Molecule.from_smiles("CCO", name="ethanol")
        d = molecule_descriptors(mol)
        assert d["molecular_weight"] == pytest.approx(46.07, abs=0.05)
        assert d["tpsa"] == pytest.approx(20.23, abs=0.5)
        assert d["n_h_donors"] == 1
        assert d["n_h_acceptors"] == 1
        assert d["n_rotatable_bonds"] == 0
        assert d["logp"] < 1.0
        assert d["lipinski_violations"] == 0
