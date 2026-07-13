"""Tests for :func:`molforge.chem.molecule_descriptors`.

Atom-count descriptors come straight off the (fake) mol; molecular weight and
formal charge go through the ``molforge.core._rdkit`` shim, so those are
monkeypatched. The genuine not-installed path is checked too.
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


def test_names_vocabulary() -> None:
    assert sorted(DESCRIPTOR_NAMES) == [
        "formal_charge",
        "molecular_weight",
        "n_atoms",
        "n_heavy_atoms",
    ]


def test_all_descriptors(fake_props: None) -> None:
    assert molecule_descriptors(_mol()) == {
        "molecular_weight": 180.0,
        "formal_charge": -1,
        "n_atoms": 9,
        "n_heavy_atoms": 5,
    }


def test_named_subset(fake_props: None) -> None:
    assert molecule_descriptors(_mol(), names=["n_heavy_atoms", "formal_charge"]) == {
        "n_heavy_atoms": 5,
        "formal_charge": -1,
    }


def test_subset_computes_only_requested() -> None:
    # molecular_weight is NOT monkeypatched; asking only for n_atoms must not
    # touch it (else RDKit-absent would raise).
    assert molecule_descriptors(_mol(), names=["n_atoms"]) == {"n_atoms": 9}


def test_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="unknown descriptor"):
        molecule_descriptors(_mol(), names=["logp"])


def test_rdkit_absent_raises() -> None:
    # Default set includes molecular_weight, which needs RDKit.
    with pytest.raises(RDKitNotInstalledError):
        molecule_descriptors(_mol())
