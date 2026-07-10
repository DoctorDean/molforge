"""Tests for the chemistry-aware molecule readers.

RDKit isn't installed here, so per-molecule parsing is monkeypatched at the
``molforge.core._rdkit`` shim boundary; the file/line logic itself is real.
The genuine not-installed path is also checked.
"""

from __future__ import annotations

import pytest

from molforge.core import Molecule, RDKitNotInstalledError
from molforge.core import _rdkit
from molforge.io import read_molecules, read_smiles


class _FakeMol:
    def __init__(self, smiles: str) -> None:
        self.smiles = smiles

    def GetNumAtoms(self) -> int:
        return len(self.smiles)

    def GetNumHeavyAtoms(self) -> int:
        return len(self.smiles)


@pytest.fixture
def fake_smiles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_rdkit, "mol_from_smiles", lambda s, **k: _FakeMol(s))


class TestReadSmiles:
    def test_parses_lines_with_names(self, fake_smiles: None) -> None:
        mols = read_smiles("CCO ethanol\nc1ccccc1 benzene")
        assert [m.name for m in mols] == ["ethanol", "benzene"]
        assert isinstance(mols[0], Molecule)

    def test_skips_blanks_and_comments(self, fake_smiles: None) -> None:
        mols = read_smiles("\n# header\nCCO ethanol\n\n# note\nCC(=O)O\n")
        assert len(mols) == 2
        assert mols[1].name == ""  # no name column

    def test_name_column_optional(self, fake_smiles: None) -> None:
        (mol,) = read_smiles("CCO")
        assert mol.name == ""
        assert mol.to_rdkit().smiles == "CCO"

    def test_source_recorded(self, fake_smiles: None) -> None:
        mols = read_smiles("CCO x", source="lib.smi")
        assert mols[0].metadata["source"] == "lib.smi"


class TestReadMolecules:
    def test_smiles_file(self, tmp_path, fake_smiles: None) -> None:  # noqa: ANN001
        p = tmp_path / "lib.smi"
        p.write_text("CCO ethanol\nCCC propane\n")
        mols = read_molecules(p)
        assert [m.name for m in mols] == ["ethanol", "propane"]
        assert all(m.metadata["source"] == str(p) for m in mols)

    def test_sdf_dispatch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            _rdkit,
            "read_sdf_records",
            lambda path, **k: [(_FakeMol("CCO"), "lig1"), (_FakeMol("CCC"), "lig2")],
        )
        mols = read_molecules("ligs.sdf")
        assert [m.name for m in mols] == ["lig1", "lig2"]
        assert mols[0].metadata["source"] == "ligs.sdf"

    def test_explicit_format_overrides_extension(
        self, tmp_path, fake_smiles: None
    ) -> None:  # noqa: ANN001
        p = tmp_path / "molecules.txt"
        p.write_text("CCO ethanol\n")
        mols = read_molecules(p, format="smiles")
        assert mols[0].name == "ethanol"

    def test_unknown_extension_raises(self) -> None:
        with pytest.raises(ValueError, match="can't infer"):
            read_molecules("data.xyz")

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown molecule format"):
            read_molecules("data.dat", format="mol2")


class TestRDKitAbsent:
    def test_read_molecules_sdf_raises(self) -> None:
        with pytest.raises(RDKitNotInstalledError):
            read_molecules("x.sdf")

    def test_read_smiles_raises(self) -> None:
        with pytest.raises(RDKitNotInstalledError):
            read_smiles("CCO ethanol")
