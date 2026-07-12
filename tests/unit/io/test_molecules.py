"""Tests for the chemistry-aware molecule readers.

RDKit isn't installed here, so per-molecule parsing is monkeypatched at the
``molforge.core._rdkit`` shim boundary; the file/line logic itself is real.
The genuine not-installed path is also checked.
"""

from __future__ import annotations

import pytest

from molforge.core import Molecule, RDKitNotInstalledError
from molforge.core import _rdkit
from molforge.io import iter_molecules, iter_smiles, read_molecules, read_smiles


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


class TestIterSmiles:
    def test_yields_molecules_lazily(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []

        def counting(s: str, **k: object) -> _FakeMol:
            calls.append(s)
            return _FakeMol(s)

        monkeypatch.setattr(_rdkit, "mol_from_smiles", counting)
        it = iter_smiles("CCO ethanol\nCCC propane")
        assert iter(it) is it  # a lazy iterator, not a materialized list
        assert calls == []  # nothing parsed until consumed
        first = next(it)
        assert isinstance(first, Molecule)
        assert first.name == "ethanol"
        assert calls == ["CCO"]  # only the first line parsed so far

    def test_skips_blanks_and_comments(self, fake_smiles: None) -> None:
        mols = list(iter_smiles("\n# header\nCCO ethanol\n\nCC(=O)O\n"))
        assert [m.name for m in mols] == ["ethanol", ""]

    def test_source_recorded(self, fake_smiles: None) -> None:
        (mol,) = list(iter_smiles("CCO x", source="lib.smi"))
        assert mol.metadata["source"] == "lib.smi"


class TestIterMolecules:
    def test_smiles_file(self, tmp_path, fake_smiles: None) -> None:  # noqa: ANN001
        p = tmp_path / "lib.smi"
        p.write_text("CCO ethanol\nCCC propane\n")
        mols = list(iter_molecules(p))
        assert [m.name for m in mols] == ["ethanol", "propane"]
        assert all(m.metadata["source"] == str(p) for m in mols)

    def test_sdf_dispatch_streams(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_records(path: str, **k: object) -> object:
            yield (_FakeMol("CCO"), "lig1")
            yield (_FakeMol("CCC"), "lig2")

        monkeypatch.setattr(_rdkit, "iter_sdf_records", fake_records)
        mols = list(iter_molecules("ligs.sdf"))
        assert [m.name for m in mols] == ["lig1", "lig2"]
        assert mols[0].metadata["source"] == "ligs.sdf"

    def test_explicit_format_overrides_extension(
        self, tmp_path, fake_smiles: None
    ) -> None:  # noqa: ANN001
        p = tmp_path / "molecules.txt"
        p.write_text("CCO ethanol\n")
        (mol,) = list(iter_molecules(p, format="smiles"))
        assert mol.name == "ethanol"

    def test_unknown_extension_raises_eagerly(self) -> None:
        # Format is resolved on the call, before any iteration.
        with pytest.raises(ValueError, match="can't infer"):
            iter_molecules("data.xyz")

    def test_unknown_format_raises_eagerly(self) -> None:
        with pytest.raises(ValueError, match="unknown molecule format"):
            iter_molecules("data.dat", format="mol2")


class TestRDKitAbsent:
    def test_read_molecules_sdf_raises(self) -> None:
        with pytest.raises(RDKitNotInstalledError):
            read_molecules("x.sdf")

    def test_read_smiles_raises(self) -> None:
        with pytest.raises(RDKitNotInstalledError):
            read_smiles("CCO ethanol")

    def test_iter_molecules_sdf_raises_on_consume(self) -> None:
        with pytest.raises(RDKitNotInstalledError):
            list(iter_molecules("x.sdf"))

    def test_iter_smiles_raises_on_consume(self) -> None:
        with pytest.raises(RDKitNotInstalledError):
            list(iter_smiles("CCO ethanol"))

    def test_iter_sdf_records_raises_before_open(self) -> None:
        # RDKit is checked before the file is touched, so a missing file
        # still surfaces the install error rather than FileNotFoundError.
        with pytest.raises(RDKitNotInstalledError):
            list(_rdkit.iter_sdf_records("does-not-exist.sdf"))
