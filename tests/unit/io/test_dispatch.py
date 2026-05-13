"""Tests for the load / save / fetch dispatch layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from molforge.core import Protein
from molforge.io import load, save

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


class TestLoadDispatch:
    def test_load_pdb_by_extension(self) -> None:
        p = load(FIXTURES / "pdb" / "dipeptide.pdb")
        assert isinstance(p, Protein)
        assert p.n_atoms == 10

    def test_load_fasta_by_extension(self) -> None:
        records = load(FIXTURES / "fasta" / "simple.fasta")
        assert isinstance(records, list)
        assert len(records) == 2

    def test_load_explicit_format(self, tmp_path: Path) -> None:
        # Copy with a non-standard extension
        src = (FIXTURES / "pdb" / "dipeptide.pdb").read_text()
        weird = tmp_path / "structure.dat"
        weird.write_text(src)
        p = load(weird, format="pdb")
        assert p.n_atoms == 10

    def test_unknown_extension_raises(self, tmp_path: Path) -> None:
        bogus = tmp_path / "x.xyz"
        bogus.write_text("")
        with pytest.raises(ValueError, match="could not infer format"):
            load(bogus)

    def test_stub_format_raises_not_implemented(self, tmp_path: Path) -> None:
        bogus = tmp_path / "x.cif"
        bogus.write_text("")
        with pytest.raises(NotImplementedError, match="mmCIF"):
            load(bogus)


class TestSaveDispatch:
    def test_save_pdb_by_extension(self, tmp_path: Path) -> None:
        p = load(FIXTURES / "pdb" / "dipeptide.pdb")
        out = tmp_path / "out.pdb"
        save(p, out)
        assert out.exists()
        text = out.read_text()
        assert "ATOM" in text

    def test_save_fasta_by_extension(self, tmp_path: Path) -> None:
        recs = load(FIXTURES / "fasta" / "simple.fasta")
        out = tmp_path / "out.fa"
        save(recs, out)
        text = out.read_text()
        assert text.startswith(">")

    def test_save_stub_format_raises(self, tmp_path: Path) -> None:
        p = load(FIXTURES / "pdb" / "dipeptide.pdb")
        with pytest.raises(NotImplementedError):
            save(p, tmp_path / "out.cif")


class TestFetch:
    def test_fetch_is_stubbed_for_now(self) -> None:
        from molforge.io import fetch

        with pytest.raises(NotImplementedError):
            fetch("1UBQ")
