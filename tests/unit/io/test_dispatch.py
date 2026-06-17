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

    def test_planned_format_raises_not_implemented(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The dispatcher's planned-but-not-yet-implemented fallback
        path raises ``NotImplementedError`` with the planning hint.
        Exercised by monkeypatching a synthetic planned format —
        every format the dispatcher actually knows about is now
        implemented, so this guards the machinery, not any particular
        format."""
        from molforge.io import dispatch as dispatch_module

        monkeypatch.setitem(dispatch_module._EXT_TO_FORMAT, ".futurefmt", "futurefmt")
        monkeypatch.setitem(
            dispatch_module._PLANNED_READERS,
            "futurefmt",
            "FUTUREFMT reader is planned; see molforge.io.futurefmt",
        )
        bogus = tmp_path / "x.futurefmt"
        bogus.write_text("")
        with pytest.raises(NotImplementedError, match="FUTUREFMT"):
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

    def test_save_unknown_format_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Writing a format with no registered writer raises
        ``NotImplementedError``. Like the load-side equivalent, this
        is exercised with a synthetic format since every format the
        dispatcher recognises is now implemented."""
        from molforge.io import dispatch as dispatch_module

        monkeypatch.setitem(dispatch_module._EXT_TO_FORMAT, ".futurefmt", "futurefmt")
        p = load(FIXTURES / "pdb" / "dipeptide.pdb")
        with pytest.raises(NotImplementedError):
            save(p, tmp_path / "out.futurefmt")


class TestFetch:
    """Tests for io.fetch. The network path is exercised by mocking
    urllib.request.urlopen so no real download happens in CI."""

    def test_empty_id_raises(self) -> None:
        from molforge.io import fetch

        with pytest.raises(ValueError, match="non-empty"):
            fetch("")

    def test_whitespace_id_raises(self) -> None:
        from molforge.io import fetch

        with pytest.raises(ValueError, match="non-empty"):
            fetch("   ")

    def test_bad_source_raises(self) -> None:
        from molforge.io import fetch

        with pytest.raises(ValueError, match="source must be"):
            fetch("1UBQ", source="ftp")

    def test_bad_format_raises(self) -> None:
        from molforge.io import fetch

        with pytest.raises(ValueError, match="format must be"):
            fetch("1UBQ", format="xml")

    def test_rcsb_pdb_success(self) -> None:
        """A successful RCSB PDB fetch parses the downloaded text."""
        from unittest.mock import MagicMock, patch

        from molforge.io import fetch

        pdb_text = (
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\nEND\n"
        )
        mock_resp = MagicMock()
        mock_resp.read.return_value = pdb_text.encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = False

        with patch("urllib.request.urlopen", return_value=mock_resp) as m:
            protein = fetch("1ubq")

        assert isinstance(protein, Protein)
        assert protein.n_atoms == 1
        # ID should be upper-cased into the RCSB URL.
        called_url = m.call_args[0][0]
        assert called_url == "https://files.rcsb.org/download/1UBQ.pdb"

    def test_alphafold_source_builds_correct_url(self) -> None:
        from unittest.mock import MagicMock, patch

        from molforge.io import fetch

        pdb_text = (
            "ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00 90.00           C\nEND\n"
        )
        mock_resp = MagicMock()
        mock_resp.read.return_value = pdb_text.encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = False

        with patch("urllib.request.urlopen", return_value=mock_resp) as m:
            fetch("P00520", source="alphafold")

        called_url = m.call_args[0][0]
        assert called_url == ("https://alphafold.ebi.ac.uk/files/AF-P00520-F1-model_v4.pdb")

    def test_cif_format_builds_cif_url(self) -> None:
        from unittest.mock import MagicMock, patch

        from molforge.io import fetch

        cif_text = (
            "data_1ubq\n#\nloop_\n"
            "_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n"
            "_atom_site.label_atom_id\n_atom_site.label_comp_id\n"
            "_atom_site.label_asym_id\n_atom_site.label_seq_id\n"
            "_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
            "_atom_site.occupancy\n_atom_site.B_iso_or_equiv\n"
            "ATOM 1 C CA ALA A 1 0.000 0.000 0.000 1.00 20.00\n#\n"
        )
        mock_resp = MagicMock()
        mock_resp.read.return_value = cif_text.encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = False

        with patch("urllib.request.urlopen", return_value=mock_resp) as m:
            protein = fetch("1ubq", format="cif")

        assert protein.n_atoms == 1
        assert m.call_args[0][0].endswith(".cif")

    def test_http_error_becomes_oserror(self) -> None:
        """A 404 (non-existent ID) surfaces as a clear OSError."""
        import urllib.error
        from unittest.mock import patch

        from molforge.io import fetch

        err = urllib.error.HTTPError(
            url="https://files.rcsb.org/download/ZZZZ.pdb",
            code=404,
            msg="Not Found",
            hdrs=None,  # type: ignore[arg-type]
            fp=None,
        )
        with (
            patch("urllib.request.urlopen", side_effect=err),
            pytest.raises(OSError, match="HTTP 404"),
        ):
            fetch("ZZZZ")

    def test_network_error_becomes_oserror(self) -> None:
        """A connection failure surfaces as a clear OSError."""
        import urllib.error
        from unittest.mock import patch

        from molforge.io import fetch

        err = urllib.error.URLError("Name or service not known")
        with (
            patch("urllib.request.urlopen", side_effect=err),
            pytest.raises(OSError, match="could not reach"),
        ):
            fetch("1UBQ")
