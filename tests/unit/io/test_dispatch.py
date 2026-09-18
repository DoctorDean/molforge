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


class TestFetchMany:
    """Batch fetch. urllib.request.urlopen is mocked — no real downloads."""

    _PDB = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\nEND\n"

    @classmethod
    def _resp(cls) -> object:
        from unittest.mock import MagicMock

        resp = MagicMock()
        resp.read.return_value = cls._PDB.encode("utf-8")
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False
        return resp

    def test_fetches_all_in_order(self) -> None:
        from unittest.mock import patch

        from molforge.io import fetch_many

        with patch("urllib.request.urlopen", return_value=self._resp()) as mock:
            proteins = fetch_many(["1ubq", "4hhb"])

        assert len(proteins) == 2
        assert all(isinstance(p, Protein) for p in proteins)
        urls = [call[0][0] for call in mock.call_args_list]
        assert urls == [
            "https://files.rcsb.org/download/1UBQ.pdb",
            "https://files.rcsb.org/download/4HHB.pdb",
        ]

    def test_empty_iterable_returns_empty(self) -> None:
        from molforge.io import fetch_many

        assert fetch_many([]) == []

    def test_bad_on_error_raises(self) -> None:
        from molforge.io import fetch_many

        with pytest.raises(ValueError, match="on_error must be"):
            fetch_many(["1ubq"], on_error="ignore")

    def test_on_error_raise_propagates(self) -> None:
        import urllib.error
        from unittest.mock import patch

        from molforge.io import fetch_many

        err = urllib.error.HTTPError(url="u", code=404, msg="NF", hdrs=None, fp=None)  # type: ignore[arg-type]
        with patch("urllib.request.urlopen", side_effect=err), pytest.raises(OSError):
            fetch_many(["ZZZZ", "1ubq"])

    def test_on_error_skip_drops_failures(self) -> None:
        import urllib.error
        from unittest.mock import patch

        from molforge.io import fetch_many

        err = urllib.error.HTTPError(url="u", code=404, msg="NF", hdrs=None, fp=None)  # type: ignore[arg-type]
        with patch("urllib.request.urlopen", side_effect=[err, self._resp()]):
            proteins = fetch_many(["ZZZZ", "1ubq"], on_error="skip")

        assert len(proteins) == 1
        assert isinstance(proteins[0], Protein)


class TestFetchCache:
    """fetch() routes downloads through molforge.cache (GAP-0008).

    The autouse ``_isolate_cache`` fixture in tests/conftest.py already
    points the default cache at a per-test temp dir, so these exercise
    real cache reads and writes without touching ``~/.cache/molforge``.
    """

    _PDB = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\nEND\n"

    @classmethod
    def _resp(cls, text: str | None = None) -> object:
        from unittest.mock import MagicMock

        resp = MagicMock()
        resp.read.return_value = (text if text is not None else cls._PDB).encode("utf-8")
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False
        return resp

    def test_second_fetch_makes_no_request(self) -> None:
        """The whole point: N fetches of one entry cost one download."""
        from unittest.mock import patch

        from molforge.io import fetch

        with patch("urllib.request.urlopen", return_value=self._resp()) as m:
            first = fetch("1ubq")
            second = fetch("1ubq")
            third = fetch("1ubq")

        assert m.call_count == 1
        assert first.n_atoms == second.n_atoms == third.n_atoms == 1

    def test_cache_hit_matches_a_cache_miss(self) -> None:
        """A hit must be indistinguishable from a miss.

        This is why the entry stores the downloaded *text* rather than
        a serialized Protein: re-encoding through molforge's own mmCIF
        writer would make the second call return something subtly
        different from the first.
        """
        from unittest.mock import patch

        from molforge.io import fetch

        with patch("urllib.request.urlopen", return_value=self._resp()):
            miss = fetch("1ubq")
        with patch("urllib.request.urlopen", side_effect=AssertionError("no request")):
            hit = fetch("1ubq")

        assert hit.n_atoms == miss.n_atoms
        assert hit.sequence == miss.sequence
        assert hit.coords.tolist() == miss.coords.tolist()
        assert hit.atom_array.atom_name.tolist() == miss.atom_array.atom_name.tolist()
        assert hit.atom_array.element.tolist() == miss.atom_array.element.tolist()

    def test_id_case_shares_one_slot(self) -> None:
        """Both URL builders upper-case, so the ID case cannot split the key."""
        from unittest.mock import patch

        from molforge.io import fetch

        with patch("urllib.request.urlopen", return_value=self._resp()) as m:
            fetch("1ubq")
            fetch("1UBQ")
            fetch("  1Ubq  ")

        assert m.call_count == 1

    def test_source_and_format_separate_slots(self) -> None:
        """Same ID, different request — must not collide."""
        from unittest.mock import patch

        from molforge.io import fetch

        cif = (
            "data_x\n#\nloop_\n"
            "_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n"
            "_atom_site.label_atom_id\n_atom_site.label_comp_id\n"
            "_atom_site.label_asym_id\n_atom_site.label_seq_id\n"
            "_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
            "_atom_site.occupancy\n_atom_site.B_iso_or_equiv\n"
            "ATOM 1 C CA ALA A 1 0.000 0.000 0.000 1.00 20.00\n#\n"
        )
        with patch("urllib.request.urlopen") as m:
            m.side_effect = [self._resp(), self._resp(cif), self._resp()]
            fetch("1ubq")
            fetch("1ubq", format="cif")
            fetch("1ubq", source="alphafold")

        assert m.call_count == 3

    def test_timeout_does_not_split_the_key(self) -> None:
        """timeout changes how long we wait, never what comes back."""
        from unittest.mock import patch

        from molforge.io import fetch

        with patch("urllib.request.urlopen", return_value=self._resp()) as m:
            fetch("1ubq", timeout=5.0)
            fetch("1ubq", timeout=60.0)

        assert m.call_count == 1

    def test_cache_false_neither_reads_nor_writes(self) -> None:
        from unittest.mock import patch

        from molforge.io import fetch

        with patch("urllib.request.urlopen", return_value=self._resp()) as m:
            fetch("1ubq", cache=False)
            fetch("1ubq", cache=False)
        assert m.call_count == 2

        # Nothing was written, so a normal fetch still has to download.
        with patch("urllib.request.urlopen", return_value=self._resp()) as m:
            fetch("1ubq")
        assert m.call_count == 1

    def test_force_refresh_redownloads_and_replaces(self) -> None:
        """force_refresh must also *replace* the entry.

        Cache.put refuses to overwrite, so without an explicit
        invalidate the refreshed bytes would be dropped and the next
        plain fetch would serve the stale copy again.
        """
        from unittest.mock import patch

        from molforge.io import fetch

        revised = (
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
            "ATOM      2  CA  GLY A   2       1.000   1.000   1.000  1.00 20.00           C\n"
            "END\n"
        )

        with patch("urllib.request.urlopen", return_value=self._resp()):
            assert fetch("1ubq").n_atoms == 1

        with patch("urllib.request.urlopen", return_value=self._resp(revised)) as m:
            assert fetch("1ubq", force_refresh=True).n_atoms == 2
        assert m.call_count == 1

        # The refreshed copy is what a subsequent plain fetch serves.
        with patch("urllib.request.urlopen", side_effect=AssertionError("no request")):
            assert fetch("1ubq").n_atoms == 2

    def test_failed_download_caches_nothing(self) -> None:
        """A 404 must not poison the slot."""
        import urllib.error
        from unittest.mock import patch

        from molforge.io import fetch

        err = urllib.error.HTTPError(url="u", code=404, msg="NF", hdrs=None, fp=None)  # type: ignore[arg-type]
        with patch("urllib.request.urlopen", side_effect=err), pytest.raises(OSError):
            fetch("ZZZZ")

        with patch("urllib.request.urlopen", return_value=self._resp()) as m:
            fetch("ZZZZ")
        assert m.call_count == 1

    def test_disabled_env_var_is_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MOLFORGE_CACHE=disabled turns the download cache off too."""
        from unittest.mock import patch

        import molforge.cache as cache_module
        from molforge.io import fetch

        monkeypatch.setenv(cache_module.CACHE_DISABLED_ENV, "disabled")
        cache_module._reset_default_cache_for_testing()

        with patch("urllib.request.urlopen", return_value=self._resp()) as m:
            fetch("1ubq")
            fetch("1ubq")
        assert m.call_count == 2

    def test_entry_holds_the_downloaded_bytes(self) -> None:
        """The on-disk entry is the server's file, readable as-is."""
        from unittest.mock import patch

        from molforge.cache import get_default_cache
        from molforge.io import fetch
        from molforge.io.dispatch import _fetch_provenance

        with patch("urllib.request.urlopen", return_value=self._resp()):
            fetch("1ubq")

        entry = get_default_cache().path_for(_fetch_provenance("1ubq", "rcsb", "pdb"))
        assert (entry / "download.txt").read_text(encoding="utf-8") == self._PDB
        assert (entry / "type").read_text(encoding="utf-8").strip() == "structure_text"

    def test_fetch_many_reuses_cached_entries(self) -> None:
        """A re-run of the same ID set costs no downloads."""
        from unittest.mock import patch

        from molforge.io import fetch_many

        with patch("urllib.request.urlopen", return_value=self._resp()) as m:
            fetch_many(["1ubq", "4hhb"])
        assert m.call_count == 2

        with patch("urllib.request.urlopen", side_effect=AssertionError("no request")):
            proteins = fetch_many(["1ubq", "4hhb"])
        assert len(proteins) == 2

    def test_fetch_many_resumes_after_a_partial_run(self) -> None:
        """The crash-restart case GAP-0008 calls out: only the tail re-downloads."""
        import urllib.error
        from unittest.mock import patch

        from molforge.io import fetch_many

        err = urllib.error.URLError("boom")
        with (
            patch("urllib.request.urlopen", side_effect=[self._resp(), err]),
            pytest.raises(OSError),
        ):
            fetch_many(["1ubq", "4hhb"])

        # Restart: 1UBQ is already banked, so only 4HHB goes over the wire.
        with patch("urllib.request.urlopen", return_value=self._resp()) as m:
            proteins = fetch_many(["1ubq", "4hhb"])
        assert m.call_count == 1
        assert len(proteins) == 2

    def test_fetch_many_force_refresh_forwards(self) -> None:
        from unittest.mock import patch

        from molforge.io import fetch_many

        with patch("urllib.request.urlopen", return_value=self._resp()) as m:
            fetch_many(["1ubq"])
        assert m.call_count == 1

        with patch("urllib.request.urlopen", return_value=self._resp()) as m:
            fetch_many(["1ubq"], force_refresh=True)
        assert m.call_count == 1
