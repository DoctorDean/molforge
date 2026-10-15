"""Tests for the MMseqs2 homology-search wrapper.

No test runs mmseqs: the binary isn't in CI and the database is hundreds
of gigabytes. What's checked is everything around the subprocess — command
construction, the database and binary preconditions, caching, provenance,
and parsing — through the single `_run_search` seam.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from molforge.sequence import MultipleSequenceAlignment
from molforge.wrappers.msa import (
    MMseqs2,
    MMseqs2NotInstalledError,
    MSADatabaseNotFoundError,
)
from molforge.wrappers.msa.mmseqs2 import MSA_DATABASE_ENV

A3M = ">query\nMKQHKA\n>sp|P1|HIT OX=9606\nMKAHKA\n>sp|P2|HIT2 OX=10090\nMK-HKAll\n"


@pytest.fixture
def fake_db(tmp_path: Path) -> str:
    """A path that looks like an MMseqs2 database to the precondition check."""
    prefix = tmp_path / "uniref90"
    (tmp_path / "uniref90.dbtype").write_bytes(b"\x00")
    return str(prefix)


class TestConstruction:
    def test_defaults(self) -> None:
        engine = MMseqs2()
        assert engine.database is None
        assert engine.sensitivity is None
        assert engine.executable == "mmseqs"

    def test_database_is_stringified(self, tmp_path: Path) -> None:
        assert MMseqs2(database=tmp_path / "db").database == str(tmp_path / "db")

    @pytest.mark.parametrize("bad", [0.5, 10.5, -1])
    def test_sensitivity_range_enforced(self, bad: float) -> None:
        with pytest.raises(ValueError, match=r"runs 1\.0-10\.0"):
            MMseqs2(sensitivity=bad)

    def test_sensitivity_bounds_are_inclusive(self) -> None:
        assert MMseqs2(sensitivity=1.0).sensitivity == 1.0
        assert MMseqs2(sensitivity=10.0).sensitivity == 10.0

    def test_max_seqs_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_seqs must be >= 1"):
            MMseqs2(max_seqs=0)


class TestDatabaseResolution:
    def test_missing_database_names_the_build_command(self) -> None:
        """The error has to carry the fix: nobody guesses `mmseqs databases`."""
        import os

        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop(MSA_DATABASE_ENV, None)
            with pytest.raises(MSADatabaseNotFoundError, match="mmseqs databases UniRef90"):
                MMseqs2()._resolve_database()

    def test_env_var_is_the_fallback(self, fake_db: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(MSA_DATABASE_ENV, fake_db)
        assert MMseqs2()._resolve_database() == fake_db

    def test_constructor_wins_over_env(self, fake_db: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(MSA_DATABASE_ENV, "/elsewhere/db")
        assert MMseqs2(database=fake_db)._resolve_database() == fake_db

    def test_path_that_is_not_a_database_is_rejected(self, tmp_path: Path) -> None:
        """Pointing at the containing directory is the common mistake."""
        with pytest.raises(MSADatabaseNotFoundError, match="dbtype"):
            MMseqs2(database=str(tmp_path / "nothing-here"))._resolve_database()

    def test_error_explains_the_prefix_convention(self, tmp_path: Path) -> None:
        with pytest.raises(MSADatabaseNotFoundError, match="prefix"):
            MMseqs2(database=str(tmp_path / "absent"))._resolve_database()


class TestBinaryResolution:
    def test_missing_binary_gives_an_install_hint(self) -> None:
        with (
            patch("shutil.which", return_value=None),
            pytest.raises(MMseqs2NotInstalledError, match="conda install"),
        ):
            MMseqs2()._require_mmseqs()

    def test_resolved_path_is_returned(self) -> None:
        with patch("shutil.which", return_value="/opt/bin/mmseqs"):
            assert MMseqs2()._require_mmseqs() == "/opt/bin/mmseqs"


class TestSearchOptions:
    def test_unset_options_are_omitted_entirely(self) -> None:
        """Omitted, not defaulted — MMseqs2's own defaults are better than
        ours and change between releases."""
        assert MMseqs2()._search_options() == []

    def test_sensitivity_becomes_dash_s(self) -> None:
        assert MMseqs2(sensitivity=7.5)._search_options() == ["-s", "7.5"]

    def test_all_options_together(self) -> None:
        opts = MMseqs2(sensitivity=7.5, max_seqs=300, threads=8)._search_options()
        assert opts == ["-s", "7.5", "--max-seqs", "300", "--threads", "8"]


class TestSearch:
    def _engine(self, db: str) -> MMseqs2:
        return MMseqs2(database=db)

    def test_parses_the_alignment(self, fake_db: str) -> None:
        with (
            patch.object(MMseqs2, "_run_search", return_value=A3M),
            patch("shutil.which", return_value="/opt/bin/mmseqs"),
        ):
            msa = self._engine(fake_db).search("MKQHKA")
        assert isinstance(msa, MultipleSequenceAlignment)
        assert msa.depth == 3
        assert msa.query_sequence == "MKQHKA"

    def test_hits_are_labelled_with_the_database(self, fake_db: str) -> None:
        with (
            patch.object(MMseqs2, "_run_search", return_value=A3M),
            patch("shutil.which", return_value="/opt/bin/mmseqs"),
        ):
            msa = self._engine(fake_db).search("MKQHKA")
        assert {h.source_database for h in msa.hits} == {"uniref90"}

    def test_pairing_keys_survive(self, fake_db: str) -> None:
        with (
            patch.object(MMseqs2, "_run_search", return_value=A3M),
            patch("shutil.which", return_value="/opt/bin/mmseqs"),
        ):
            msa = self._engine(fake_db).search("MKQHKA")
        assert [h.pairing_key for h in msa.hits] == ["9606", "10090"]

    def test_empty_sequence_rejected(self, fake_db: str) -> None:
        with pytest.raises(ValueError, match="sequence is empty"):
            self._engine(fake_db).search("   ")

    def test_sequence_is_normalised(self, fake_db: str) -> None:
        seen = {}

        def capture(self, binary, database, query):  # type: ignore[no-untyped-def]
            seen["query"] = query
            return A3M

        with (
            patch.object(MMseqs2, "_run_search", capture),
            patch("shutil.which", return_value="/opt/bin/mmseqs"),
        ):
            self._engine(fake_db).search(" mkqhka \n")
        assert seen["query"] == "MKQHKA"


class TestCaching:
    def test_second_search_does_not_re_run(self, fake_db: str) -> None:
        """A deep search is minutes to hours; the same query against the
        same database is the same answer."""
        calls = []

        def once(self, binary, database, query):  # type: ignore[no-untyped-def]
            calls.append(query)
            return A3M

        with (
            patch.object(MMseqs2, "_run_search", once),
            patch("shutil.which", return_value="/opt/bin/mmseqs"),
        ):
            engine = MMseqs2(database=fake_db)
            first = engine.search("MKQHKA")
            second = engine.search("MKQHKA")

        assert len(calls) == 1
        assert first.to_a3m() == second.to_a3m()

    def test_cached_alignment_keeps_its_pairing_metadata(self, fake_db: str) -> None:
        """a3m can't carry source_database or pairing_key, so a cache that
        round-tripped through it would quietly demote every hit."""
        with (
            patch.object(MMseqs2, "_run_search", return_value=A3M),
            patch("shutil.which", return_value="/opt/bin/mmseqs"),
        ):
            engine = MMseqs2(database=fake_db)
            engine.search("MKQHKA")
            cached = engine.search("MKQHKA")

        assert [h.pairing_key for h in cached.hits] == ["9606", "10090"]
        assert {h.source_database for h in cached.hits} == {"uniref90"}

    def test_different_databases_do_not_share_an_entry(self, tmp_path: Path) -> None:
        for name in ("db_a", "db_b"):
            (tmp_path / f"{name}.dbtype").write_bytes(b"\x00")
        from molforge.cache import cache_key

        keys = {
            cache_key(MMseqs2(database=str(tmp_path / n))._build_provenance("MKQ"))
            for n in ("db_a", "db_b")
        }
        assert len(keys) == 2

    def test_sensitivity_splits_the_cache(self, fake_db: str) -> None:
        from molforge.cache import cache_key

        keys = {
            cache_key(MMseqs2(database=fake_db, sensitivity=s)._build_provenance("MKQ"))
            for s in (1.0, 7.5, None)
        }
        assert len(keys) == 3

    def test_threads_do_not_split_the_cache(self, fake_db: str) -> None:
        """How many cores ran the search doesn't change its result."""
        from molforge.cache import cache_key

        a = MMseqs2(database=fake_db, threads=1)._build_provenance("MKQ")
        b = MMseqs2(database=fake_db, threads=32)._build_provenance("MKQ")
        assert cache_key(a) == cache_key(b)


class TestProvenance:
    def test_records_engine_and_operation(self, fake_db: str) -> None:
        prov = MMseqs2(database=fake_db)._build_provenance("MKQ")
        assert prov.engine == "MMseqs2"
        assert prov.operation == "search_msa"

    def test_records_the_search_configuration(self, fake_db: str) -> None:
        prov = MMseqs2(database=fake_db, sensitivity=7.5, max_seqs=300)._build_provenance("MKQ")
        assert prov.parameters["database"] == fake_db
        assert prov.parameters["sensitivity"] == 7.5
        assert prov.parameters["max_seqs"] == 300

    def test_attaches_to_a_searched_alignment_chain(self, fake_db: str) -> None:
        from molforge.reproducibility import pipeline_manifest

        manifest = pipeline_manifest(MMseqs2(database=fake_db)._build_provenance("MKQ"))
        assert manifest.steps[0].engine == "MMseqs2"
        assert manifest.steps[0].operation == "search_msa"


class TestInvocationFailures:
    def test_timeout_suggests_a_longer_one(self) -> None:
        with (
            patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="mmseqs", timeout=1)),
            pytest.raises(RuntimeError, match="timed out"),
        ):
            MMseqs2(timeout=1)._invoke(["mmseqs", "search"], step="search")

    def test_nonzero_exit_echoes_stderr(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="Invalid database"
        )
        with (
            patch("subprocess.run", return_value=completed),
            pytest.raises(RuntimeError, match="Invalid database"),
        ):
            MMseqs2()._invoke(["mmseqs", "search"], step="search")

    def test_failure_names_the_step(self) -> None:
        completed = subprocess.CompletedProcess(args=[], returncode=2, stdout="", stderr="boom")
        with (
            patch("subprocess.run", return_value=completed),
            pytest.raises(RuntimeError, match="result2msa"),
        ):
            MMseqs2()._invoke(["mmseqs"], step="result2msa")


class TestA3mFormatMode:
    def test_uses_mode_five(self) -> None:
        """Read off MMseqs2's Parameters.cpp: 5 is A3M, 6 is A3M *with
        alignment info*, which a folding engine does not read. The wiki
        and the web disagree about this, so it is pinned here."""
        from molforge.wrappers.msa.mmseqs2 import _A3M_FORMAT_MODE

        assert _A3M_FORMAT_MODE == "5"


class TestReplay:
    def test_search_is_replayable(self, fake_db: str) -> None:
        from molforge.reproducibility import replay

        with (
            patch.object(MMseqs2, "_run_search", return_value=A3M),
            patch("shutil.which", return_value="/opt/bin/mmseqs"),
        ):
            prov = MMseqs2(database=fake_db)._build_provenance("MKQHKA")
            out = replay(prov)
        assert isinstance(out, MultipleSequenceAlignment)
        assert out.query_sequence == "MKQHKA"

    def test_engine_resolves_from_the_registry(self) -> None:
        from molforge.reproducibility import _resolve_engine

        assert _resolve_engine("MMseqs2").__name__ == "MMseqs2"
