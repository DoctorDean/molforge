"""Tests for the ProteinMPNN wrapper.

These don't require ProteinMPNN (or torch) to be installed. They
exercise:
  - Construction with parameter validation
  - Installation-path resolution
  - FASTA output parsing in isolation
  - Fixed-positions JSONL writing
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from molforge.generative import DesignedSequence, GenerativeEngineNotInstalledError
from molforge.wrappers.generative import ProteinMPNN


class TestConstruction:
    def test_defaults(self) -> None:
        engine = ProteinMPNN()
        assert engine.name == "ProteinMPNN"
        assert engine.model_name == "v_48_020"
        assert engine.num_seqs == 8
        assert engine.sampling_temp == 0.1
        assert engine.ca_only is False
        assert engine.use_soluble_model is False

    def test_custom_settings(self) -> None:
        engine = ProteinMPNN(
            model_name="v_48_010",
            num_seqs=16,
            sampling_temp=0.3,
            ca_only=True,
            use_soluble_model=True,
            omit_aas="XC",
            seed=42,
        )
        assert engine.model_name == "v_48_010"
        assert engine.num_seqs == 16
        assert engine.sampling_temp == 0.3
        assert engine.ca_only is True
        assert engine.use_soluble_model is True
        assert engine.omit_aas == "XC"
        assert engine.seed == 42

    def test_invalid_model_name(self) -> None:
        with pytest.raises(ValueError, match="unknown model_name"):
            ProteinMPNN(model_name="not_a_real_model")

    def test_invalid_sampling_temp(self) -> None:
        with pytest.raises(ValueError, match="sampling_temp"):
            ProteinMPNN(sampling_temp=0.0)
        with pytest.raises(ValueError, match="sampling_temp"):
            ProteinMPNN(sampling_temp=3.0)

    def test_invalid_num_seqs(self) -> None:
        with pytest.raises(ValueError, match="num_seqs"):
            ProteinMPNN(num_seqs=0)


class TestInstallationResolution:
    def test_missing_install_clear_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PROTEINMPNN_HOME", raising=False)
        engine = ProteinMPNN()
        with pytest.raises(GenerativeEngineNotInstalledError, match="ProteinMPNN not found"):
            engine._resolve_proteinmpnn_dir()

    def test_env_var_used(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        fake_install = tmp_path / "ProteinMPNN"
        fake_install.mkdir()
        (fake_install / "protein_mpnn_run.py").write_text("# placeholder\n")
        monkeypatch.setenv("PROTEINMPNN_HOME", str(fake_install))
        engine = ProteinMPNN()
        assert engine._resolve_proteinmpnn_dir() == fake_install

    def test_explicit_dir_without_script_errors(self, tmp_path: Path) -> None:
        engine = ProteinMPNN(proteinmpnn_dir=tmp_path)
        with pytest.raises(GenerativeEngineNotInstalledError, match=r"protein_mpnn_run\.py"):
            engine._resolve_proteinmpnn_dir()


# Sample ProteinMPNN FASTA output, lifted from the real format
_SAMPLE_FASTA = """>backbone, score=2.345, global_score=2.345, fixed_chains=[], designed_chains=['A'], model_name=v_48_020, git_hash=abc, seed=42
GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG
>T=0.1, sample=1, score=1.234, global_score=1.345, fixed_chains=[], designed_chains=['A'], model_name=v_48_020, git_hash=abc, seed=42
MKEVKAVAVLVEEDREYWLKHQGEALKRLDYALADTRRGRLAEEFNREEY
>T=0.1, sample=2, score=1.567, global_score=1.678, fixed_chains=[], designed_chains=['A'], model_name=v_48_020, git_hash=abc, seed=42
MAEEEAVAVLVEEDREYWLKHQGEALKRLDYALADTRRGRLAEEFNREEK
>T=0.1, sample=3, score=0.987, global_score=1.123, fixed_chains=[], designed_chains=['A'], model_name=v_48_020, git_hash=abc, seed=42
MKDVKAVAVLVEEDREYWLKHQGEALKRLDYALADTRRGRLAEEFNREES
"""


class TestFastaParsing:
    def test_returns_correct_number_of_designs(self) -> None:
        # First record is the native, so 4 records -> 3 designs
        designs = ProteinMPNN._parse_fasta(_SAMPLE_FASTA)
        assert len(designs) == 3

    def test_sequences_extracted(self) -> None:
        designs = ProteinMPNN._parse_fasta(_SAMPLE_FASTA)
        for d in designs:
            assert isinstance(d, DesignedSequence)
            assert len(d.sequence) == 50
            assert d.sequence.startswith("M")

    def test_sorted_by_score(self) -> None:
        designs = ProteinMPNN._parse_fasta(_SAMPLE_FASTA)
        scores = [d.score for d in designs]
        assert scores == sorted(scores)
        # Best is sample=3 with score=0.987
        assert designs[0].score == pytest.approx(0.987)

    def test_metadata_populated(self) -> None:
        designs = ProteinMPNN._parse_fasta(_SAMPLE_FASTA)
        for d in designs:
            assert d.metadata["engine"] == "ProteinMPNN"
            assert d.metadata["model_name"] == "v_48_020"
            assert d.metadata["T"] == pytest.approx(0.1)


class TestParseScoreHelper:
    def test_extracts_score(self) -> None:
        header = "T=0.1, sample=1, score=1.234, global_score=1.345"
        assert ProteinMPNN._parse_score(header) == pytest.approx(1.234)

    def test_missing_score_returns_nan(self) -> None:
        import math

        header = "T=0.1, sample=1, no_score_field_here"
        assert math.isnan(ProteinMPNN._parse_score(header))


class TestFixedPositionsJsonl:
    def test_writes_expected_format(self, tmp_path: Path) -> None:
        out = tmp_path / "fixed.jsonl"
        pdb = tmp_path / "my_backbone.pdb"
        pdb.write_text("dummy")
        ProteinMPNN._write_fixed_positions(out, pdb, {"A": [10, 11, 12], "B": [5]})
        loaded = json.loads(out.read_text())
        assert loaded == {"my_backbone": {"A": [10, 11, 12], "B": [5]}}


class TestDesignedSequenceRepr:
    def test_short_sequence(self) -> None:
        d = DesignedSequence(sequence="MKTV", score=1.5)
        assert "MKTV" in repr(d)
        assert "1.500" in repr(d)

    def test_long_sequence_truncated(self) -> None:
        d = DesignedSequence(sequence="A" * 100, score=1.0)
        assert "..." in repr(d)
        assert "A" * 100 not in repr(d)


class TestParseMetadataHelper:
    """Edge cases for the FASTA-header key=value parser."""

    def test_numeric_and_string_values_distinguished(self) -> None:
        meta = ProteinMPNN._parse_metadata("T=0.1, model_name=v_48_020, sample=3")
        assert meta["T"] == pytest.approx(0.1)
        assert isinstance(meta["T"], float)
        assert meta["model_name"] == "v_48_020"
        assert isinstance(meta["model_name"], str)
        assert meta["sample"] == pytest.approx(3.0)

    def test_tokens_without_equals_are_skipped(self) -> None:
        # A bare token (no '=') must not crash the parser or appear as a key.
        meta = ProteinMPNN._parse_metadata("score=1.2, garbage_token, T=0.1")
        assert meta["score"] == pytest.approx(1.2)
        assert meta["T"] == pytest.approx(0.1)
        assert "garbage_token" not in meta

    def test_engine_key_always_present(self) -> None:
        assert ProteinMPNN._parse_metadata("")["engine"] == "ProteinMPNN"


class TestParseOutputs:
    """`_parse_outputs` discovers the FASTA file ProteinMPNN writes."""

    def _make_seqs_dir(self, out_folder: Path) -> Path:
        seqs = out_folder / "seqs"
        seqs.mkdir(parents=True)
        return seqs

    def test_reads_single_fasta(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        seqs = self._make_seqs_dir(out)
        (seqs / "backbone.fa").write_text(_SAMPLE_FASTA, encoding="utf-8")
        designs = ProteinMPNN()._parse_outputs(out, "backbone")
        assert len(designs) == 3
        assert designs[0].score == pytest.approx(0.987)

    def test_picks_file_matching_pdb_stem(self, tmp_path: Path) -> None:
        # Multi-PDB run: several .fa files; the one matching the stem wins.
        out = tmp_path / "out"
        seqs = self._make_seqs_dir(out)
        (seqs / "other.fa").write_text(">native\nGGGG\n>d\nMKTV\n", encoding="utf-8")
        (seqs / "target.fa").write_text(_SAMPLE_FASTA, encoding="utf-8")
        designs = ProteinMPNN()._parse_outputs(out, "target")
        assert len(designs) == 3

    def test_accepts_fasta_extension(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        seqs = self._make_seqs_dir(out)
        (seqs / "backbone.fasta").write_text(_SAMPLE_FASTA, encoding="utf-8")
        assert len(ProteinMPNN()._parse_outputs(out, "backbone")) == 3

    def test_no_output_raises(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        self._make_seqs_dir(out)  # empty seqs dir
        with pytest.raises(RuntimeError, match="no FASTA output"):
            ProteinMPNN()._parse_outputs(out, "backbone")


class TestRunCli:
    """`_run_cli` is the subprocess-driving seam — exercised with a
    mocked ``subprocess.run`` so no real ProteinMPNN is needed."""

    def _fake_install(self, tmp_path: Path) -> Path:
        pmpnn = tmp_path / "ProteinMPNN"
        pmpnn.mkdir()
        (pmpnn / "protein_mpnn_run.py").write_text("# placeholder\n")
        return pmpnn

    def test_invokes_subprocess_and_parses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pmpnn = self._fake_install(tmp_path)
        pdb = tmp_path / "backbone.pdb"
        pdb.write_text("ATOM\n")

        captured: dict[str, object] = {}

        def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            # ProteinMPNN writes seqs/<stem>.fa under --out_folder.
            captured["cmd"] = cmd
            out_folder = Path(cmd[cmd.index("--out_folder") + 1])
            seqs = out_folder / "seqs"
            seqs.mkdir(parents=True, exist_ok=True)
            (seqs / f"{pdb.stem}.fa").write_text(_SAMPLE_FASTA, encoding="utf-8")
            return None

        monkeypatch.setattr("subprocess.run", fake_run)
        designs = ProteinMPNN()._run_cli(
            backbone=pdb,
            pmpnn_dir=pmpnn,
            chains_to_design=None,
            fixed_positions=None,
            timeout=None,
        )
        assert len(designs) == 3
        cmd = captured["cmd"]
        assert "--pdb_path" in cmd and str(pdb) in cmd

    def test_chains_and_fixed_positions_passed_through(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pmpnn = self._fake_install(tmp_path)
        pdb = tmp_path / "backbone.pdb"
        pdb.write_text("ATOM\n")

        captured: dict[str, object] = {}

        def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            captured["cmd"] = cmd
            out_folder = Path(cmd[cmd.index("--out_folder") + 1])
            seqs = out_folder / "seqs"
            seqs.mkdir(parents=True, exist_ok=True)
            (seqs / f"{pdb.stem}.fa").write_text(_SAMPLE_FASTA, encoding="utf-8")
            return None

        monkeypatch.setattr("subprocess.run", fake_run)
        ProteinMPNN()._run_cli(
            backbone=pdb,
            pmpnn_dir=pmpnn,
            chains_to_design="A",
            fixed_positions={"A": [1, 2, 3]},
            timeout=None,
        )
        cmd = captured["cmd"]
        assert "--pdb_path_chains" in cmd
        assert cmd[cmd.index("--pdb_path_chains") + 1] == "A"
        assert "--fixed_positions_jsonl" in cmd

    def test_subprocess_failure_becomes_runtime_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess

        pmpnn = self._fake_install(tmp_path)
        pdb = tmp_path / "backbone.pdb"
        pdb.write_text("ATOM\n")

        def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr="boom")

        monkeypatch.setattr("subprocess.run", fake_run)
        with pytest.raises(RuntimeError, match="ProteinMPNN failed"):
            ProteinMPNN()._run_cli(
                backbone=pdb,
                pmpnn_dir=pmpnn,
                chains_to_design=None,
                fixed_positions=None,
                timeout=None,
            )

    def test_engine_flags_appear_in_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pmpnn = self._fake_install(tmp_path)
        pdb = tmp_path / "backbone.pdb"
        pdb.write_text("ATOM\n")

        captured: dict[str, object] = {}

        def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            captured["cmd"] = cmd
            out_folder = Path(cmd[cmd.index("--out_folder") + 1])
            seqs = out_folder / "seqs"
            seqs.mkdir(parents=True, exist_ok=True)
            (seqs / f"{pdb.stem}.fa").write_text(_SAMPLE_FASTA, encoding="utf-8")
            return None

        monkeypatch.setattr("subprocess.run", fake_run)
        ProteinMPNN(ca_only=True, use_soluble_model=True)._run_cli(
            backbone=pdb,
            pmpnn_dir=pmpnn,
            chains_to_design=None,
            fixed_positions=None,
            timeout=None,
        )
        cmd = captured["cmd"]
        assert "--ca_only" in cmd
        assert "--use_soluble_model" in cmd

    def test_generate_resolves_install_then_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The public entry point: generate() resolves the install dir,
        # then delegates to _run_cli.
        pmpnn = self._fake_install(tmp_path)
        monkeypatch.setenv("PROTEINMPNN_HOME", str(pmpnn))
        pdb = tmp_path / "backbone.pdb"
        pdb.write_text("ATOM\n")

        def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            out_folder = Path(cmd[cmd.index("--out_folder") + 1])
            seqs = out_folder / "seqs"
            seqs.mkdir(parents=True, exist_ok=True)
            (seqs / f"{pdb.stem}.fa").write_text(_SAMPLE_FASTA, encoding="utf-8")
            return None

        monkeypatch.setattr("subprocess.run", fake_run)
        designs = ProteinMPNN().generate(pdb)
        assert len(designs) == 3
        assert designs[0].score == pytest.approx(0.987)
