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
