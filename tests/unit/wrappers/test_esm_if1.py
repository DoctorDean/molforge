"""Tests for the ESM-IF1 inverse-folding wrapper.

ESM-IF1 needs the fair-esm package and ~145 MB of weights, plus
torch-geometric for the GVP-GNN layers. These aren't usually
available in CI by default, so the test strategy mirrors test_esmfold
and test_proteinmpnn: end-to-end tests skip when fair-esm is missing,
and the data-conversion seams are driven directly with mocked model
output.

The seams worth testing without the model:

1. Constructor validation (no torch / no esm needed).
2. _compute_recovery — pure-Python utility.
3. _sample_designs with self._model / self._alphabet mocked — drives
   the DesignedSequence construction, scoring, recovery, and
   Provenance attachment.
4. _materialise_backbone — Protein vs path handling.

The end-to-end TestRealESMIF1 class runs the full pipeline against a
small fixture when fair-esm is on PATH.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from molforge.core import AtomArray, Protein
from molforge.core import metadata_keys as mk
from molforge.core.provenance import Provenance
from molforge.generative import DesignedSequence, GenerativeEngineNotInstalledError
from molforge.wrappers.generative import ESMIF1
from molforge.wrappers.generative.esm_if1 import _compute_recovery


def _fair_esm_available() -> bool:
    return importlib.util.find_spec("esm") is not None


def _tiny_backbone() -> Protein:
    """Trivial Protein for tests that don't touch the real model."""
    arr = AtomArray(2)
    arr.coords[:] = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float32)
    arr.element[:] = ["N", "C"]
    arr.atom_name[:] = ["N", "CA"]
    arr.residue_name[:] = "ALA"
    arr.residue_id[:] = 1
    arr.chain_id[:] = "A"
    return Protein(arr, name="test_backbone")


# ---------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------


class TestConstruction:
    def test_defaults(self) -> None:
        engine = ESMIF1()
        assert engine.model_name == "esm_if1_gvp4_t16_142M_UR50"
        assert engine.device is None
        assert engine.num_seqs == 8
        assert engine.temperature == 1.0
        assert engine.score_sequences is True
        assert engine.compute_recovery is True
        assert engine.seed is None
        # Lazy: nothing loaded yet.
        assert engine._model is None
        assert engine._alphabet is None

    def test_custom_options(self) -> None:
        engine = ESMIF1(
            device="cuda",
            num_seqs=16,
            temperature=0.1,
            score_sequences=False,
            compute_recovery=False,
            seed=42,
        )
        assert engine.device == "cuda"
        assert engine.num_seqs == 16
        assert engine.temperature == 0.1
        assert engine.score_sequences is False
        assert engine.compute_recovery is False
        assert engine.seed == 42

    def test_invalid_num_seqs(self) -> None:
        with pytest.raises(ValueError, match="num_seqs must be >= 1"):
            ESMIF1(num_seqs=0)
        with pytest.raises(ValueError, match="num_seqs must be >= 1"):
            ESMIF1(num_seqs=-1)

    def test_invalid_temperature(self) -> None:
        with pytest.raises(ValueError, match="temperature must be > 0"):
            ESMIF1(temperature=0.0)
        with pytest.raises(ValueError, match="temperature must be > 0"):
            ESMIF1(temperature=-0.5)

    def test_construction_does_not_load_model(self) -> None:
        """Construction is lazy — no torch needed, no network calls,
        no model loading. Allows molforge users to import ESMIF1
        even when fair-esm isn't installed."""
        engine = ESMIF1()
        assert engine._model is None
        assert engine._alphabet is None


# ---------------------------------------------------------------------
# Missing-dependency error path
# ---------------------------------------------------------------------


class TestMissingDependency:
    @pytest.mark.skipif(_fair_esm_available(), reason="fair-esm is installed")
    def test_friendly_error_when_esm_missing(self) -> None:
        """When fair-esm isn't installed, _ensure_loaded raises a
        GenerativeEngineNotInstalledError with install guidance."""
        engine = ESMIF1()
        with pytest.raises(GenerativeEngineNotInstalledError) as exc:
            engine._ensure_loaded()
        msg = str(exc.value)
        assert "fair-esm" in msg
        # Mentions the molforge[ml] install path.
        assert "molforge[ml]" in msg or "ml" in msg


# ---------------------------------------------------------------------
# _compute_recovery (pure-Python utility)
# ---------------------------------------------------------------------


class TestComputeRecovery:
    def test_perfect_match(self) -> None:
        assert _compute_recovery("AAGG", "AAGG") == 1.0

    def test_partial_match(self) -> None:
        # 3 of 4 positions match.
        assert _compute_recovery("AAGG", "AAGC") == 0.75

    def test_no_match(self) -> None:
        assert _compute_recovery("AAGG", "CCCC") == 0.0

    def test_empty_designed(self) -> None:
        assert _compute_recovery("", "AAGG") == 0.0

    def test_empty_native(self) -> None:
        assert _compute_recovery("AAGG", "") == 0.0

    def test_both_empty(self) -> None:
        assert _compute_recovery("", "") == 0.0

    def test_length_mismatch_uses_overlap(self) -> None:
        """When the lengths differ (shouldn't normally happen but
        defensive coding), recovery is computed over the overlap."""
        # "AGG" vs "AAGG"[:3] = "AAG":
        # A=A (match), G!=A (mismatch), G=G (match) -> 2/3.
        assert _compute_recovery("AGG", "AAGG") == pytest.approx(2 / 3)


# ---------------------------------------------------------------------
# _sample_designs seam (mocked model)
# ---------------------------------------------------------------------


class TestSampleDesigns:
    """Drive _sample_designs with mocked esm.inverse_folding.util
    functions to exercise the DesignedSequence construction path
    without needing the real model."""

    @patch.object(ESMIF1, "_score_one")
    @patch.object(ESMIF1, "_sample_one")
    def test_basic_sampling(
        self,
        mock_sample: MagicMock,
        mock_score: MagicMock,
    ) -> None:
        # Three sequential samples, each with a different
        # log-likelihood.
        mock_sample.side_effect = ["AAAA", "GGGG", "MAGG"]
        mock_score.side_effect = [
            -1.0,
            -0.5,
            -1.2,
        ]

        engine = ESMIF1(num_seqs=3)
        engine._model = MagicMock()
        engine._alphabet = MagicMock()

        coords = "fake_coords"
        designs = engine._sample_designs(coords, native_seq="MAGG")

        assert len(designs) == 3
        # Each design is a DesignedSequence with sequence + score +
        # recovery + metadata.
        for design in designs:
            assert isinstance(design, DesignedSequence)
            assert design.metadata["engine"] == "ESM-IF1"
            assert design.metadata["temperature"] == 1.0

        # Scores: negative log-likelihood (so positive numbers, lower
        # = better).
        assert designs[0].score == 1.0  # -(-1.0)
        assert designs[1].score == 0.5  # -(-0.5)
        assert designs[2].score == 1.2  # -(-1.2)

        # Recovery: position-by-position match against native.
        # "AAAA" vs "MAGG": positions 0-3, matches at index 1 only -> 0.25
        # "GGGG" vs "MAGG": matches at 2, 3 -> 0.5
        # "MAGG" vs "MAGG": all 4 match -> 1.0
        assert designs[0].recovery == 0.25
        assert designs[1].recovery == 0.5
        assert designs[2].recovery == 1.0

    @patch.object(ESMIF1, "_score_one")
    @patch.object(ESMIF1, "_sample_one")
    def test_skip_scoring(
        self,
        mock_sample: MagicMock,
        mock_score: MagicMock,
    ) -> None:
        """score_sequences=False skips the score_sequence call."""
        mock_sample.side_effect = ["AAAA", "GGGG"]
        mock_score.side_effect = AssertionError("should not be called")

        engine = ESMIF1(num_seqs=2, score_sequences=False)
        engine._model = MagicMock()
        engine._alphabet = MagicMock()

        designs = engine._sample_designs("fake_coords", native_seq="MAGG")

        # All scores are 0.0 in this mode.
        for design in designs:
            assert design.score == 0.0
        # score_sequence was never called.
        assert mock_score.call_count == 0

    @patch.object(ESMIF1, "_score_one")
    @patch.object(ESMIF1, "_sample_one")
    def test_skip_recovery(
        self,
        mock_sample: MagicMock,
        mock_score: MagicMock,
    ) -> None:
        """compute_recovery=False produces designs with recovery=None."""
        mock_sample.side_effect = ["AAAA"]
        mock_score.side_effect = [-1.0]

        engine = ESMIF1(num_seqs=1, compute_recovery=False)
        engine._model = MagicMock()
        engine._alphabet = MagicMock()

        designs = engine._sample_designs("fake_coords", native_seq="MAGG")
        assert designs[0].recovery is None

    @patch.object(ESMIF1, "_score_one")
    @patch.object(ESMIF1, "_sample_one")
    def test_empty_native_sequence_yields_none_recovery(
        self,
        mock_sample: MagicMock,
        mock_score: MagicMock,
    ) -> None:
        """If the loader couldn't recover a native sequence
        (unusual but defensible — e.g. a poly-GLY scaffold from
        RFdiffusion has a 'native' but you don't want to compare
        against it), recovery is None."""
        mock_sample.side_effect = ["AAAA"]
        mock_score.side_effect = [-1.0]

        engine = ESMIF1(num_seqs=1)
        engine._model = MagicMock()
        engine._alphabet = MagicMock()

        designs = engine._sample_designs("fake_coords", native_seq="")
        assert designs[0].recovery is None


# ---------------------------------------------------------------------
# Full generate() integration (mocked model + ensure_loaded)
# ---------------------------------------------------------------------


class TestGenerate:
    """Drive the full generate() flow with the heavy pieces mocked:
    _ensure_loaded does nothing, the upstream load_coords returns
    synthetic coords, and the sample/score functions are mocked.
    Verifies the Provenance attachment + sort-order behaviour."""

    def _make_engine(self) -> ESMIF1:
        engine = ESMIF1(num_seqs=3)
        engine._model = MagicMock()
        engine._alphabet = MagicMock()
        return engine

    @patch.object(ESMIF1, "_score_one")
    @patch.object(ESMIF1, "_sample_one")
    @patch.object(ESMIF1, "_load_coords")
    @patch.object(ESMIF1, "_ensure_loaded")
    def test_generate_returns_sorted_designs(
        self,
        mock_load: MagicMock,
        mock_load_coords: MagicMock,
        mock_sample: MagicMock,
        mock_score: MagicMock,
    ) -> None:
        # _ensure_loaded is a no-op (model already set in fixture).
        mock_load.return_value = None
        mock_load_coords.return_value = ("fake_coords", "MAGG")
        mock_sample.side_effect = ["AAAA", "GGGG", "MAGG"]
        # ll_fullseq values: -1.5, -0.5, -1.0  ->  scores 1.5, 0.5, 1.0
        # Best (lowest) is the GGGG design at score 0.5.
        mock_score.side_effect = [-1.5, -0.5, -1.0]

        engine = self._make_engine()
        designs = engine.generate(_tiny_backbone())

        assert len(designs) == 3
        # Sorted best-first by score (lower = better).
        scores = [d.score for d in designs]
        assert scores == sorted(scores)
        # Best design is the GGGG one.
        assert designs[0].sequence == "GGGG"
        assert designs[0].score == 0.5

    @patch.object(ESMIF1, "_score_one")
    @patch.object(ESMIF1, "_sample_one")
    @patch.object(ESMIF1, "_load_coords")
    @patch.object(ESMIF1, "_ensure_loaded")
    def test_provenance_shared_across_designs(
        self,
        mock_load: MagicMock,
        mock_load_coords: MagicMock,
        mock_sample: MagicMock,
        mock_score: MagicMock,
    ) -> None:
        """All designs from one call share the same Provenance —
        immutable, so by-reference sharing is safe (same pattern as
        ProteinMPNN and Gnina)."""
        mock_load.return_value = None
        mock_load_coords.return_value = ("fake_coords", "MAGG")
        mock_sample.side_effect = ["AAAA", "GGGG", "MAGG"]
        mock_score.side_effect = [-1.0] * 3

        engine = self._make_engine()
        designs = engine.generate(_tiny_backbone())

        provs = [d.metadata[mk.PROVENANCE] for d in designs]
        # All references point at the same Provenance instance.
        assert provs[0] is provs[1] is provs[2]
        assert provs[0].engine == "ESM-IF1"

    @patch.object(ESMIF1, "_score_one")
    @patch.object(ESMIF1, "_sample_one")
    @patch.object(ESMIF1, "_load_coords")
    @patch.object(ESMIF1, "_ensure_loaded")
    def test_provenance_chains_through_upstream(
        self,
        mock_load: MagicMock,
        mock_load_coords: MagicMock,
        mock_sample: MagicMock,
        mock_score: MagicMock,
    ) -> None:
        """If the input backbone has its own provenance (e.g. it
        came from RFdiffusion), the chain extends back through it.
        This is the headline RFdiffusion → ESM-IF1 design loop."""
        mock_load.return_value = None
        mock_load_coords.return_value = ("fake_coords", "MAGG")
        mock_sample.side_effect = ["AAAA"]
        mock_score.side_effect = [-1.0]

        backbone = _tiny_backbone()
        backbone.metadata[mk.PROVENANCE] = Provenance.from_engine(
            engine="RFdiffusion",
            parameters={"num_designs": 1},
            inputs={},
        )

        engine = ESMIF1(num_seqs=1)
        engine._model = MagicMock()
        engine._alphabet = MagicMock()

        designs = engine.generate(backbone)
        chain = designs[0].metadata[mk.PROVENANCE].chain()
        engines = [s.engine for s in chain]
        assert engines == ["RFdiffusion", "ESM-IF1"]

    @patch.object(ESMIF1, "_score_one")
    @patch.object(ESMIF1, "_sample_one")
    @patch.object(ESMIF1, "_load_coords")
    @patch.object(ESMIF1, "_ensure_loaded")
    def test_provenance_parameters_capture_engine_config(
        self,
        mock_load: MagicMock,
        mock_load_coords: MagicMock,
        mock_sample: MagicMock,
        mock_score: MagicMock,
    ) -> None:
        """Provenance.parameters captures every constructor field
        relevant to reproducing the call."""
        mock_load.return_value = None
        mock_load_coords.return_value = ("fake_coords", "MAGG")
        mock_sample.side_effect = ["AAAA"]
        mock_score.side_effect = [-1.0]

        engine = ESMIF1(
            num_seqs=1,
            temperature=0.5,
            score_sequences=True,
            compute_recovery=True,
            seed=42,
        )
        engine._model = MagicMock()
        engine._alphabet = MagicMock()

        designs = engine.generate(_tiny_backbone(), chain_id="B")
        prov = designs[0].metadata[mk.PROVENANCE]
        params = prov.parameters
        assert params["model_name"] == "esm_if1_gvp4_t16_142M_UR50"
        assert params["num_seqs"] == 1
        assert params["temperature"] == 0.5
        assert params["score_sequences"] is True
        assert params["compute_recovery"] is True
        assert params["chain_id"] == "B"
        assert params["seed"] == 42


# ---------------------------------------------------------------------
# Backbone materialisation
# ---------------------------------------------------------------------


class TestMaterialiseBackbone:
    def test_path_passes_through(self, tmp_path: Path) -> None:
        """Given a string or Path, materialise returns it unchanged
        (no temp PDB written)."""
        engine = ESMIF1()
        path = tmp_path / "input.pdb"
        path.write_text("dummy")
        out = engine._materialise_backbone(path, tmp_path)
        assert out == path

    def test_protein_writes_to_temp_pdb(self, tmp_path: Path) -> None:
        """Given a Protein, materialise writes it to a temp PDB
        in the given temp dir and returns the path."""
        # We need a Protein that actually saves successfully. Use
        # a fixture file instead of constructing one.
        from molforge.io import load

        fixture = (
            Path(__file__).resolve().parents[2] / "fixtures" / "pdb" / "ala_tripeptide_heavy.pdb"
        )
        if not fixture.exists():
            pytest.skip(f"fixture {fixture} not found")
        protein = load(fixture)

        engine = ESMIF1()
        out = engine._materialise_backbone(protein, tmp_path)
        assert out.parent == tmp_path
        assert out.is_file()
        assert out.suffix == ".pdb"


# ---------------------------------------------------------------------
# Source-inspection regression net (cheap protection)
# ---------------------------------------------------------------------


class TestSourceInspection:
    def test_uses_score_sequence_first_return(self) -> None:
        """score_sequence returns (ll_fullseq, ll_withcoord); we
        use ll_fullseq. Catches a future refactor that mixes up
        which one to use."""
        from molforge.wrappers.generative import esm_if1

        src = Path(esm_if1.__file__).read_text()
        # The destructure that picks ll_fullseq (and discards
        # ll_withcoord) must be present.
        assert "ll_fullseq, _ = esm.inverse_folding.util.score_sequence" in src

    def test_negative_log_likelihood_convention(self) -> None:
        """molforge's DesignedSequence convention: lower score =
        better. ESM-IF1 returns positive log-likelihoods, so we
        negate. Catches a refactor that flips the sign."""
        from molforge.wrappers.generative import esm_if1

        src = Path(esm_if1.__file__).read_text()
        # The "score = -float(ll_fullseq)" line must be present.
        assert "score = -float(ll_fullseq)" in src


# ---------------------------------------------------------------------
# End-to-end (real ESM-IF1)
# ---------------------------------------------------------------------


@pytest.mark.skipif(not _fair_esm_available(), reason="fair-esm not installed")
class TestRealESMIF1:
    """Run the real ESM-IF1 model. Skipped when fair-esm isn't
    available. Slow on first run (downloads ~145 MB of weights),
    fast thereafter (cached under ~/.cache/torch/hub)."""

    def test_sample_three_sequences(self, tmp_path: Path) -> None:
        from molforge.io import load

        fixture = (
            Path(__file__).resolve().parents[2] / "fixtures" / "pdb" / "ala_tripeptide_heavy.pdb"
        )
        if not fixture.exists():
            pytest.skip(f"fixture {fixture} not found")
        backbone = load(fixture)

        engine = ESMIF1(num_seqs=3, temperature=0.1, seed=42)
        designs = engine.generate(backbone)

        assert len(designs) == 3
        for d in designs:
            assert isinstance(d, DesignedSequence)
            assert d.sequence  # non-empty
            assert d.score >= 0  # negative log-likelihood, positive
            assert d.metadata["engine"] == "ESM-IF1"
            assert mk.PROVENANCE in d.metadata
        # Sorted best-first.
        scores = [d.score for d in designs]
        assert scores == sorted(scores)
