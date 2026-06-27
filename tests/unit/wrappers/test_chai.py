"""Tests for the Chai-1 folding wrapper.

Chai-1 requires a CUDA-capable GPU with bfloat16 support and the
chai_lab PyPI package. Neither is typically available in CI, so the
strategy mirrors test_boltz.py: end-to-end tests skip when chai_lab
isn't installed, and every other seam is exercised in isolation
with synthetic input.

The seams worth testing without the model:

1. Constructor validation (no torch / no chai_lab needed).
2. FASTA construction (string concatenation).
3. NPZ score loading (synthetic numpy archives).
4. CIF B-factor → per-residue pLDDT extraction.
5. Sample collection from a synthetic output directory.
6. Output parsing → Protein with the right metadata and provenance.
7. Friendly error message when chai_lab is missing.
"""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from molforge.core import metadata_keys as mk
from molforge.core.provenance import Provenance
from molforge.wrappers.folding import Chai1
from molforge.wrappers.folding._base import FoldingEngineNotInstalledError
from molforge.wrappers.folding.chai import (
    _CHAI_NUM_SAMPLES,
    _load_scores_npz,
    _per_residue_plddt_from_cif,
)


def _chai_lab_available() -> bool:
    return importlib.util.find_spec("chai_lab") is not None


# A minimal CIF with three residues; B-factor column carries the
# values we want to read back as per-residue pLDDT.
_MINIMAL_CIF = """\
data_test
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.B_iso_or_equiv
_atom_site.type_symbol
ATOM 1 N  MET A 1 0.0 0.0 0.0 75.0 N
ATOM 2 CA MET A 1 1.5 0.0 0.0 80.0 C
ATOM 3 C  MET A 1 3.0 0.0 0.0 85.0 C
ATOM 4 N  LYS A 2 4.5 0.0 0.0 60.0 N
ATOM 5 CA LYS A 2 6.0 0.0 0.0 65.0 C
ATOM 6 C  LYS A 2 7.5 0.0 0.0 70.0 C
ATOM 7 N  GLN A 3 9.0 0.0 0.0 90.0 N
ATOM 8 CA GLN A 3 10.5 0.0 0.0 95.0 C
ATOM 9 C  GLN A 3 12.0 0.0 0.0 88.0 C
"""


def _write_synthetic_outputs(
    tmpdir: Path,
    *,
    n_samples: int = _CHAI_NUM_SAMPLES,
    aggregate_scores: list[float] | None = None,
    cif_text: str = _MINIMAL_CIF,
) -> None:
    """Write a complete set of Chai-1-shaped outputs into a directory.

    Used by tests that need to drive ``_collect_samples`` and
    ``_parse_outputs`` against a realistic directory layout without
    running the model.
    """
    if aggregate_scores is None:
        # Default: monotonically increasing so sample N-1 wins.
        aggregate_scores = [0.5 + i * 0.1 for i in range(n_samples)]
    assert len(aggregate_scores) == n_samples

    for i in range(n_samples):
        (tmpdir / f"pred.model_idx_{i}.cif").write_text(cif_text)
        np.savez(
            tmpdir / f"scores.model_idx_{i}.npz",
            aggregate_score=np.array(aggregate_scores[i]),
            ptm=np.array(0.80 + i * 0.01),
            iptm=np.array(0.0),
            has_inter_chain_clashes=np.array(False),
        )


# ---------------------------------------------------------------------
# Constructor + validation
# ---------------------------------------------------------------------


class TestConstruction:
    def test_defaults(self) -> None:
        engine = Chai1()
        assert engine.device is None
        assert engine.use_msa_server is False
        assert engine.msa_server_url is None
        assert engine.num_trunk_recycles is None
        assert engine.num_diffn_timesteps is None
        assert engine.seed is None
        assert engine.cache_dir is None

    def test_custom_options(self) -> None:
        engine = Chai1(
            device="cuda:1",
            use_msa_server=True,
            msa_server_url="https://example.com/msa",
            num_trunk_recycles=5,
            num_diffn_timesteps=300,
            seed=42,
            cache_dir="/tmp/chai-weights",
        )
        assert engine.device == "cuda:1"
        assert engine.use_msa_server is True
        assert engine.msa_server_url == "https://example.com/msa"
        assert engine.num_trunk_recycles == 5
        assert engine.num_diffn_timesteps == 300
        assert engine.seed == 42
        assert engine.cache_dir == "/tmp/chai-weights"

    def test_invalid_num_trunk_recycles(self) -> None:
        with pytest.raises(ValueError, match="num_trunk_recycles"):
            Chai1(num_trunk_recycles=0)
        with pytest.raises(ValueError, match="num_trunk_recycles"):
            Chai1(num_trunk_recycles=-3)

    def test_invalid_num_diffn_timesteps(self) -> None:
        with pytest.raises(ValueError, match="num_diffn_timesteps"):
            Chai1(num_diffn_timesteps=0)
        with pytest.raises(ValueError, match="num_diffn_timesteps"):
            Chai1(num_diffn_timesteps=-5)

    def test_construction_is_lazy(self) -> None:
        """Construction must not touch chai_lab, torch, or any heavy
        dependency. Users can import and construct Chai1 even when
        chai_lab isn't installed."""
        # If construction touched chai_lab we'd see an ImportError;
        # the fact that this returns at all is the test.
        Chai1(device="cuda")
        Chai1(use_msa_server=True)


# ---------------------------------------------------------------------
# FASTA construction
# ---------------------------------------------------------------------


class TestFastaConstruction:
    def test_single_protein_fasta(self) -> None:
        engine = Chai1()
        fasta = engine._build_fasta("MKQH", name="myprotein")
        assert fasta == ">protein|name=myprotein\nMKQH\n"

    def test_header_uses_chai_typed_format(self) -> None:
        """Chai-1 differentiates protein/ligand/dna/rna by the FASTA
        header type prefix. v1 wrapper hard-codes 'protein|name=...'
        — this test catches a future regression that drops the
        type prefix."""
        engine = Chai1()
        fasta = engine._build_fasta("M", name="x")
        assert fasta.startswith(">protein|")
        assert "name=x" in fasta


# ---------------------------------------------------------------------
# Missing chai_lab error
# ---------------------------------------------------------------------


class TestMissingDependency:
    @pytest.mark.skipif(_chai_lab_available(), reason="chai_lab is installed in this env")
    def test_friendly_error_when_chai_lab_missing(self) -> None:
        """When chai_lab isn't installed, _run_inference raises a
        FoldingEngineNotInstalledError with install guidance."""
        engine = Chai1()
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            fasta = tmpdir / "in.fasta"
            fasta.write_text(">protein|name=x\nMK\n")
            with pytest.raises(FoldingEngineNotInstalledError) as exc:
                engine._run_inference(fasta, tmpdir)
        msg = str(exc.value)
        assert "chai_lab" in msg
        # Mentions GPU requirement, since users without CUDA need
        # to know upfront.
        assert "GPU" in msg or "CUDA" in msg


# ---------------------------------------------------------------------
# NPZ score loading
# ---------------------------------------------------------------------


class TestLoadScoresNpz:
    def test_unwraps_scalar_arrays(self, tmp_path: Path) -> None:
        """Chai-1 stores scalars as 0-d arrays inside the NPZ; the
        loader should unwrap them to native Python scalars so
        downstream code doesn't have to deal with numpy types."""
        npz = tmp_path / "scores.npz"
        np.savez(
            npz,
            aggregate_score=np.array(0.85),
            ptm=np.array(0.92),
            iptm=np.array(0.5),
        )
        loaded = _load_scores_npz(npz)
        assert loaded["aggregate_score"] == pytest.approx(0.85)
        assert loaded["ptm"] == pytest.approx(0.92)
        assert loaded["iptm"] == pytest.approx(0.5)
        # Native Python floats, not numpy scalars.
        assert isinstance(loaded["aggregate_score"], float)

    def test_preserves_multi_dim_arrays(self, tmp_path: Path) -> None:
        """Multi-dim entries like per_chain_pair_iptm stay as numpy
        arrays for users who want them."""
        npz = tmp_path / "scores.npz"
        np.savez(
            npz,
            aggregate_score=np.array(0.7),
            per_chain_ptm=np.array([[0.8, 0.6]]),
        )
        loaded = _load_scores_npz(npz)
        assert isinstance(loaded["per_chain_ptm"], np.ndarray)
        assert loaded["per_chain_ptm"].shape == (1, 2)

    def test_preserves_bool_scalars(self, tmp_path: Path) -> None:
        """``has_inter_chain_clashes`` is a 0-d bool; should unwrap
        to a Python bool (not numpy.bool_)."""
        npz = tmp_path / "scores.npz"
        np.savez(npz, has_inter_chain_clashes=np.array(False))
        loaded = _load_scores_npz(npz)
        # ``False`` and ``np.False_`` compare equal but aren't the
        # same type. Item() converts to native bool.
        assert loaded["has_inter_chain_clashes"] is False


# ---------------------------------------------------------------------
# CIF B-factor → per-residue pLDDT
# ---------------------------------------------------------------------


class TestPerResiduePlddtFromCif:
    def test_extracts_ca_bfactors_per_residue(self, tmp_path: Path) -> None:
        """CA atoms carry the per-residue pLDDT; the helper returns
        one value per residue in residue order."""
        cif = tmp_path / "test.cif"
        cif.write_text(_MINIMAL_CIF)
        from molforge.io.mmcif import read_cif

        protein = read_cif(cif)
        plddt = _per_residue_plddt_from_cif(protein)
        # CA b-factors from the fixture: 80, 65, 95.
        assert plddt.shape == (3,)
        np.testing.assert_allclose(plddt, [80.0, 65.0, 95.0])
        assert plddt.dtype == np.float32

    def test_falls_back_to_residue_mean_when_no_ca(self, tmp_path: Path) -> None:
        """An all-ligand structure or non-standard chemistry might
        have no CA atoms. The helper falls back to per-residue mean
        rather than crashing."""
        no_ca_cif = """\
data_test
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.B_iso_or_equiv
_atom_site.type_symbol
HETATM 1 C1 LIG A 1 0.0 0.0 0.0 50.0 C
HETATM 2 C2 LIG A 1 1.5 0.0 0.0 60.0 C
HETATM 3 C3 LIG A 2 3.0 0.0 0.0 70.0 C
HETATM 4 C4 LIG A 2 4.5 0.0 0.0 80.0 C
"""
        cif = tmp_path / "ligand.cif"
        cif.write_text(no_ca_cif)
        from molforge.io.mmcif import read_cif

        protein = read_cif(cif)
        plddt = _per_residue_plddt_from_cif(protein)
        # Residue 1: mean(50, 60) = 55; residue 2: mean(70, 80) = 75.
        assert plddt.shape == (2,)
        np.testing.assert_allclose(plddt, [55.0, 75.0])

    def test_empty_protein_returns_empty_array(self) -> None:
        """A protein with no atoms returns an empty array (no
        IndexError or divide-by-zero)."""
        from molforge.core import AtomArray, Protein

        empty = Protein(AtomArray(0))
        plddt = _per_residue_plddt_from_cif(empty)
        assert plddt.shape == (0,)


# ---------------------------------------------------------------------
# Sample collection
# ---------------------------------------------------------------------


class TestCollectSamples:
    def test_collects_all_five(self, tmp_path: Path) -> None:
        _write_synthetic_outputs(tmp_path)
        engine = Chai1()
        samples = engine._collect_samples(tmp_path)
        assert len(samples) == _CHAI_NUM_SAMPLES
        for i, sample in enumerate(samples):
            assert sample["index"] == i
            assert sample["cif_text"]
            assert "_atom_site" in sample["cif_text"]
            assert "aggregate_score" in sample["scores"]

    def test_missing_cif_raises(self, tmp_path: Path) -> None:
        _write_synthetic_outputs(tmp_path)
        # Remove sample 2's CIF — Chai-1 producing fewer than 5
        # samples is a sign of trouble we should surface clearly.
        (tmp_path / "pred.model_idx_2.cif").unlink()
        with pytest.raises(RuntimeError, match=r"pred\.model_idx_2\.cif"):
            Chai1()._collect_samples(tmp_path)

    def test_missing_npz_raises(self, tmp_path: Path) -> None:
        _write_synthetic_outputs(tmp_path)
        (tmp_path / "scores.model_idx_3.npz").unlink()
        with pytest.raises(RuntimeError, match=r"scores\.model_idx_3\.npz"):
            Chai1()._collect_samples(tmp_path)


# ---------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------


class TestParseOutputs:
    def test_picks_best_by_aggregate_score(self, tmp_path: Path) -> None:
        # Sample 3 wins (highest aggregate_score).
        _write_synthetic_outputs(tmp_path, aggregate_scores=[0.5, 0.7, 0.6, 0.9, 0.55])
        engine = Chai1()
        samples = engine._collect_samples(tmp_path)
        protein = engine._parse_outputs(samples=samples, sequence="MKQ")
        assert protein.metadata["best_sample_index"] == 3
        assert protein.metadata["aggregate_score"] == pytest.approx(0.9)

    def test_metadata_keys_populated(self, tmp_path: Path) -> None:
        """The Protein returned must carry every metadata key the
        wrapper's docstring promises."""
        _write_synthetic_outputs(tmp_path)
        engine = Chai1()
        samples = engine._collect_samples(tmp_path)
        protein = engine._parse_outputs(samples=samples, sequence="MK")

        # The contract from the docstring.
        assert protein.metadata["engine"] == "Chai-1"
        assert protein.metadata["source_sequence"] == "MK"
        assert mk.CONFIDENCE_PER_RESIDUE in protein.metadata
        assert mk.MEAN_CONFIDENCE in protein.metadata
        assert "aggregate_score" in protein.metadata
        assert "ptm" in protein.metadata
        assert "iptm" in protein.metadata
        assert "best_sample_index" in protein.metadata
        assert "per_sample_scores" in protein.metadata
        assert mk.PROVENANCE in protein.metadata

    def test_per_residue_confidence_from_ca_bfactors(self, tmp_path: Path) -> None:
        """The pLDDT array comes from the chosen CIF's CA B-factors.
        The fixture has 3 residues with CA b-factors 80, 65, 95."""
        _write_synthetic_outputs(tmp_path)
        engine = Chai1()
        samples = engine._collect_samples(tmp_path)
        protein = engine._parse_outputs(samples=samples, sequence="MKQ")

        plddt = protein.metadata[mk.CONFIDENCE_PER_RESIDUE]
        np.testing.assert_allclose(plddt, [80.0, 65.0, 95.0])
        assert protein.metadata[mk.MEAN_CONFIDENCE] == pytest.approx((80.0 + 65.0 + 95.0) / 3)

    def test_per_sample_scores_preserved_for_all_five(self, tmp_path: Path) -> None:
        """Even after picking the best, the other four samples'
        headline scores remain in metadata so users can inspect
        ranking spread."""
        _write_synthetic_outputs(tmp_path, aggregate_scores=[0.5, 0.7, 0.6, 0.9, 0.55])
        engine = Chai1()
        samples = engine._collect_samples(tmp_path)
        protein = engine._parse_outputs(samples=samples, sequence="MKQ")

        per_sample = protein.metadata["per_sample_scores"]
        assert len(per_sample) == _CHAI_NUM_SAMPLES
        scores = [s["aggregate_score"] for s in per_sample]
        assert scores == pytest.approx([0.5, 0.7, 0.6, 0.9, 0.55])

    def test_provenance_captures_engine_config(self, tmp_path: Path) -> None:
        """Provenance.parameters captures every constructor field
        relevant to reproducing the call."""
        _write_synthetic_outputs(tmp_path)
        engine = Chai1(
            device="cuda:0",
            use_msa_server=True,
            num_trunk_recycles=5,
            num_diffn_timesteps=300,
            seed=42,
        )
        samples = engine._collect_samples(tmp_path)
        protein = engine._parse_outputs(samples=samples, sequence="MK")

        prov = protein.metadata[mk.PROVENANCE]
        assert isinstance(prov, Provenance)
        assert prov.engine == "Chai-1"
        params = prov.parameters
        assert params["device"] == "cuda:0"
        assert params["use_msa_server"] is True
        assert params["num_trunk_recycles"] == 5
        assert params["num_diffn_timesteps"] == 300
        assert params["seed"] == 42
        assert prov.inputs == {"sequence": "MK"}

    def test_empty_samples_list_raises(self) -> None:
        with pytest.raises(RuntimeError, match="no samples"):
            Chai1()._parse_outputs(samples=[], sequence="MK")

    def test_missing_aggregate_score_falls_back(self, tmp_path: Path) -> None:
        """If some samples have no aggregate_score (unusual but
        defensive), they should sort behind samples that do."""
        # Sample 1 has the score, others don't — sample 1 should win.
        for i in range(_CHAI_NUM_SAMPLES):
            (tmp_path / f"pred.model_idx_{i}.cif").write_text(_MINIMAL_CIF)
            if i == 1:
                np.savez(
                    tmp_path / f"scores.model_idx_{i}.npz",
                    aggregate_score=np.array(0.5),
                    ptm=np.array(0.8),
                    iptm=np.array(0.0),
                )
            else:
                # No aggregate_score key.
                np.savez(
                    tmp_path / f"scores.model_idx_{i}.npz",
                    ptm=np.array(0.8),
                    iptm=np.array(0.0),
                )

        engine = Chai1()
        samples = engine._collect_samples(tmp_path)
        protein = engine._parse_outputs(samples=samples, sequence="MK")
        assert protein.metadata["best_sample_index"] == 1


# ---------------------------------------------------------------------
# Full predict() pipeline (mocked _run_inference)
# ---------------------------------------------------------------------


class TestPredictPipeline:
    """Drive the full predict() flow with _run_inference mocked. The
    mock writes a complete synthetic output directory to the path it
    was given, then control returns to the wrapper which collects +
    parses just like the real thing."""

    @patch.object(Chai1, "_run_inference")
    def test_predict_end_to_end(self, mock_run: MagicMock) -> None:
        def _populate(fasta_path: Path, output_dir: Path) -> None:
            _write_synthetic_outputs(output_dir)

        mock_run.side_effect = _populate

        engine = Chai1()
        protein = engine.predict("MKQ")

        assert mock_run.call_count == 1
        # Verify _run_inference was called with the right shape of args.
        call_args = mock_run.call_args
        fasta_arg, outdir_arg = call_args[0]
        assert isinstance(fasta_arg, Path)
        assert isinstance(outdir_arg, Path)
        # Output Protein has the expected metadata.
        assert protein.metadata["engine"] == "Chai-1"
        assert protein.metadata["source_sequence"] == "MKQ"
        assert protein.metadata["best_sample_index"] == 4  # default scores monotonic

    @patch.object(Chai1, "_run_inference")
    def test_predict_validates_sequence(self, mock_run: MagicMock) -> None:
        """An invalid sequence raises before _run_inference is even
        called — we don't want to waste a GPU forward pass on bad
        input."""

        def _populate(fasta_path: Path, output_dir: Path) -> None:
            _write_synthetic_outputs(output_dir)

        mock_run.side_effect = _populate

        engine = Chai1()
        # Non-letter characters in sequence.
        with pytest.raises(ValueError):
            engine.predict("M1KQ")
        # _run_inference never called.
        assert mock_run.call_count == 0

    @patch.object(Chai1, "_run_inference")
    def test_predict_writes_typed_fasta(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """The FASTA written by predict() before _run_inference uses
        the Chai-style ``>protein|name=...`` header."""
        captured_fasta: dict[str, Any] = {}

        def _capture(fasta_path: Path, output_dir: Path) -> None:
            captured_fasta["text"] = fasta_path.read_text()
            _write_synthetic_outputs(output_dir)

        mock_run.side_effect = _capture
        Chai1().predict("MKQ")

        assert captured_fasta["text"].startswith(">protein|name=")
        assert "MKQ\n" in captured_fasta["text"]


# ---------------------------------------------------------------------
# Source-inspection regression net
# ---------------------------------------------------------------------


class TestSourceInspection:
    """Cheap invariants worth a few lines of test to lock in."""

    def test_num_samples_constant_is_five(self) -> None:
        """Chai-1's diffusion sample count is hard-coded upstream.
        If we ever pick this up from the chai_lab package we should
        revisit this assumption; until then, lock it in."""
        assert _CHAI_NUM_SAMPLES == 5

    def test_engine_string_consistent_in_source(self) -> None:
        """The 'Chai-1' engine string appears in metadata and in
        Provenance; one regression-net check that both stay in sync
        with the class-level ``name`` attribute."""
        import molforge.wrappers.folding.chai as chai_mod

        text = Path(chai_mod.__file__).read_text()
        # The string literal "Chai-1" must appear at least three times:
        # the class-level name, the engine metadata key, and the
        # Provenance.engine kwarg.
        assert text.count('"Chai-1"') >= 3


# ---------------------------------------------------------------------
# End-to-end (real chai_lab)
# ---------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not _chai_lab_available(), reason="chai_lab not installed")
class TestRealChai1:
    """Run the real Chai-1 model. Skipped when chai_lab isn't
    available. Requires a CUDA GPU. Downloads ~3 GB of weights on
    first call."""

    def test_predict_a_small_protein(self) -> None:
        engine = Chai1(use_msa_server=False, seed=42)
        # 30-residue test sequence — small enough to run on a
        # single GPU without timeouts.
        protein = engine.predict("MKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF")
        assert protein.metadata["engine"] == "Chai-1"
        assert mk.CONFIDENCE_PER_RESIDUE in protein.metadata
        assert protein.atom_array.n_atoms > 0
        # All 5 samples surface in metadata.
        assert len(protein.metadata["per_sample_scores"]) == 5
