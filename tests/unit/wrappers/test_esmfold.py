"""Tests for the ESMFold wrapper.

These tests do not require ``torch`` or ``transformers`` to be installed.
They exercise the wiring (lazy import behaviour, error messages,
post-processing) by mocking the heavy parts. End-to-end tests with the
real model are marked ``@pytest.mark.slow`` and skipped by default.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from molforge.wrappers.folding import ESMFold, FoldingEngineNotInstalledError


def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


class TestConstruction:
    def test_default_construction(self) -> None:
        engine = ESMFold()
        assert engine.name == "ESMFold"
        assert engine.model_name == "facebook/esmfold_v1"
        assert engine.device is None
        assert engine.chunk_size is None
        assert engine.dtype == "float32"

    def test_custom_settings(self) -> None:
        engine = ESMFold(device="cpu", chunk_size=64, dtype="float16")
        assert engine.device == "cpu"
        assert engine.chunk_size == 64
        assert engine.dtype == "float16"

    def test_construction_does_not_import_torch(self) -> None:
        """Constructing must not trigger heavy imports — that's the whole point of lazy loading."""
        # Just verify model/tokenizer stay None until predict is called.
        engine = ESMFold()
        assert engine._model is None
        assert engine._tokenizer is None


class TestMissingDependencies:
    @pytest.mark.skipif(_torch_available(), reason="torch is installed")
    def test_predict_without_torch_raises_clear_error(self) -> None:
        engine = ESMFold()
        with pytest.raises(FoldingEngineNotInstalledError, match="molforge\\[ml\\]"):
            engine.predict("MKTV")


class TestSequenceValidation:
    """These checks run before any model loading, so they're test-safe regardless of torch."""

    def test_empty_sequence_raises(self) -> None:
        engine = ESMFold()
        # The validator runs before _ensure_loaded, so we get ValueError
        # not FoldingEngineNotInstalledError even without torch.
        with pytest.raises(ValueError, match="empty"):
            engine.predict("")

    def test_non_letter_sequence_raises(self) -> None:
        engine = ESMFold()
        with pytest.raises(ValueError, match="non-letter"):
            engine.predict("MKTV*")


class TestPostProcessing:
    """Test _pdb_to_protein in isolation — this is the path that doesn't need torch."""

    def test_pdb_to_protein_attaches_metadata(self) -> None:
        # Build a minimal valid PDB string with non-trivial pLDDT
        pdb = (
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 80.00           N  \n"
            "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 80.00           C  \n"
            "ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00 80.00           C  \n"
            "ATOM      4  N   GLY A   2       3.000   0.000   0.000  1.00 60.00           N  \n"
            "ATOM      5  CA  GLY A   2       4.000   0.000   0.000  1.00 60.00           C  \n"
            "END\n"
        )
        engine = ESMFold()
        protein = engine._pdb_to_protein(pdb, sequence="AG")

        assert protein.n_atoms == 5
        assert protein.metadata["engine"] == "ESMFold"
        assert protein.metadata["source_sequence"] == "AG"
        assert protein.metadata["model_name"] == "facebook/esmfold_v1"

        per_residue = protein.metadata["confidence_per_residue"]
        assert per_residue.shape == (2,)
        np.testing.assert_allclose(per_residue, [80.0, 60.0], atol=1e-2)
        assert protein.metadata["mean_confidence"] == pytest.approx(70.0, abs=0.01)

        # Per-atom confidence is preserved separately too.
        per_atom = protein.metadata["confidence_per_atom"]
        assert per_atom.shape == (5,)
        np.testing.assert_allclose(per_atom[:3], 80.0)
        np.testing.assert_allclose(per_atom[3:], 60.0)


@pytest.mark.slow
@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
class TestEndToEnd:
    """End-to-end tests against the real model. Run with `pytest -m slow`."""

    def test_short_sequence_folds(self) -> None:
        # 20-residue sequence; CPU inference is feasible.
        engine = ESMFold(device="cpu")
        protein = engine.predict("MKTVRQERLKSIVRILERSK")
        assert protein.n_residues == 20
        assert protein.sequence == "MKTVRQERLKSIVRILERSK"
        assert "confidence_per_residue" in protein.metadata
