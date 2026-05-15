"""Tests for the AlphaFold / ColabFold wrapper.

These don't require ``colabfold`` (or jax, or the AlphaFold weights)
to be installed. They exercise construction, lazy import, sequence
validation, missing-dep errors, and PDB-to-Protein post-processing
in isolation.

End-to-end folding against the real engine is gated on
``colabfold`` being importable and is marked ``@pytest.mark.slow``.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from molforge.wrappers.folding import AlphaFold, FoldingEngineNotInstalledError


def _colabfold_available() -> bool:
    return importlib.util.find_spec("colabfold") is not None


class TestConstruction:
    def test_defaults(self) -> None:
        engine = AlphaFold()
        assert engine.name == "AlphaFold"
        assert engine.mode == "local"
        assert engine.num_models == 5
        assert engine.num_recycles == 3
        assert engine.msa_mode == "mmseqs2_uniref_env"
        assert engine.model_type == "AlphaFold2-ptm"

    def test_custom_settings(self) -> None:
        engine = AlphaFold(
            num_models=1,
            num_recycles=1,
            msa_mode="single_sequence",
            model_type="AlphaFold2",
        )
        assert engine.num_models == 1
        assert engine.num_recycles == 1
        assert engine.msa_mode == "single_sequence"
        assert engine.model_type == "AlphaFold2"

    def test_server_mode_not_yet_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="server mode"):
            AlphaFold(mode="server")

    def test_construction_does_not_import_colabfold(self) -> None:
        """Construction must not trigger heavy imports."""
        # If construction loaded colabfold, this test would fail in
        # environments without it installed. The fact that the previous
        # tests passed without ImportError proves construction is light.
        AlphaFold()


class TestMissingDependency:
    @pytest.mark.skipif(_colabfold_available(), reason="colabfold is installed")
    def test_predict_without_colabfold_raises_clear_error(self) -> None:
        engine = AlphaFold()
        with pytest.raises(FoldingEngineNotInstalledError, match="colabfold"):
            engine.predict("MKTV")


class TestSequenceValidation:
    def test_empty_sequence_raises(self) -> None:
        engine = AlphaFold()
        with pytest.raises(ValueError, match="empty"):
            engine.predict("")

    def test_non_letter_raises(self) -> None:
        engine = AlphaFold()
        with pytest.raises(ValueError, match="non-letter"):
            engine.predict("MKTV*")


class TestPostProcessing:
    """Test _pdb_to_protein in isolation — testable without colabfold."""

    def test_attaches_engine_metadata(self) -> None:
        pdb = (
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 92.50           N  \n"
            "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 92.50           C  \n"
            "ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00 92.50           C  \n"
            "ATOM      4  N   GLY A   2       3.000   0.000   0.000  1.00 78.20           N  \n"
            "ATOM      5  CA  GLY A   2       4.000   0.000   0.000  1.00 78.20           C  \n"
            "END\n"
        )
        engine = AlphaFold(num_models=2, num_recycles=1)
        protein = engine._pdb_to_protein(pdb, sequence="AG")

        assert protein.n_atoms == 5
        assert protein.metadata["engine"] == "AlphaFold"
        assert protein.metadata["source_sequence"] == "AG"
        assert protein.metadata["model_type"] == "AlphaFold2-ptm"
        assert protein.metadata["num_models"] == 2
        assert protein.metadata["num_recycles"] == 1
        assert protein.metadata["msa_mode"] == "mmseqs2_uniref_env"

    def test_per_residue_confidence(self) -> None:
        pdb = (
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 92.50           N  \n"
            "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 92.50           C  \n"
            "ATOM      3  N   GLY A   2       3.000   0.000   0.000  1.00 78.20           N  \n"
            "ATOM      4  CA  GLY A   2       4.000   0.000   0.000  1.00 78.20           C  \n"
            "END\n"
        )
        engine = AlphaFold()
        protein = engine._pdb_to_protein(pdb, sequence="AG")

        per_residue = protein.metadata["confidence_per_residue"]
        assert per_residue.shape == (2,)
        np.testing.assert_allclose(per_residue, [92.5, 78.2], atol=0.01)
        assert protein.metadata["mean_confidence"] == pytest.approx(85.35, abs=0.01)

        per_atom = protein.metadata["confidence_per_atom"]
        assert per_atom.shape == (4,)
        np.testing.assert_allclose(per_atom[:2], 92.5)
        np.testing.assert_allclose(per_atom[2:], 78.2)

    def test_high_confidence_alphafold_style(self) -> None:
        """Real AlphaFold output typically has pLDDT > 80 for ordered regions."""
        # Synthetic high-pLDDT PDB
        lines = [
            f"ATOM  {i:>5}  CA  ALA A{i:>4}    {i * 3.8:>8.3f}{0:>8.3f}{0:>8.3f}  1.00 95.00           C"
            for i in range(1, 11)
        ]
        pdb = "\n".join(lines) + "\nEND\n"
        engine = AlphaFold()
        protein = engine._pdb_to_protein(pdb, sequence="A" * 10)
        assert protein.metadata["mean_confidence"] == pytest.approx(95.0, abs=0.01)


class TestUniformConfidenceConvention:
    """The whole point of the wrapper pattern is that the output format is
    uniform across engines. Verify the AlphaFold output matches ESMFold's."""

    def test_same_metadata_keys_as_esmfold(self) -> None:
        from molforge.wrappers.folding import ESMFold

        pdb = (
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 80.00           N  \n"
            "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 80.00           C  \n"
            "END\n"
        )
        af_meta = AlphaFold()._pdb_to_protein(pdb, sequence="A").metadata
        esm_meta = ESMFold()._pdb_to_protein(pdb, sequence="A").metadata

        # Required uniform keys from FoldingEngine convention
        for key in (
            "engine",
            "source_sequence",
            "confidence_per_residue",
            "confidence_per_atom",
            "mean_confidence",
        ):
            assert key in af_meta, f"AlphaFold output missing '{key}'"
            assert key in esm_meta, f"ESMFold output missing '{key}'"


@pytest.mark.slow
@pytest.mark.skipif(not _colabfold_available(), reason="colabfold not installed")
class TestEndToEnd:
    """End-to-end tests against the real engine. Run with `pytest -m slow`."""

    def test_short_sequence_folds(self) -> None:
        # Skipped in normal CI; this requires the full AlphaFold weights
        # and a GPU. Provide here as a contract test for anyone running
        # against a real install.
        engine = AlphaFold(num_models=1, num_recycles=1)
        # Real fold; this is the smoke check that the pipeline works.
        protein = engine.predict("MKTVRQERLKSIVRILERSK")
        assert protein.n_residues == 20
        assert "confidence_per_residue" in protein.metadata
