"""Tests for AlphaFold-specific helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molforge.io import is_alphafold_pdb, load_alphafold

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "pdb"


class TestDetection:
    def test_alphafold_file_detected(self) -> None:
        assert is_alphafold_pdb(FIXTURES / "alphafold_mock.pdb") is True

    def test_regular_pdb_not_detected(self) -> None:
        assert is_alphafold_pdb(FIXTURES / "dipeptide.pdb") is False


class TestLoad:
    def test_returns_protein(self) -> None:
        p = load_alphafold(FIXTURES / "alphafold_mock.pdb")
        assert p.n_atoms == 5

    def test_plddt_in_metadata(self) -> None:
        p = load_alphafold(FIXTURES / "alphafold_mock.pdb")
        assert "plddt" in p.metadata
        plddt = p.metadata["plddt"]
        assert plddt.shape == (5,)
        # Ala has pLDDT 90.50, Gly has 75.20
        np.testing.assert_allclose(plddt[:3], [90.5, 90.5, 90.5], atol=1e-2)
        np.testing.assert_allclose(plddt[3:], [75.2, 75.2], atol=1e-2)

    def test_mean_plddt(self) -> None:
        p = load_alphafold(FIXTURES / "alphafold_mock.pdb")
        # mean of [90.5*3, 75.2*2] = (271.5 + 150.4) / 5 = 84.38
        assert p.metadata["mean_plddt"] == pytest.approx(84.38, abs=0.05)

    def test_per_residue_plddt(self) -> None:
        p = load_alphafold(FIXTURES / "alphafold_mock.pdb")
        per_res = p.metadata["plddt_per_residue"]
        assert per_res.shape == (2,)
        np.testing.assert_allclose(per_res, [90.5, 75.2], atol=1e-2)

    def test_source_tagged(self) -> None:
        p = load_alphafold(FIXTURES / "alphafold_mock.pdb")
        assert p.metadata.get("source") == "alphafold"

    def test_b_factor_preserved(self) -> None:
        p = load_alphafold(FIXTURES / "alphafold_mock.pdb")
        # B-factor column should still hold the same pLDDT values
        # for compatibility with downstream tools
        assert p.atom_array.b_factor[0] == pytest.approx(90.5, abs=0.05)
