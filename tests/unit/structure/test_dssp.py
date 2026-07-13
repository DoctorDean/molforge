"""Tests for DSSP secondary-structure assignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molforge.core import AtomArray, Protein
from molforge.io import read_pdb
from molforge.structure import dssp, dssp_3state
from molforge.structure.dssp import _place_hydrogens

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "pdb"


class TestEmptyAndSmall:
    def test_empty_protein(self) -> None:
        result = dssp(Protein(AtomArray(0)))
        assert result["codes_8"] == []
        assert result["codes_3"] == []

    def test_tiny_protein(self) -> None:
        # Tripeptide is too short to form any secondary structure;
        # DSSP should return mostly "-" / "C".
        p = read_pdb(FIXTURES / "tripeptide.pdb")
        result = dssp(p)
        # 3 residues
        assert len(result["codes_8"]) == 3
        assert len(result["codes_3"]) == 3
        # Coil-only for short structures
        assert all(c == "C" for c in result["codes_3"])


class TestHelix:
    """The idealized helix fixture is constructed to form i->i+4 H-bonds,
    so DSSP should identify the middle residues as alpha-helical (H)."""

    @pytest.fixture
    def helix_result(self) -> dict[str, object]:
        p = read_pdb(FIXTURES / "helix.pdb")
        return dssp(p)

    def test_correct_residue_count(self, helix_result: dict[str, object]) -> None:
        assert len(helix_result["codes_8"]) == 15

    def test_majority_is_helix(self, helix_result: dict[str, object]) -> None:
        codes = helix_result["codes_3"]
        # The idealized helix should produce H assignments across the
        # middle. Terminal residues commonly slip to C because they
        # can't H-bond at both ends.
        middle = codes[3:12]
        h_count = sum(1 for c in middle if c == "H")
        assert h_count >= 7, f"expected >=7 H residues in middle 9; got codes={middle}"

    def test_no_strand_in_helix_fixture(self, helix_result: dict[str, object]) -> None:
        codes = helix_result["codes_3"]
        # An alpha helix should not produce beta-strand assignments
        assert "E" not in codes


class TestThreeState:
    def test_string_form(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        codes = dssp_3state(p)
        assert isinstance(codes, str)
        assert len(codes) == 15
        assert set(codes) <= {"H", "E", "C"}

    def test_tripeptide_all_coil(self) -> None:
        p = read_pdb(FIXTURES / "tripeptide.pdb")
        codes = dssp_3state(p)
        assert codes == "CCC"


class TestEightStateAlphabet:
    def test_codes_in_valid_alphabet(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        result = dssp(p)
        valid = {"H", "G", "I", "E", "B", "T", "S", "-"}
        for c in result["codes_8"]:
            assert c in valid


class TestMetadata:
    def test_residue_labels_match_residue_count(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        result = dssp(p)
        assert len(result["residue_labels"]) == len(result["codes_8"])

    def test_residue_labels_format(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        result = dssp(p)
        chain, resid, ins = result["residue_labels"][0]
        assert chain == "A"
        assert resid == 1
        assert ins == ""

    def test_hbond_energy_matrix_shape(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        result = dssp(p)
        e = result["hbond_energies"]
        assert e.shape == (15, 15)
        assert e.dtype == np.float32


class TestHydrogenPlacement:
    """The backbone amide H must sit 1.0 Å from N along the previous
    residue's C=O bond direction (Kabsch-Sander). The earlier code placed
    it along (N - C_prev) — ~57° off — which degraded H-bond detection on
    real proteins (DSSP-vs-mdtraj agreement fell to ~50-80%; the fix
    restores it to ~95-100%).
    """

    def test_h_along_previous_carbonyl(self) -> None:
        # Residue 0 provides the carbonyl; residue 1 is the amide donor.
        n = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
        c = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32)
        o = np.array([[1.0, 1.0, 0.0], [3.0, 1.0, 0.0]], dtype=np.float32)
        mask = np.array([True, True])
        h_coords, h_mask = _place_hydrogens(n, c, o, mask, chain_starts=[0])
        # First residue of the chain has no preceding C=O -> no H.
        assert not h_mask[0]
        assert h_mask[1]
        # C=O of residue 0 is (C0 - O0) = (0, -1, 0); H is N1 + that unit
        # vector = (2, -1, 0). The old (N - C_prev) rule would give (3,0,0).
        np.testing.assert_allclose(h_coords[1], [2.0, -1.0, 0.0], atol=1e-5)
        nh = h_coords[1] - n[1]
        assert float(np.linalg.norm(nh)) == pytest.approx(1.0, abs=1e-5)
        co = c[0] - o[0]
        np.testing.assert_allclose(nh, co / np.linalg.norm(co), atol=1e-5)


class TestNonProteinHandling:
    def test_water_atoms_get_dash(self) -> None:
        # Dipeptide fixture has water; water residues should be "-"
        p = read_pdb(FIXTURES / "dipeptide.pdb")
        result = dssp(p)
        # Find the water entry by label
        labels = result["residue_labels"]
        codes = result["codes_8"]
        water_indices = [i for i, (_, _, _) in enumerate(labels) if labels[i][0] == "W"]
        assert water_indices, "expected water in dipeptide fixture"
        for i in water_indices:
            assert codes[i] == "-"
