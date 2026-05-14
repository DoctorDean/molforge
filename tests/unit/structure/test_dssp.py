"""Tests for DSSP secondary-structure assignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molforge.core import AtomArray, Protein
from molforge.io import read_pdb
from molforge.structure import dssp, dssp_3state

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
