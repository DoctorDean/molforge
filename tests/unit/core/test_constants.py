"""Tests for residue/atom name constants and helper functions."""

from __future__ import annotations

from molforge.core import (
    ONE_TO_THREE,
    THREE_TO_ONE,
    is_ion,
    is_standard_amino_acid,
    is_water,
    three_to_one,
)


class TestAminoAcidTables:
    def test_twenty_canonical(self) -> None:
        assert len(THREE_TO_ONE) == 20

    def test_round_trip(self) -> None:
        for three, one in THREE_TO_ONE.items():
            assert ONE_TO_THREE[one] == three

    def test_three_to_one_canonical(self) -> None:
        assert three_to_one("ALA") == "A"
        assert three_to_one("TRP") == "W"

    def test_three_to_one_case_insensitive(self) -> None:
        assert three_to_one("ala") == "A"
        assert three_to_one(" Ala ") == "A"

    def test_three_to_one_non_canonical(self) -> None:
        assert three_to_one("MSE") == "M"
        assert three_to_one("SEP") == "S"
        assert three_to_one("HSD") == "H"

    def test_three_to_one_nucleotide(self) -> None:
        assert three_to_one("DA") == "A"
        assert three_to_one("U") == "U"

    def test_three_to_one_unknown_returns_x(self) -> None:
        assert three_to_one("ZZZ") == "X"

    def test_three_to_one_custom_unknown(self) -> None:
        assert three_to_one("ZZZ", unknown="?") == "?"


class TestClassifiers:
    def test_is_standard_amino_acid(self) -> None:
        assert is_standard_amino_acid("ALA") is True
        assert is_standard_amino_acid("MSE") is False
        assert is_standard_amino_acid("HOH") is False

    def test_is_water(self) -> None:
        assert is_water("HOH") is True
        assert is_water("WAT") is True
        assert is_water("ALA") is False

    def test_is_ion(self) -> None:
        assert is_ion("NA") is True
        assert is_ion("ZN") is True
        assert is_ion("ALA") is False
