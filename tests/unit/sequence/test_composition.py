"""Tests for sequence composition / property helpers."""

from __future__ import annotations

import pytest

from molforge.sequence import (
    aromaticity,
    composition,
    gravy,
    length,
    molecular_weight,
)


class TestComposition:
    def test_count_mode(self) -> None:
        comp = composition("AAAAGG")
        assert comp["A"] == 4
        assert comp["G"] == 2
        assert comp["W"] == 0

    def test_fraction_mode(self) -> None:
        comp = composition("AAAAGG", as_fraction=True)
        assert comp["A"] == pytest.approx(4 / 6)
        assert comp["G"] == pytest.approx(2 / 6)
        assert sum(comp.values()) == pytest.approx(1.0)

    def test_case_insensitive(self) -> None:
        assert composition("aaaa")["A"] == 4

    def test_ignores_unknown(self) -> None:
        comp = composition("AABXZ")
        assert comp["A"] == 2
        # B/Z/X are not in the 20 standard, so they're ignored
        assert sum(comp.values()) == 2

    def test_empty(self) -> None:
        comp = composition("")
        assert sum(comp.values()) == 0


class TestLength:
    def test_basic(self) -> None:
        assert length("MKTV") == 4

    def test_ignores_non_standard(self) -> None:
        assert length("MKTVX") == 4


class TestMolecularWeight:
    def test_single_residue(self) -> None:
        # Glycine residue mass (57.05) + water (18.02) ≈ 75.07
        mw = molecular_weight("G")
        assert mw == pytest.approx(57.0519 + 18.01528, abs=0.01)

    def test_zero_for_empty(self) -> None:
        assert molecular_weight("") == 0.0

    def test_dipeptide(self) -> None:
        mw_a_only = molecular_weight("A")
        mw_aa = molecular_weight("AA")
        # Adding one alanine adds residue mass (71.08), not residue+water
        assert mw_aa - mw_a_only == pytest.approx(71.0788, abs=0.01)


class TestGravy:
    def test_hydrophobic_high(self) -> None:
        # All Ile = +4.5
        assert gravy("IIII") == pytest.approx(4.5, abs=0.01)

    def test_hydrophilic_low(self) -> None:
        # All Arg = -4.5
        assert gravy("RRRR") == pytest.approx(-4.5, abs=0.01)

    def test_empty(self) -> None:
        assert gravy("") == 0.0


class TestAromaticity:
    def test_all_aromatic(self) -> None:
        assert aromaticity("FYW") == 1.0

    def test_no_aromatic(self) -> None:
        assert aromaticity("AAAA") == 0.0

    def test_mixed(self) -> None:
        # 1 of 4 is aromatic
        assert aromaticity("AAFA") == 0.25

    def test_empty(self) -> None:
        assert aromaticity("") == 0.0
