"""Tests for DockQ complex-quality metrics."""

from __future__ import annotations

from pathlib import Path

import pytest

from molforge.io import read_pdb
from molforge.metrics import dockq, fnat, irms, lrms

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "pdb"


class TestFnat:
    def test_native_vs_self_is_one(self) -> None:
        p = read_pdb(FIXTURES / "mini_complex_native.pdb")
        assert fnat(p, p) == pytest.approx(1.0)

    def test_good_model_recovers_most_contacts(self) -> None:
        native = read_pdb(FIXTURES / "mini_complex_native.pdb")
        model = read_pdb(FIXTURES / "mini_complex_good.pdb")
        # Small noise (0.3 Å) means most native contacts survive
        f = fnat(model, native)
        assert 0.5 < f <= 1.0

    def test_bad_model_loses_all_contacts(self) -> None:
        native = read_pdb(FIXTURES / "mini_complex_native.pdb")
        bad = read_pdb(FIXTURES / "mini_complex_bad.pdb")
        # Chain B is 30 Å away -- no interface contacts survive
        assert fnat(bad, native) == 0.0

    def test_default_chains_auto_picked(self) -> None:
        native = read_pdb(FIXTURES / "mini_complex_native.pdb")
        # Should automatically pick A and B
        result = fnat(native, native)
        assert result == pytest.approx(1.0)

    def test_explicit_chains(self) -> None:
        native = read_pdb(FIXTURES / "mini_complex_native.pdb")
        assert fnat(native, native, chain_a="A", chain_b="B") == pytest.approx(1.0)


class TestIrms:
    def test_native_vs_self_is_zero(self) -> None:
        p = read_pdb(FIXTURES / "mini_complex_native.pdb")
        # iRMS on self is zero (or near-zero for floating point)
        assert irms(p, p) == pytest.approx(0.0, abs=1e-4)

    def test_good_model_has_low_irms(self) -> None:
        native = read_pdb(FIXTURES / "mini_complex_native.pdb")
        model = read_pdb(FIXTURES / "mini_complex_good.pdb")
        # Backbone perturbed by 0.3 Å noise -> iRMS should be < 1 Å
        score = irms(model, native)
        assert 0.1 < score < 1.0


class TestLrms:
    def test_native_vs_self_is_zero(self) -> None:
        p = read_pdb(FIXTURES / "mini_complex_native.pdb")
        assert lrms(p, p) == pytest.approx(0.0, abs=1e-4)

    def test_bad_model_has_huge_lrms(self) -> None:
        native = read_pdb(FIXTURES / "mini_complex_native.pdb")
        bad = read_pdb(FIXTURES / "mini_complex_bad.pdb")
        # Ligand chain shifted 30 Å away after superposing the receptor
        assert lrms(bad, native) > 10.0


class TestDockQ:
    def test_native_vs_self_is_one(self) -> None:
        p = read_pdb(FIXTURES / "mini_complex_native.pdb")
        result = dockq(p, p)
        assert result["DockQ"] == pytest.approx(1.0, abs=1e-3)
        assert result["fnat"] == pytest.approx(1.0)
        assert result["iRMS"] == pytest.approx(0.0, abs=1e-4)
        assert result["LRMS"] == pytest.approx(0.0, abs=1e-4)

    def test_good_model_high_score(self) -> None:
        native = read_pdb(FIXTURES / "mini_complex_native.pdb")
        model = read_pdb(FIXTURES / "mini_complex_good.pdb")
        result = dockq(model, native)
        # Most metrics should be in the "high quality" band
        assert result["DockQ"] > 0.7

    def test_bad_model_low_score(self) -> None:
        native = read_pdb(FIXTURES / "mini_complex_native.pdb")
        bad = read_pdb(FIXTURES / "mini_complex_bad.pdb")
        result = dockq(bad, native)
        # No native contacts + huge LRMS = very low DockQ
        assert result["DockQ"] < 0.3
        assert result["fnat"] == 0.0
        assert result["LRMS"] > 10.0

    def test_keys_present(self) -> None:
        p = read_pdb(FIXTURES / "mini_complex_native.pdb")
        result = dockq(p, p)
        assert set(result.keys()) == {"DockQ", "fnat", "iRMS", "LRMS"}
