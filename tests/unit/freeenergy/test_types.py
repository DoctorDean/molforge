"""Tests for the free-energy value types, ranking, and engine base.

All pure — no external tool, no trajectory. Pins the enthalpy sum, the
entropy None-vs-zero distinction, the ΔΔG sign and error propagation,
the tightest-first ordering, and the abstract-engine contract.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from molforge.freeenergy import (
    Decomposition,
    DeltaDeltaG,
    FreeEnergyComponents,
    FreeEnergyRanking,
    FreeEnergyResult,
    MMGBSAEngine,
    MMGBSAEngineNotInstalledError,
    ResidueContribution,
)


def _result(delta_g: float, uncertainty: float = 0.5) -> FreeEnergyResult:
    return FreeEnergyResult(delta_g=delta_g, uncertainty=uncertainty, method="MM/GBSA")


class TestComponents:
    def test_enthalpy_is_sum_of_four_terms(self) -> None:
        c = FreeEnergyComponents(
            vdw=-40.0, electrostatic=-30.0, polar_solvation=55.0, nonpolar_solvation=-5.0
        )
        assert c.enthalpy == pytest.approx(-20.0)

    def test_entropy_defaults_to_none_not_zero(self) -> None:
        c = FreeEnergyComponents(-1.0, -1.0, -1.0, -1.0)
        assert c.entropy is None  # unknown, not zero

    def test_entropy_can_be_set(self) -> None:
        c = FreeEnergyComponents(-1.0, -1.0, -1.0, -1.0, entropy=12.3)
        assert c.entropy == 12.3
        # enthalpy ignores entropy
        assert c.enthalpy == pytest.approx(-4.0)

    def test_frozen(self) -> None:
        c = FreeEnergyComponents(-1.0, -1.0, -1.0, -1.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.vdw = 0.0  # type: ignore[misc]


class TestResult:
    def test_defaults(self) -> None:
        r = _result(-9.0)
        assert r.components is None
        assert r.convergence is None
        assert r.provenance is None
        assert r.metadata == {}
        assert r.method == "MM/GBSA"
        assert r.decomposition is None

    def test_decomposition_attaches(self) -> None:
        d = Decomposition([_contrib("LEU 40", -6.5)])
        r = FreeEnergyResult(delta_g=-9.0, uncertainty=0.4, method="MM/GBSA", decomposition=d)
        assert r.decomposition is d
        assert r.decomposition["LEU 40"].total == pytest.approx(-6.5)

    def test_zero_uncertainty_allowed(self) -> None:
        assert _result(-9.0, 0.0).uncertainty == 0.0

    def test_negative_uncertainty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            _result(-9.0, -0.1)

    def test_metadata_not_shared_between_instances(self) -> None:
        a = _result(-9.0)
        b = _result(-8.0)
        a.metadata["frames"] = 100
        assert b.metadata == {}

    def test_convergence_array(self) -> None:
        trace = np.array([-5.0, -7.0, -8.5, -9.0], dtype=np.float64)
        r = FreeEnergyResult(delta_g=-9.0, uncertainty=0.4, method="MM/GBSA", convergence=trace)
        assert r.convergence is not None
        assert r.convergence[-1] == pytest.approx(-9.0)


class TestDeltaDeltaG:
    def test_frozen_carrier(self) -> None:
        d = DeltaDeltaG("a", "b", value=-1.0, uncertainty=0.7, tighter="b")
        with pytest.raises(dataclasses.FrozenInstanceError):
            d.value = 0.0  # type: ignore[misc]


class TestRanking:
    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            FreeEnergyRanking({})

    def test_ranked_tightest_first(self) -> None:
        rk = FreeEnergyRanking({"A": _result(-9.5), "B": _result(-8.2), "C": _result(-11.0)})
        assert [label for label, _ in rk.ranked] == ["C", "A", "B"]

    def test_best_is_lowest_delta_g(self) -> None:
        rk = FreeEnergyRanking({"A": _result(-9.5), "C": _result(-11.0)})
        assert rk.best[0] == "C"

    def test_results_property_is_a_copy(self) -> None:
        rk = FreeEnergyRanking({"A": _result(-9.5)})
        got = rk.results
        got["A"] = _result(0.0)
        assert rk.results["A"].delta_g == pytest.approx(-9.5)

    def test_delta_delta_g_sign_and_tighter(self) -> None:
        rk = FreeEnergyRanking({"ref": _result(-9.0), "other": _result(-11.0)})
        d = rk.delta_delta_g("ref", "other")
        # other is 2 kcal/mol tighter → ΔΔG negative
        assert d.value == pytest.approx(-2.0)
        assert d.tighter == "other"

    def test_delta_delta_g_uncertainty_propagates(self) -> None:
        rk = FreeEnergyRanking({"ref": _result(-9.0, 0.3), "other": _result(-11.0, 0.4)})
        d = rk.delta_delta_g("ref", "other")
        assert d.uncertainty == pytest.approx((0.3**2 + 0.4**2) ** 0.5)  # 0.5

    def test_delta_delta_g_tie_resolves_to_reference(self) -> None:
        rk = FreeEnergyRanking({"ref": _result(-9.0), "other": _result(-9.0)})
        d = rk.delta_delta_g("ref", "other")
        assert d.value == pytest.approx(0.0)
        assert d.tighter == "ref"

    def test_delta_delta_g_unknown_label_raises(self) -> None:
        rk = FreeEnergyRanking({"A": _result(-9.0)})
        with pytest.raises(KeyError):
            rk.delta_delta_g("A", "missing")

    def test_len_and_iter(self) -> None:
        rk = FreeEnergyRanking({"A": _result(-9.5), "B": _result(-8.2), "C": _result(-11.0)})
        assert len(rk) == 3
        assert [label for label, _ in rk] == ["C", "A", "B"]  # iter follows ranking


class TestEngineBase:
    def test_engine_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            MMGBSAEngine()  # type: ignore[abstract]

    def test_not_installed_is_importerror(self) -> None:
        assert issubclass(MMGBSAEngineNotInstalledError, ImportError)

    def test_concrete_subclass(self) -> None:
        class Dummy(MMGBSAEngine):
            name = "Dummy"

            def run(self, trajectory, *, receptor, ligand, solvent_model="gb", **kwargs):  # type: ignore[no-untyped-def]
                return _result(-7.0)

        engine = Dummy()
        assert engine.name == "Dummy"
        assert repr(engine) == "Dummy()"
        out = engine.run(object(), receptor="chain A", ligand="resname LIG")
        assert out.delta_g == pytest.approx(-7.0)


def _contrib(residue: str, total: float, uncertainty: float = 0.1) -> ResidueContribution:
    return ResidueContribution(
        residue=residue,
        total=total,
        uncertainty=uncertainty,
        internal=0.0,
        vdw=total,
        electrostatic=0.0,
        polar_solvation=0.0,
        nonpolar_solvation=0.0,
    )


class TestResidueContribution:
    def test_fields_and_frozen(self) -> None:
        c = ResidueContribution("LEU 40", -18.86, 0.79, 22.06, -6.83, -32.53, -1.57, 0.01)
        assert c.residue == "LEU 40"
        assert c.total == pytest.approx(-18.86)
        assert c.vdw == pytest.approx(-6.83)
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.total = 0.0  # type: ignore[misc]

    def test_negative_uncertainty_rejected(self) -> None:
        with pytest.raises(ValueError, match="uncertainty must be >= 0"):
            _contrib("X", -1.0, uncertainty=-0.5)


class TestDecomposition:
    def test_order_lookup_and_membership(self) -> None:
        d = Decomposition([_contrib("LEU 40", -18.86), _contrib("THR 41", -23.99)])
        assert len(d) == 2
        assert list(d) == ["LEU 40", "THR 41"]  # report order
        assert d["THR 41"].total == pytest.approx(-23.99)
        assert "LEU 40" in d
        assert [c.residue for c in d.residues] == ["LEU 40", "THR 41"]

    def test_total_sums_contributions(self) -> None:
        d = Decomposition([_contrib("A", -18.86), _contrib("B", -23.99), _contrib("C", 9.12)])
        assert d.total == pytest.approx(-33.73)

    def test_hotspots_favorable_first(self) -> None:
        d = Decomposition(
            [_contrib("LEU 40", -18.86), _contrib("THR 41", -23.99), _contrib("RAL 241", 9.12)]
        )
        assert [c.residue for c in d.hotspots(2)] == ["THR 41", "LEU 40"]

    def test_hotspots_opposing(self) -> None:
        d = Decomposition([_contrib("THR 41", -23.99), _contrib("RAL 241", 9.12)])
        assert d.hotspots(1, favorable=False)[0].residue == "RAL 241"

    def test_hotspots_all_when_n_none(self) -> None:
        d = Decomposition([_contrib("A", -1.0), _contrib("B", -2.0)])
        assert len(d.hotspots()) == 2

    def test_duplicate_residue_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate residue"):
            Decomposition([_contrib("A", -1.0), _contrib("A", -2.0)])

    def test_empty_allowed(self) -> None:
        d = Decomposition([])
        assert len(d) == 0
        assert d.total == 0.0
        assert d.hotspots() == []
