"""Tests for the alchemlyb / FEP ingest.

alchemlyb isn't a dependency, so these use lightweight fakes: plain numpy
arrays for :func:`from_delta_f`, and a ``_Frame`` (array-like carrying
``.attrs``) inside a ``_FakeEstimator`` for :func:`from_alchemlyb`. The kT
→ kcal/mol conversion is checked against alchemlyb's own documented
example (3.041156 kT at 300 K → 1.813021 kcal/mol).
"""

from __future__ import annotations

import numpy as np
import pytest

from molforge.freeenergy import FreeEnergyRanking
from molforge.wrappers.freeenergy import (
    from_alchemlyb,
    from_delta_f,
    relative_binding_free_energy,
)

# alchemlyb docs' MBAR (Coulomb) example: delta_f_ row 0, and its errors.
DELTA_F = np.array(
    [
        [0.0, 1.619069, 2.557990, 2.986302, 3.041156],
        [-1.619069, 0.0, 0.938921, 1.367232, 1.422086],
        [-2.557990, -0.938921, 0.0, 0.428311, 0.483165],
        [-2.986302, -1.367232, -0.428311, 0.0, 0.054854],
        [-3.041156, -1.422086, -0.483165, -0.054854, 0.0],
    ]
)
D_DELTA_F = np.array(
    [
        [0.0, 0.008802, 0.014432, 0.018097, 0.020879],
        [0.008802, 0.0, 0.006642, 0.011404, 0.015143],
        [0.014432, 0.006642, 0.0, 0.005362, 0.009983],
        [0.018097, 0.011404, 0.005362, 0.0, 0.005133],
        [0.020879, 0.015143, 0.009983, 0.005133, 0.0],
    ]
)


class _Frame:
    """Minimal stand-in for a pandas DataFrame: array-like + ``.attrs``."""

    def __init__(self, array: np.ndarray, attrs: dict) -> None:
        self._a = np.asarray(array, dtype=float)
        self.attrs = attrs

    def __array__(self, dtype=None):  # noqa: ANN001
        return self._a if dtype is None else self._a.astype(dtype)


class _FakeEstimator:
    def __init__(self, delta_f, d_delta_f) -> None:  # noqa: ANN001
        self.delta_f_ = delta_f
        self.d_delta_f_ = d_delta_f


class MBAR(_FakeEstimator):
    """Named so ``type(estimator).__name__`` is the real estimator name."""


class TestFromDeltaF:
    def test_kt_conversion_matches_alchemlyb(self) -> None:
        r = from_delta_f(DELTA_F, D_DELTA_F, temperature=300.0)
        # alchemlyb to_kcalmol(3.041156 kT) == 1.813021 kcal/mol
        assert r.delta_g == pytest.approx(1.813021, abs=1e-4)
        assert r.uncertainty == pytest.approx(0.020879 * 8.314462618 / 4184.0 * 300.0, rel=1e-6)

    def test_kcalmol_passthrough(self) -> None:
        r = from_delta_f(
            np.array([[0.0, 1.813021]]), np.array([[0.0, 0.0124]]), energy_unit="kcal/mol"
        )
        assert r.delta_g == pytest.approx(1.813021)
        assert r.uncertainty == pytest.approx(0.0124)

    def test_kjmol_conversion(self) -> None:
        r = from_delta_f(
            np.array([[0.0, 7.585673]]), np.array([[0.0, 0.05]]), energy_unit="kJ/mol"
        )
        assert r.delta_g == pytest.approx(7.585673 / 4.184, rel=1e-9)

    def test_defaults_and_shape(self) -> None:
        r = from_delta_f(DELTA_F, D_DELTA_F, temperature=300.0)
        assert r.method == "FEP"
        assert r.components is None  # alchemical dG has no MM/GBSA breakdown
        assert r.metadata["n_lambda_states"] == 5
        assert r.metadata["source_energy_unit"] == "kT"
        assert r.metadata["temperature"] == 300.0

    def test_uncertainty_non_negative(self) -> None:
        # A negative error entry still yields a non-negative uncertainty.
        r = from_delta_f(np.array([[0.0, -3.0]]), np.array([[0.0, -0.2]]), energy_unit="kcal/mol")
        assert r.delta_g == pytest.approx(-3.0)
        assert r.uncertainty == pytest.approx(0.2)

    def test_method_and_metadata_override(self) -> None:
        r = from_delta_f(
            DELTA_F, D_DELTA_F, temperature=300.0, method="TI", metadata={"leg": "coul"}
        )
        assert r.method == "TI"
        assert r.metadata["leg"] == "coul"


class TestAttrs:
    def test_reads_unit_and_temperature_from_attrs(self) -> None:
        frame = _Frame(DELTA_F, {"temperature": 300, "energy_unit": "kT"})
        err = _Frame(D_DELTA_F, {"temperature": 300, "energy_unit": "kT"})
        r = from_delta_f(frame, err)
        assert r.delta_g == pytest.approx(1.813021, abs=1e-4)
        assert r.metadata["temperature"] == 300.0

    def test_explicit_args_override_attrs(self) -> None:
        # attrs say kcal/mol; explicit energy_unit=kT + temperature wins.
        frame = _Frame(DELTA_F, {"energy_unit": "kcal/mol"})
        err = _Frame(D_DELTA_F, {"energy_unit": "kcal/mol"})
        r = from_delta_f(frame, err, temperature=300.0, energy_unit="kT")
        assert r.delta_g == pytest.approx(1.813021, abs=1e-4)


class TestFromAlchemlyb:
    def test_infers_method_and_records_estimator(self) -> None:
        est = MBAR(
            _Frame(DELTA_F, {"temperature": 300, "energy_unit": "kT"}),
            _Frame(D_DELTA_F, {"temperature": 300, "energy_unit": "kT"}),
        )
        r = from_alchemlyb(est)
        assert r.method == "MBAR"
        assert r.metadata["estimator"] == "MBAR"
        assert r.delta_g == pytest.approx(1.813021, abs=1e-4)

    def test_method_override(self) -> None:
        est = MBAR(
            _Frame(DELTA_F, {"temperature": 300, "energy_unit": "kT"}),
            _Frame(D_DELTA_F, {"temperature": 300, "energy_unit": "kT"}),
        )
        r = from_alchemlyb(est, method="FEP/MBAR")
        assert r.method == "FEP/MBAR"
        assert r.metadata["estimator"] == "MBAR"  # still recorded


class TestErrors:
    def test_kt_without_temperature_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature is required"):
            from_delta_f(DELTA_F, D_DELTA_F)  # unit defaults to kT, no T

    def test_unknown_unit_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown energy_unit"):
            from_delta_f(DELTA_F, D_DELTA_F, energy_unit="hartree")

    def test_non_2d_raises(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            from_delta_f(np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0]), energy_unit="kcal/mol")


class TestRankingIntegration:
    def test_fep_results_rank_and_compare(self) -> None:
        # Two ligands' alchemical ΔGs plug straight into FreeEnergyRanking.
        lig_a = from_delta_f(
            np.array([[0.0, -9.0]]), np.array([[0.0, 0.3]]), energy_unit="kcal/mol", method="FEP"
        )
        lig_b = from_delta_f(
            np.array([[0.0, -11.0]]), np.array([[0.0, 0.4]]), energy_unit="kcal/mol", method="FEP"
        )
        ranking = FreeEnergyRanking({"lig_a": lig_a, "lig_b": lig_b})

        assert ranking.best[0] == "lig_b"  # tighter (more negative)
        ddg = ranking.delta_delta_g("lig_a", "lig_b")
        assert ddg.value == pytest.approx(-2.0)
        assert ddg.uncertainty == pytest.approx((0.3**2 + 0.4**2) ** 0.5)


class TestRelativeBindingFreeEnergy:
    def _leg(self, dg: float, unc: float):
        return from_delta_f(
            np.array([[0.0, dg]]), np.array([[0.0, unc]]), energy_unit="kcal/mol"
        )

    def test_cycle_value_and_propagation(self) -> None:
        # complex leg -12, solvent leg -3 -> ΔΔG = -9 (B tighter), σ = hypot(.3,.4)
        ddg = relative_binding_free_energy(
            self._leg(-12.0, 0.3), self._leg(-3.0, 0.4), reference="A", other="B"
        )
        assert ddg.value == pytest.approx(-9.0)
        assert ddg.uncertainty == pytest.approx(0.5)
        assert ddg.reference == "A" and ddg.other == "B"
        assert ddg.tighter == "B"

    def test_reference_tighter(self) -> None:
        # complex leg less favorable than solvent -> positive ΔΔG, reference tighter
        ddg = relative_binding_free_energy(
            self._leg(-3.0, 0.2), self._leg(-12.0, 0.2), reference="A", other="B"
        )
        assert ddg.value == pytest.approx(9.0)
        assert ddg.tighter == "A"

    def test_exact_tie_is_reference(self) -> None:
        ddg = relative_binding_free_energy(
            self._leg(-5.0, 0.1), self._leg(-5.0, 0.1), reference="A", other="B"
        )
        assert ddg.value == pytest.approx(0.0)
        assert ddg.tighter == "A"  # tie -> reference

    def test_star_map_ranks_relative_to_reference(self) -> None:
        # A star map (edges from a reference) -> reference-relative results
        # plug into FreeEnergyRanking.
        edges = {
            "B": relative_binding_free_energy(self._leg(-12.0, 0.3), self._leg(-3.0, 0.3), reference="A", other="B"),
            "C": relative_binding_free_energy(self._leg(-7.0, 0.3), self._leg(-3.0, 0.3), reference="A", other="C"),
        }
        results = {"A": from_delta_f(np.array([[0.0, 0.0]]), np.array([[0.0, 0.0]]), energy_unit="kcal/mol")}
        for name, ddg in edges.items():
            results[name] = from_delta_f(
                np.array([[0.0, ddg.value]]),
                np.array([[0.0, ddg.uncertainty]]),
                energy_unit="kcal/mol",
                method="FEP (ΔΔG)",
            )
        ranking = FreeEnergyRanking(results)
        assert [name for name, _ in ranking.ranked] == ["B", "C", "A"]  # B (-9) < C (-4) < A (0)
