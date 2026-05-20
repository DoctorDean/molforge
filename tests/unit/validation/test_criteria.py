"""Tests for the Criterion / NamedCriterion / CriteriaSet system."""

from __future__ import annotations

import pytest

from molforge.validation import CriteriaSet, Criterion, NamedCriterion


class TestAtomicCriteria:
    def test_gt(self) -> None:
        c = Criterion.gt("plddt", 80.0)
        assert c.evaluate({"plddt": 85.0}) is True
        assert c.evaluate({"plddt": 80.0}) is False  # strict
        assert c.evaluate({"plddt": 70.0}) is False

    def test_ge(self) -> None:
        c = Criterion.ge("plddt", 80.0)
        assert c.evaluate({"plddt": 80.0}) is True
        assert c.evaluate({"plddt": 79.999}) is False

    def test_lt(self) -> None:
        c = Criterion.lt("rmsd", 2.0)
        assert c.evaluate({"rmsd": 1.5}) is True
        assert c.evaluate({"rmsd": 2.0}) is False
        assert c.evaluate({"rmsd": 2.5}) is False

    def test_le(self) -> None:
        c = Criterion.le("rmsd", 2.0)
        assert c.evaluate({"rmsd": 2.0}) is True
        assert c.evaluate({"rmsd": 2.001}) is False

    def test_eq(self) -> None:
        c = Criterion.eq("model_name", "v_48_020")
        assert c.evaluate({"model_name": "v_48_020"}) is True
        assert c.evaluate({"model_name": "v_48_010"}) is False

    def test_ne(self) -> None:
        c = Criterion.ne("model_name", "v_48_020")
        assert c.evaluate({"model_name": "v_48_010"}) is True
        assert c.evaluate({"model_name": "v_48_020"}) is False


class TestMissingMetric:
    def test_missing_metric_raises(self) -> None:
        c = Criterion.gt("plddt", 80.0)
        with pytest.raises(KeyError, match="plddt"):
            c.evaluate({"tm_score": 0.6})

    def test_none_value_fails(self) -> None:
        c = Criterion.gt("plddt", 80.0)
        assert c.evaluate({"plddt": None}) is False


class TestUnknownOperator:
    def test_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown operator"):
            Criterion._atomic("plddt", "bogus", 80.0)


class TestComposition:
    def test_and(self) -> None:
        plddt_ok = Criterion.gt("plddt", 80.0)
        tm_ok = Criterion.gt("tm", 0.5)
        combined = plddt_ok & tm_ok
        assert combined.evaluate({"plddt": 85, "tm": 0.6}) is True
        assert combined.evaluate({"plddt": 75, "tm": 0.6}) is False
        assert combined.evaluate({"plddt": 85, "tm": 0.4}) is False
        assert combined.evaluate({"plddt": 75, "tm": 0.4}) is False

    def test_or(self) -> None:
        a_high_plddt = Criterion.gt("plddt_a", 80.0)
        b_high_plddt = Criterion.gt("plddt_b", 80.0)
        either = a_high_plddt | b_high_plddt
        assert either.evaluate({"plddt_a": 85, "plddt_b": 70}) is True
        assert either.evaluate({"plddt_a": 75, "plddt_b": 85}) is True
        assert either.evaluate({"plddt_a": 75, "plddt_b": 70}) is False
        assert either.evaluate({"plddt_a": 85, "plddt_b": 85}) is True

    def test_not(self) -> None:
        c = ~Criterion.gt("plddt", 80.0)
        assert c.evaluate({"plddt": 85}) is False
        assert c.evaluate({"plddt": 75}) is True

    def test_complex_composition(self) -> None:
        # (pLDDT > 80 AND TM > 0.5) OR (mean_lddt > 0.9)
        c = (Criterion.gt("plddt", 80) & Criterion.gt("tm", 0.5)) | Criterion.gt("mean_lddt", 0.9)
        # First branch passes
        assert c.evaluate({"plddt": 85, "tm": 0.6, "mean_lddt": 0.5}) is True
        # Second branch passes
        assert c.evaluate({"plddt": 70, "tm": 0.3, "mean_lddt": 0.95}) is True
        # Neither passes
        assert c.evaluate({"plddt": 70, "tm": 0.3, "mean_lddt": 0.5}) is False

    def test_combined_with_non_criterion_returns_notimplemented(self) -> None:
        c = Criterion.gt("plddt", 80)
        assert c.__and__("not a criterion") is NotImplemented
        assert c.__or__(42) is NotImplemented


class TestMetricNames:
    def test_atomic(self) -> None:
        c = Criterion.gt("plddt", 80)
        assert c.metric_names == frozenset({"plddt"})

    def test_composed_collects_all(self) -> None:
        c = Criterion.gt("plddt", 80) & Criterion.lt("rmsd", 2.0)
        assert c.metric_names == frozenset({"plddt", "rmsd"})

    def test_or_collects_all(self) -> None:
        c = Criterion.gt("a", 1) | Criterion.gt("b", 1) | Criterion.gt("c", 1)
        assert c.metric_names == frozenset({"a", "b", "c"})

    def test_not_preserves_names(self) -> None:
        c = ~Criterion.gt("plddt", 80)
        assert c.metric_names == frozenset({"plddt"})


class TestRepr:
    def test_atomic_repr(self) -> None:
        r = repr(Criterion.gt("plddt", 80))
        assert "plddt" in r
        assert ">" in r
        assert "80" in r

    def test_and_repr(self) -> None:
        c = Criterion.gt("plddt", 80) & Criterion.lt("rmsd", 2.0)
        r = repr(c)
        assert "AND" in r
        assert "plddt" in r
        assert "rmsd" in r

    def test_or_repr(self) -> None:
        c = Criterion.gt("a", 1) | Criterion.gt("b", 1)
        assert "OR" in repr(c)

    def test_not_repr(self) -> None:
        c = ~Criterion.gt("plddt", 80)
        assert "NOT" in repr(c)


class TestNamedCriterion:
    def test_basic(self) -> None:
        nc = NamedCriterion(
            name="fold_quality",
            criterion=Criterion.gt("plddt", 80),
            description="Mean pLDDT above 80",
        )
        assert nc.name == "fold_quality"
        assert nc.evaluate({"plddt": 85}) is True
        assert nc.evaluate({"plddt": 70}) is False

    def test_metric_names_proxy(self) -> None:
        nc = NamedCriterion(name="x", criterion=Criterion.gt("plddt", 80))
        assert nc.metric_names == frozenset({"plddt"})


class TestCriteriaSet:
    def test_empty(self) -> None:
        cs = CriteriaSet()
        assert cs.evaluate({}) == {}
        assert cs.passes({}) is True  # vacuous truth
        assert cs.metric_names == frozenset()

    def test_add_and_evaluate(self) -> None:
        cs = (
            CriteriaSet()
            .add("plddt_ok", Criterion.gt("plddt", 80))
            .add("tm_ok", Criterion.gt("tm", 0.5))
        )
        result = cs.evaluate({"plddt": 85, "tm": 0.6})
        assert result == {"plddt_ok": True, "tm_ok": True}
        assert cs.passes({"plddt": 85, "tm": 0.6}) is True

    def test_one_failing_criterion_fails_overall(self) -> None:
        cs = (
            CriteriaSet()
            .add("plddt_ok", Criterion.gt("plddt", 80))
            .add("tm_ok", Criterion.gt("tm", 0.5))
        )
        result = cs.evaluate({"plddt": 85, "tm": 0.4})
        assert result == {"plddt_ok": True, "tm_ok": False}
        assert cs.passes({"plddt": 85, "tm": 0.4}) is False

    def test_metric_names_union(self) -> None:
        cs = (
            CriteriaSet()
            .add("a_ok", Criterion.gt("a", 1))
            .add("b_ok", Criterion.lt("b", 5))
            .add("ab_ok", Criterion.gt("a", 0) & Criterion.gt("c", 0))
        )
        assert cs.metric_names == frozenset({"a", "b", "c"})

    def test_chaining_returns_self(self) -> None:
        cs = CriteriaSet()
        result = cs.add("x", Criterion.gt("x", 0))
        assert result is cs
