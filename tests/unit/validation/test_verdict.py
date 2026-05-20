"""Tests for Verdict and rank_verdicts."""

from __future__ import annotations

from molforge.validation import Verdict, rank_verdicts


class TestVerdictBasics:
    def test_default_fields(self) -> None:
        v = Verdict(design_id="abc")
        assert v.design_id == "abc"
        assert v.values == {}
        assert v.criteria_results == {}
        assert v.passed is False
        assert v.score == float("inf")
        assert v.metadata == {}

    def test_failed_passed_criteria_properties(self) -> None:
        v = Verdict(
            design_id="x",
            criteria_results={"a": True, "b": False, "c": True, "d": False},
        )
        # Order of criterion lookup is dict-insertion order
        assert v.failed_criteria == ["b", "d"]
        assert v.passed_criteria == ["a", "c"]

    def test_all_passed(self) -> None:
        v = Verdict(design_id="x", criteria_results={"a": True, "b": True})
        assert v.failed_criteria == []
        assert v.passed_criteria == ["a", "b"]

    def test_none_passed(self) -> None:
        v = Verdict(design_id="x", criteria_results={"a": False})
        assert v.failed_criteria == ["a"]
        assert v.passed_criteria == []


class TestVerdictRepr:
    def test_pass_shown(self) -> None:
        v = Verdict(design_id="MKTV", passed=True, score=1.234)
        r = repr(v)
        assert "PASS" in r
        assert "MKTV" in r
        assert "1.234" in r

    def test_fail_shown(self) -> None:
        v = Verdict(design_id="MKTV", passed=False, score=2.5)
        assert "FAIL" in repr(v)

    def test_long_id_truncated(self) -> None:
        v = Verdict(design_id="A" * 100, passed=True, score=1.0)
        r = repr(v)
        assert "..." in r
        assert "A" * 100 not in r


class TestRankVerdicts:
    def _make(self, did: str, score: float, passed: bool = True, **values: float) -> Verdict:
        return Verdict(design_id=did, score=score, passed=passed, values=dict(values))

    def test_sort_by_score_ascending(self) -> None:
        a = self._make("a", 1.0)
        b = self._make("b", 3.0)
        c = self._make("c", 2.0)
        out = rank_verdicts([a, b, c])
        assert [v.design_id for v in out] == ["a", "c", "b"]

    def test_only_passed_filters(self) -> None:
        a = self._make("a", 1.0, passed=True)
        b = self._make("b", 0.5, passed=False)
        c = self._make("c", 2.0, passed=True)
        out = rank_verdicts([a, b, c], only_passed=True)
        assert [v.design_id for v in out] == ["a", "c"]
        # b is dropped even though its score is the lowest

    def test_sort_by_metric(self) -> None:
        a = self._make("a", 5.0, plddt=85.0)
        b = self._make("b", 5.0, plddt=92.0)
        c = self._make("c", 5.0, plddt=78.0)
        # by="plddt" sorts ascending by plddt regardless of score
        out = rank_verdicts([a, b, c], by="plddt")
        assert [v.design_id for v in out] == ["c", "a", "b"]

    def test_missing_metric_sorts_last(self) -> None:
        a = self._make("a", 5.0)  # no plddt
        b = self._make("b", 5.0, plddt=80.0)
        out = rank_verdicts([a, b], by="plddt")
        # missing -> inf -> sorts last
        assert [v.design_id for v in out] == ["b", "a"]

    def test_stable_sort_preserves_input_order(self) -> None:
        a = self._make("a", 1.0)
        b = self._make("b", 1.0)
        c = self._make("c", 1.0)
        out = rank_verdicts([c, a, b])
        # All ties -> input order preserved
        assert [v.design_id for v in out] == ["c", "a", "b"]

    def test_empty_input(self) -> None:
        assert rank_verdicts([]) == []
        assert rank_verdicts([], only_passed=True) == []
