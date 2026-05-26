"""Tests for cross_validate and consensus."""

from __future__ import annotations

import pytest

from molforge.validation import (
    CriteriaSet,
    Criterion,
    Verdict,
    consensus,
    cross_validate,
)


# A simple deterministic "validator" that returns precomputed values
# keyed on the design (a string).
def _make_lookup_validator(
    table: dict[str, dict[str, float]],
) -> callable:  # type: ignore[valid-type]
    def validator(design: str) -> dict[str, float]:
        if design not in table:
            raise ValueError(f"unknown design {design!r}")
        return table[design]

    return validator


class TestCrossValidateBasics:
    def test_single_validator_namespaces_metrics(self) -> None:
        designs = ["a", "b"]
        validator = _make_lookup_validator(
            {
                "a": {"plddt": 85.0, "tm": 0.7},
                "b": {"plddt": 70.0, "tm": 0.4},
            }
        )
        criteria = (
            CriteriaSet()
            .add("plddt_ok", Criterion.gt("esmfold.plddt", 80))
            .add("tm_ok", Criterion.gt("esmfold.tm", 0.5))
        )
        verdicts = cross_validate(
            designs=designs,
            validators={"esmfold": validator},
            criteria=criteria,
        )
        assert len(verdicts) == 2
        # Metrics are namespaced
        assert "esmfold.plddt" in verdicts[0].values
        assert "esmfold.tm" in verdicts[0].values
        # First passes both
        assert verdicts[0].passed is True
        assert verdicts[0].criteria_results == {"plddt_ok": True, "tm_ok": True}
        # Second fails both
        assert verdicts[1].passed is False
        assert verdicts[1].criteria_results == {"plddt_ok": False, "tm_ok": False}

    def test_design_order_preserved(self) -> None:
        designs = ["c", "a", "b"]
        validator = _make_lookup_validator({d: {"x": 1.0} for d in designs})
        cs = CriteriaSet().add("x_ok", Criterion.gt("v.x", 0))
        verdicts = cross_validate(designs=designs, validators={"v": validator}, criteria=cs)
        assert [v.design_id for v in verdicts] == ["c", "a", "b"]

    def test_multiple_validators_merge_into_values(self) -> None:
        designs = ["a"]
        v1 = _make_lookup_validator({"a": {"plddt": 85.0}})
        v2 = _make_lookup_validator({"a": {"tm": 0.7}})
        cs = (
            CriteriaSet()
            .add("plddt_ok", Criterion.gt("esmfold.plddt", 80))
            .add("tm_ok", Criterion.gt("alphafold.tm", 0.5))
        )
        verdicts = cross_validate(
            designs=designs,
            validators={"esmfold": v1, "alphafold": v2},
            criteria=cs,
        )
        assert verdicts[0].values == {
            "esmfold.plddt": 85.0,
            "alphafold.tm": 0.7,
        }
        assert verdicts[0].passed is True


class TestCrossValidateScoring:
    def test_score_from_metric(self) -> None:
        designs = ["a", "b"]
        validator = _make_lookup_validator(
            {
                "a": {"plddt": 85.0},
                "b": {"plddt": 70.0},
            }
        )
        cs = CriteriaSet().add("plddt_ok", Criterion.gt("v.plddt", 80))
        verdicts = cross_validate(
            designs=designs,
            validators={"v": validator},
            criteria=cs,
            score_metric="v.plddt",
        )
        assert verdicts[0].score == 85.0
        assert verdicts[1].score == 70.0

    def test_default_score_counts_failed_criteria(self) -> None:
        designs = ["a", "b"]
        validator = _make_lookup_validator(
            {
                "a": {"plddt": 85.0, "tm": 0.7},
                "b": {"plddt": 70.0, "tm": 0.4},
            }
        )
        cs = (
            CriteriaSet()
            .add("plddt_ok", Criterion.gt("v.plddt", 80))
            .add("tm_ok", Criterion.gt("v.tm", 0.5))
        )
        verdicts = cross_validate(designs=designs, validators={"v": validator}, criteria=cs)
        # 'a' passes both, score = 0
        assert verdicts[0].score == 0.0
        # 'b' fails both, score = 2
        assert verdicts[1].score == 2.0

    def test_score_metric_missing_yields_inf(self) -> None:
        designs = ["a"]
        validator = _make_lookup_validator({"a": {"plddt": 85.0}})
        cs = CriteriaSet().add("plddt_ok", Criterion.gt("v.plddt", 80))
        verdicts = cross_validate(
            designs=designs,
            validators={"v": validator},
            criteria=cs,
            score_metric="v.nonexistent",
        )
        # Score metric missing -> falls back to counting failed criteria
        assert verdicts[0].score == 0.0


class TestCrossValidateErrorHandling:
    def test_record_mode_collects_errors(self) -> None:
        designs = ["a", "b"]

        def failing_validator(d: str) -> dict[str, float]:
            if d == "b":
                raise RuntimeError("oops")
            return {"plddt": 85.0}

        cs = CriteriaSet().add("plddt_ok", Criterion.gt("v.plddt", 80))
        # on_error="record" is now opt-in (the default flipped to "raise").
        verdicts = cross_validate(
            designs=designs,
            validators={"v": failing_validator},
            criteria=cs,
            on_error="record",
        )
        assert verdicts[0].passed is True
        # b failed: errors recorded, marked failed, criterion eval gracefully
        # marks plddt_ok as False (no plddt available)
        assert verdicts[1].passed is False
        assert verdicts[1].criteria_results == {"plddt_ok": False}
        assert "validator_errors" in verdicts[1].metadata
        assert "v" in verdicts[1].metadata["validator_errors"]
        assert "oops" in verdicts[1].metadata["validator_errors"]["v"]

    def test_raise_is_the_default(self) -> None:
        """A validator exception propagates when on_error is not given."""

        def failing_validator(d: str) -> dict[str, float]:
            raise RuntimeError("kaboom")

        cs = CriteriaSet().add("x", Criterion.gt("v.x", 0))
        with pytest.raises(RuntimeError, match="kaboom"):
            cross_validate(
                designs=["a"],
                validators={"v": failing_validator},
                criteria=cs,
            )

    def test_raise_mode_propagates(self) -> None:
        def failing_validator(d: str) -> dict[str, float]:
            raise RuntimeError("oops")

        cs = CriteriaSet().add("x", Criterion.gt("v.x", 0))
        with pytest.raises(RuntimeError, match="oops"):
            cross_validate(
                designs=["a"],
                validators={"v": failing_validator},
                criteria=cs,
                on_error="raise",
            )

    def test_unknown_on_error_raises(self) -> None:
        with pytest.raises(ValueError, match="on_error"):
            cross_validate(
                designs=["a"],
                validators={"v": lambda d: {"x": 1}},
                criteria=CriteriaSet(),
                on_error="bogus",
            )

    def test_validator_with_error_doesnt_pass_even_if_criteria_pass(self) -> None:
        """If a validator throws, the verdict should be marked failed even if
        the criteria (somehow) pass on the partial data."""
        designs = ["a"]

        def validator_a(d: str) -> dict[str, float]:
            return {"plddt": 85.0}

        def failing_validator(d: str) -> dict[str, float]:
            raise RuntimeError("oops")

        cs = CriteriaSet().add("plddt_ok", Criterion.gt("ok.plddt", 80))
        verdicts = cross_validate(
            designs=designs,
            validators={"ok": validator_a, "broken": failing_validator},
            criteria=cs,
            on_error="record",
        )
        # plddt criterion passes (ok validator worked), but broken validator
        # errored -> overall verdict fails
        assert verdicts[0].criteria_results == {"plddt_ok": True}
        assert verdicts[0].passed is False
        assert "validator_errors" in verdicts[0].metadata


class TestCrossValidateMisc:
    def test_custom_design_id(self) -> None:
        designs = [("alpha", 1), ("beta", 2)]
        validator = lambda d: {"x": d[1]}  # noqa: E731
        cs = CriteriaSet().add("x_ok", Criterion.gt("v.x", 0))
        verdicts = cross_validate(
            designs=designs,
            validators={"v": validator},
            criteria=cs,
            design_id=lambda d: d[0],
        )
        assert [v.design_id for v in verdicts] == ["alpha", "beta"]

    def test_long_default_id_truncated(self) -> None:
        # When no design_id is provided, fall back to str(design) capped at 60
        designs = ["A" * 100]
        validator = lambda d: {"x": 1}  # noqa: E731
        cs = CriteriaSet().add("x_ok", Criterion.gt("v.x", 0))
        verdicts = cross_validate(designs=designs, validators={"v": validator}, criteria=cs)
        assert verdicts[0].design_id.endswith("...")
        assert len(verdicts[0].design_id) <= 60

    def test_empty_designs(self) -> None:
        cs = CriteriaSet().add("x", Criterion.gt("v.x", 0))
        verdicts = cross_validate(
            designs=[],
            validators={"v": lambda d: {"x": 1}},
            criteria=cs,
        )
        assert verdicts == []


class TestConsensusBasics:
    def _verdict(self, did: str, passed: bool, **values: float) -> Verdict:
        return Verdict(
            design_id=did,
            values=dict(values),
            criteria_results={"x_ok": passed},
            passed=passed,
        )

    def test_all_mode_requires_unanimity(self) -> None:
        esm = [self._verdict("a", True), self._verdict("b", True)]
        af = [self._verdict("a", True), self._verdict("b", False)]
        merged = consensus({"esm": esm, "af": af}, mode="all")
        assert len(merged) == 2
        # a passes both -> pass
        assert merged[0].design_id == "a"
        assert merged[0].passed is True
        # b only passes esm -> fail in all-mode
        assert merged[1].design_id == "b"
        assert merged[1].passed is False

    def test_any_mode(self) -> None:
        esm = [self._verdict("a", False), self._verdict("b", False)]
        af = [self._verdict("a", True), self._verdict("b", False)]
        merged = consensus({"esm": esm, "af": af}, mode="any")
        assert merged[0].passed is True  # af confirms
        assert merged[1].passed is False  # nobody confirms

    def test_majority_mode(self) -> None:
        a = [self._verdict("x", True), self._verdict("y", True)]
        b = [self._verdict("x", True), self._verdict("y", False)]
        c = [self._verdict("x", False), self._verdict("y", False)]
        # 3 validators -> majority = 2
        merged = consensus({"a": a, "b": b, "c": c}, mode="majority")
        # x: 2/3 pass -> pass
        assert merged[0].passed is True
        # y: 1/3 pass -> fail
        assert merged[1].passed is False

    def test_threshold_mode(self) -> None:
        a = [self._verdict("x", True)]
        b = [self._verdict("x", True)]
        c = [self._verdict("x", False)]
        merged = consensus({"a": a, "b": b, "c": c}, mode="threshold", threshold=2)
        assert merged[0].passed is True
        merged = consensus({"a": a, "b": b, "c": c}, mode="threshold", threshold=3)
        assert merged[0].passed is False


class TestConsensusValidation:
    def _verdict(self, did: str, passed: bool) -> Verdict:
        return Verdict(design_id=did, passed=passed, criteria_results={"x": passed})

    def test_unknown_mode(self) -> None:
        with pytest.raises(ValueError, match="unknown mode"):
            consensus({"a": [self._verdict("x", True)]}, mode="bogus")

    def test_threshold_required_with_threshold_mode(self) -> None:
        with pytest.raises(ValueError, match="threshold mode requires"):
            consensus({"a": [self._verdict("x", True)]}, mode="threshold")

    def test_threshold_with_non_threshold_mode(self) -> None:
        with pytest.raises(ValueError, match="threshold is only valid"):
            consensus({"a": [self._verdict("x", True)]}, mode="all", threshold=1)

    def test_threshold_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            consensus(
                {"a": [self._verdict("x", True)]},
                mode="threshold",
                threshold=2,  # only 1 validator
            )

    def test_design_id_mismatch_raises(self) -> None:
        esm = [self._verdict("a", True), self._verdict("b", True)]
        af = [self._verdict("a", True)]  # missing 'b'
        with pytest.raises(ValueError, match="disagree on design IDs"):
            consensus({"esm": esm, "af": af}, mode="all")

    def test_empty_input(self) -> None:
        assert consensus({}, mode="all") == []


class TestConsensusMetadata:
    def test_records_counts(self) -> None:
        a = [Verdict(design_id="x", passed=True, criteria_results={"c": True})]
        b = [Verdict(design_id="x", passed=False, criteria_results={"c": False})]
        c = [Verdict(design_id="x", passed=True, criteria_results={"c": True})]
        merged = consensus({"a": a, "b": b, "c": c}, mode="all")
        assert merged[0].metadata["consensus_mode"] == "all"
        assert merged[0].metadata["n_validators"] == 3
        assert merged[0].metadata["n_passed"] == 2
        assert merged[0].metadata["required"] == 3

    def test_score_counts_failed_validators(self) -> None:
        a = [Verdict(design_id="x", passed=True, criteria_results={})]
        b = [Verdict(design_id="x", passed=False, criteria_results={})]
        merged = consensus({"a": a, "b": b}, mode="all")
        # 1 of 2 failed -> score = 1
        assert merged[0].score == 1.0

    def test_criteria_results_namespaced(self) -> None:
        a = [Verdict(design_id="x", passed=True, criteria_results={"plddt_ok": True})]
        b = [Verdict(design_id="x", passed=True, criteria_results={"plddt_ok": False})]
        merged = consensus({"esm": a, "af": b}, mode="any")
        # Criterion results from both validators preserved with namespacing
        assert merged[0].criteria_results == {
            "esm.plddt_ok": True,
            "af.plddt_ok": False,
        }

    def test_values_merged_from_all_validators(self) -> None:
        a = [Verdict(design_id="x", passed=True, values={"esm.plddt": 85.0})]
        b = [Verdict(design_id="x", passed=True, values={"af.plddt": 88.0})]
        merged = consensus({"esm": a, "af": b}, mode="all")
        assert merged[0].values == {"esm.plddt": 85.0, "af.plddt": 88.0}
