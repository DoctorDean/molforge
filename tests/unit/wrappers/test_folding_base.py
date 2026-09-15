"""Tests for the FoldingEngine abstract base class and helpers."""

from __future__ import annotations

import pytest

from molforge.core import AtomArray, Protein
from molforge.wrappers.folding._base import (
    FoldingEngine,
    _reject_unknown_kwargs,
    _validate_sequence,
)


class TestValidateSequence:
    def test_basic_cleanup(self) -> None:
        assert _validate_sequence("MKTV") == "MKTV"

    def test_strips_whitespace(self) -> None:
        assert _validate_sequence("  MK\nTV  \t") == "MKTV"

    def test_uppercases(self) -> None:
        assert _validate_sequence("mktv") == "MKTV"

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            _validate_sequence("   ")

    def test_non_letter_raises(self) -> None:
        with pytest.raises(ValueError, match="non-letter"):
            _validate_sequence("MKT*V")

    def test_non_letter_lists_offenders(self) -> None:
        with pytest.raises(ValueError, match=r"\['\*'"):
            _validate_sequence("MKT*V")


class _DummyEngine(FoldingEngine):
    """Minimal concrete engine for testing the ABC contract."""

    name = "Dummy"

    def predict(self, sequence: str, **kwargs: object) -> Protein:
        return Protein(AtomArray(0), name=f"dummy:{sequence}")


class TestEngineContract:
    def test_must_implement_predict(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            FoldingEngine()  # type: ignore[abstract]

    def test_subclass_can_be_instantiated(self) -> None:
        engine = _DummyEngine()
        assert isinstance(engine, FoldingEngine)

    def test_predict_many_serial_default(self) -> None:
        engine = _DummyEngine()
        results = engine.predict_many(["AAA", "GGG"])
        assert len(results) == 2
        assert results[0].name == "dummy:AAA"
        assert results[1].name == "dummy:GGG"

    def test_repr(self) -> None:
        assert repr(_DummyEngine()) == "_DummyEngine()"


class TestRejectUnknownKwargs:
    """Unconsumed per-call keywords raise instead of being dropped.

    The ``**kwargs`` on every ``predict`` used to swallow anything, so
    ``predict(seq, seed=7)`` ran unseeded while the provenance claimed
    otherwise. These checks pin the replacement: a locatable TypeError.
    """

    def test_empty_kwargs_pass(self) -> None:
        assert _reject_unknown_kwargs({}, engine="E", method="predict") is None

    def test_supported_kwarg_passes(self) -> None:
        assert (
            _reject_unknown_kwargs({"seed": 7}, engine="E", method="predict", supported={"seed"})
            is None
        )

    def test_unknown_kwarg_raises(self) -> None:
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            _reject_unknown_kwargs({"temperature": 0.5}, engine="E", method="predict")

    def test_message_names_engine_method_and_offender(self) -> None:
        with pytest.raises(TypeError) as excinfo:
            _reject_unknown_kwargs({"nsamples": 3}, engine="ESMFold", method="predict")
        message = str(excinfo.value)
        assert "ESMFold.predict()" in message
        assert "'nsamples'" in message

    def test_all_offenders_reported_sorted(self) -> None:
        with pytest.raises(TypeError) as excinfo:
            _reject_unknown_kwargs({"b": 2, "a": 1}, engine="E", method="predict")
        assert "'a', 'b'" in str(excinfo.value)

    def test_supported_options_listed_when_some_exist(self) -> None:
        """A near-miss should point at what the caller probably meant."""
        with pytest.raises(TypeError) as excinfo:
            _reject_unknown_kwargs({"sed": 1}, engine="Boltz", method="predict", supported={"seed"})
        assert "Supported per-call option(s): 'seed'" in str(excinfo.value)

    def test_says_none_accepted_when_no_options_exist(self) -> None:
        with pytest.raises(TypeError) as excinfo:
            _reject_unknown_kwargs({"seed": 7}, engine="ESMFold", method="predict")
        assert "takes no per-call options" in str(excinfo.value)

    def test_constructor_example_shown_when_given(self) -> None:
        with pytest.raises(TypeError) as excinfo:
            _reject_unknown_kwargs(
                {"seed": 7},
                engine="ESMFold",
                method="predict",
                constructor_example="ESMFold(chunk_size=64)",
            )
        assert "e.g. ESMFold(chunk_size=64)" in str(excinfo.value)

    def test_no_dangling_example_when_omitted(self) -> None:
        with pytest.raises(TypeError) as excinfo:
            _reject_unknown_kwargs({"seed": 7}, engine="E", method="predict")
        assert "e.g." not in str(excinfo.value)


class _KwargsEngine(FoldingEngine):
    """Engine that rejects per-call keywords, as the real wrappers do."""

    name = "Strict"

    def predict(self, sequence: str, **kwargs: object) -> Protein:
        _reject_unknown_kwargs(kwargs, engine="Strict", method="predict")
        return Protein(AtomArray(0), name=f"strict:{sequence}")


class TestPredictManyForwarding:
    """``predict_many`` forwards kwargs, so rejection reaches batch callers too."""

    def test_batch_without_kwargs_works(self) -> None:
        assert len(_KwargsEngine().predict_many(["AAA", "GGG"])) == 2

    def test_batch_with_unknown_kwarg_raises(self) -> None:
        with pytest.raises(TypeError, match="'seed'"):
            _KwargsEngine().predict_many(["AAA"], seed=7)
