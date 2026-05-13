"""Tests for the FoldingEngine abstract base class and helpers."""

from __future__ import annotations

import pytest

from molforge.core import AtomArray, Protein
from molforge.wrappers.folding._base import (
    FoldingEngine,
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
