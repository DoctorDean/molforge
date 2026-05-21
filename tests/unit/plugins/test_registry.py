"""Tests for the plugin registry."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from molforge.plugins import (
    available,
    clear,
    discover,
    get,
    register_engine,
    register_parser,
    register_scorer,
)


@pytest.fixture(autouse=True)
def _isolate_registry():  # type: ignore[no-untyped-def]
    """Ensure each test starts with a clean registry."""
    clear()
    yield
    clear()


class TestBasicRegistration:
    def test_register_and_retrieve_engine(self) -> None:
        def factory() -> str:
            return "ok"

        register_engine("test_engine", factory)
        assert "test_engine" in available("engine")
        assert get("engine", "test_engine")() == "ok"

    def test_register_parser(self) -> None:
        register_parser("foo", lambda path: "parsed")
        assert "foo" in available("parser")
        assert get("parser", "foo")("anything") == "parsed"

    def test_register_scorer(self) -> None:
        register_scorer("bar", lambda x: 42.0)
        assert "bar" in available("scorer")
        assert get("scorer", "bar")(None) == 42.0

    def test_get_missing_raises_keyerror(self) -> None:
        with pytest.raises(KeyError):
            get("engine", "does_not_exist")

    def test_available_filters_by_kind(self) -> None:
        register_engine("e1", lambda: None)
        register_parser("p1", lambda x: None)
        register_scorer("s1", lambda x: None)
        assert sorted(available("engine")) == ["e1"]
        assert sorted(available("parser")) == ["p1"]
        assert sorted(available("scorer")) == ["s1"]

    def test_available_no_filter_returns_all(self) -> None:
        register_engine("e", lambda: None)
        register_parser("p", lambda x: None)
        names = available()
        assert sorted(names) == ["e", "p"]

    def test_clear_empties_registry(self) -> None:
        register_engine("e", lambda: None)
        register_parser("p", lambda x: None)
        clear()
        assert available() == []


def _fake_entry_point(name: str, register_fn: Any) -> Any:
    """Build a stand-in for an importlib.metadata.EntryPoint."""
    ep = MagicMock()
    ep.name = name
    ep.load.return_value = register_fn
    return ep


class TestDiscover:
    def test_loads_entry_point(self) -> None:
        def my_register() -> None:
            register_engine("from_plugin", lambda: "registered_value")

        ep = _fake_entry_point("my_plugin", my_register)
        with patch(
            "molforge.plugins.registry.metadata.entry_points",
            return_value=[ep],
        ):
            loaded = discover()
        assert loaded == ["my_plugin"]
        assert "from_plugin" in available("engine")
        assert get("engine", "from_plugin")() == "registered_value"

    def test_loads_multiple_entry_points(self) -> None:
        register_calls = []

        def reg_a() -> None:
            register_calls.append("a")
            register_engine("eng_a", lambda: None)

        def reg_b() -> None:
            register_calls.append("b")
            register_parser("par_b", lambda x: None)

        eps = [
            _fake_entry_point("plug_a", reg_a),
            _fake_entry_point("plug_b", reg_b),
        ]
        with patch(
            "molforge.plugins.registry.metadata.entry_points",
            return_value=eps,
        ):
            loaded = discover()
        assert set(loaded) == {"plug_a", "plug_b"}
        assert "eng_a" in available("engine")
        assert "par_b" in available("parser")
        # Both registration functions were actually called
        assert sorted(register_calls) == ["a", "b"]

    def test_broken_plugin_suppressed_and_others_still_load(self) -> None:
        """A broken plugin (raises on load or during register) shouldn't
        prevent other plugins from loading."""

        def good_register() -> None:
            register_engine("good", lambda: "ok")

        broken_ep = MagicMock()
        broken_ep.name = "broken"
        broken_ep.load.side_effect = ImportError("module missing")

        register_raises_ep = _fake_entry_point(
            "register_raises",
            MagicMock(side_effect=RuntimeError("boom")),
        )

        good_ep = _fake_entry_point("good_plugin", good_register)

        with patch(
            "molforge.plugins.registry.metadata.entry_points",
            return_value=[broken_ep, register_raises_ep, good_ep],
        ):
            loaded = discover()

        # Only the good one ends up loaded
        assert loaded == ["good_plugin"]
        assert "good" in available("engine")

    def test_no_entry_points_returns_empty_list(self) -> None:
        with patch(
            "molforge.plugins.registry.metadata.entry_points",
            return_value=[],
        ):
            loaded = discover()
        assert loaded == []
        assert available() == []
