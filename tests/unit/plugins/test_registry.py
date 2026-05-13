"""Tests for the plugin registry."""

from __future__ import annotations

from molforge.plugins import available, get, register_engine


def test_register_and_retrieve_engine() -> None:
    def factory() -> str:
        return "ok"

    register_engine("test_engine", factory)
    assert "test_engine" in available("engine")
    assert get("engine", "test_engine")() == "ok"
