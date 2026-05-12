"""Smoke tests for biocore.metrics — verify the subpackage imports cleanly."""

from __future__ import annotations


def test_import() -> None:
    import biocore.metrics as mod

    assert mod is not None
