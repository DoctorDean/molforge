"""Smoke tests for biocore.plugins — verify the subpackage imports cleanly."""

from __future__ import annotations


def test_import() -> None:
    import biocore.plugins as mod

    assert mod is not None
