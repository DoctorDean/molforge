"""Smoke tests for biocore.core — verify the subpackage imports cleanly."""

from __future__ import annotations


def test_import() -> None:
    import biocore.core as mod

    assert mod is not None
