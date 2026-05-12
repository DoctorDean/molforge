"""Smoke tests for biocore.ml — verify the subpackage imports cleanly."""

from __future__ import annotations


def test_import() -> None:
    import biocore.ml as mod

    assert mod is not None
