"""Smoke tests for biocore.sequence — verify the subpackage imports cleanly."""

from __future__ import annotations


def test_import() -> None:
    import biocore.sequence as mod

    assert mod is not None
