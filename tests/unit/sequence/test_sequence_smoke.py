"""Smoke tests for molforge.sequence — verify the subpackage imports cleanly."""

from __future__ import annotations


def test_import() -> None:
    import molforge.sequence as mod

    assert mod is not None
