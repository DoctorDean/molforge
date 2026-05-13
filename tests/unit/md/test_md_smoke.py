"""Smoke tests for molforge.md — verify the subpackage imports cleanly."""

from __future__ import annotations


def test_import() -> None:
    import molforge.md as mod

    assert mod is not None
