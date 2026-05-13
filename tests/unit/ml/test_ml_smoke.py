"""Smoke tests for molforge.ml — verify the subpackage imports cleanly."""

from __future__ import annotations


def test_import() -> None:
    import molforge.ml as mod

    assert mod is not None
