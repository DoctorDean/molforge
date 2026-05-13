"""Smoke tests for molforge.io — verify the subpackage imports cleanly."""

from __future__ import annotations


def test_import() -> None:
    import molforge.io as mod

    assert mod is not None
