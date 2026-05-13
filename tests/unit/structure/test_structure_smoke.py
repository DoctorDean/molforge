"""Smoke tests for molforge.structure — verify the subpackage imports cleanly."""

from __future__ import annotations


def test_import() -> None:
    import molforge.structure as mod

    assert mod is not None
