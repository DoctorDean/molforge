"""Smoke tests for molforge.docking — verify the subpackage imports cleanly."""

from __future__ import annotations


def test_import() -> None:
    import molforge.docking as mod

    assert mod is not None
