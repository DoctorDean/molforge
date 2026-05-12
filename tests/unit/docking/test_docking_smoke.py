"""Smoke tests for biocore.docking — verify the subpackage imports cleanly."""

from __future__ import annotations


def test_import() -> None:
    import biocore.docking as mod

    assert mod is not None
