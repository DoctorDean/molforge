"""Smoke tests for biocore.structure — verify the subpackage imports cleanly."""

from __future__ import annotations


def test_import() -> None:
    import biocore.structure as mod

    assert mod is not None
