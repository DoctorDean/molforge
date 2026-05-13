"""Smoke tests for molforge.core — verify the subpackage imports cleanly."""

from __future__ import annotations


def test_import() -> None:
    import molforge.core as mod

    assert mod is not None
    assert hasattr(mod, "Protein")
    assert hasattr(mod, "AtomArray")


def test_top_level_exports_present() -> None:
    from molforge.core import (
        ATOM_FIELDS,
        Atom,
        AtomArray,
        Chain,
        Protein,
        Residue,
        three_to_one,
    )

    assert all(c is not None for c in [Atom, AtomArray, Chain, Protein, Residue])
    assert callable(three_to_one)
    assert "coords" in ATOM_FIELDS
