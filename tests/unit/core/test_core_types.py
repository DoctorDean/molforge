"""Tests for the core data-model types."""

from __future__ import annotations

from biocore.core import Atom, AtomArray, Chain, Protein, Residue


def test_protein_is_constructible() -> None:
    p = Protein(name="test")
    assert p.name == "test"
    assert p.chains == []


def test_chain_is_constructible() -> None:
    c = Chain(chain_id="A")
    assert c.chain_id == "A"


def test_residue_is_constructible() -> None:
    r = Residue(name="ALA", seq_id=1)
    assert r.name == "ALA"
    assert r.seq_id == 1


def test_atom_is_constructible() -> None:
    a = Atom(name="CA", element="C")
    assert a.name == "CA"
    assert a.element == "C"


def test_atom_array_is_constructible() -> None:
    aa = AtomArray()
    assert len(aa) == 0
