"""Tests for :mod:`molforge.chem` validity and deduplication.

RDKit is absent here, so validity and identity are monkeypatched at the
``molforge.core._rdkit`` shim boundary using a tiny fake mol. The genuine
not-installed paths (which must raise, consistent with the rest of the
package) are checked too.
"""

from __future__ import annotations

import pytest

import molforge.chem as chem
from molforge.core import Molecule, RDKitNotInstalledError
from molforge.core import _rdkit


class _FakeMol:
    """Minimal stand-in exposing only what Molecule touches."""

    def __init__(self, ident: str = "x", *, ok: bool = True) -> None:
        self.ident = ident
        self.ok = ok

    def GetNumAtoms(self) -> int:
        return 3

    def GetNumHeavyAtoms(self) -> int:
        return 3


def _mol(ident: str = "x", *, ok: bool = True, name: str = "") -> Molecule:
    return Molecule.from_rdkit(_FakeMol(ident, ok=ok), name=name)


class TestIsValid:
    def test_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_rdkit, "sanitize_ok", lambda m: m.ok)
        assert chem.is_valid(_mol(ok=True)) is True

    def test_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_rdkit, "sanitize_ok", lambda m: m.ok)
        assert chem.is_valid(_mol(ok=False)) is False

    def test_delegates_underlying_mol(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, object] = {}

        def fake(m: object) -> bool:
            seen["mol"] = m
            return True

        monkeypatch.setattr(_rdkit, "sanitize_ok", fake)
        molecule = _mol()
        chem.is_valid(molecule)
        assert seen["mol"] is molecule.to_rdkit()  # the shared mol, not a copy


class TestUnique:
    @pytest.fixture
    def by_inchikey(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_rdkit, "to_inchikey", lambda m: f"KEY-{m.ident}")

    @pytest.fixture
    def by_smiles(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_rdkit, "to_smiles", lambda m: f"SMI-{m.ident}")

    def test_dedup_keeps_first(self, by_inchikey: None) -> None:
        out = chem.unique([_mol("a", name="a-first"), _mol("b", name="b"), _mol("a", name="a-dup")])
        assert [m.name for m in out] == ["a-first", "b"]  # duplicate 'a' dropped

    def test_order_preserved(self, by_inchikey: None) -> None:
        out = chem.unique([_mol("c", name="c"), _mol("a", name="a"), _mol("b", name="b")])
        assert [m.name for m in out] == ["c", "a", "b"]

    def test_key_smiles(self, by_smiles: None) -> None:
        out = chem.unique(
            [_mol("a", name="a1"), _mol("a", name="a2"), _mol("d", name="d")],
            key="smiles",
        )
        assert [m.name for m in out] == ["a1", "d"]

    def test_empty(self, by_inchikey: None) -> None:
        assert chem.unique([]) == []

    def test_bad_key_rejected(self) -> None:
        # Validated before any molecule is touched, so no RDKit is needed.
        with pytest.raises(ValueError, match="inchikey"):
            chem.unique([_mol()], key="formula")


class TestRDKitAbsent:
    def test_is_valid_raises(self) -> None:
        with pytest.raises(RDKitNotInstalledError):
            chem.is_valid(_mol())

    def test_unique_raises(self) -> None:
        with pytest.raises(RDKitNotInstalledError):
            chem.unique([_mol("a"), _mol("b")])
