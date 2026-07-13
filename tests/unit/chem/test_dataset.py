"""Tests for :class:`molforge.chem.MoleculeDataset`.

The core combinators (map/take/collect) are chemistry-agnostic — they just
move molecules through — so no RDKit or shim mocking is needed. A tiny fake
mol wrapped in a real :class:`Molecule` is enough to assert laziness,
short-circuiting, and the re-iterable-iff-source-is contract.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import count

import pytest

from molforge.chem import MoleculeDataset
from molforge.core import Molecule, RDKitNotInstalledError, _rdkit
from molforge.validation import Criterion


class _FakeMol:
    def __init__(self, tag: str, *, heavy: int = 1) -> None:
        self.tag = tag
        self._heavy = heavy

    def GetNumAtoms(self) -> int:
        return self._heavy

    def GetNumHeavyAtoms(self) -> int:
        return self._heavy


def _mol(tag: str, *, heavy: int = 1) -> Molecule:
    return Molecule.from_rdkit(_FakeMol(tag, heavy=heavy), name=tag)


def _relabel(suffix: str) -> Callable[[Molecule], Molecule]:
    def fn(m: Molecule) -> Molecule:
        return Molecule.from_rdkit(m.to_rdkit(), name=m.name + suffix)

    return fn


class TestCoreIteration:
    def test_iter_and_collect(self) -> None:
        ds = MoleculeDataset([_mol("a"), _mol("b")])
        assert [m.name for m in ds] == ["a", "b"]
        assert [m.name for m in ds.collect()] == ["a", "b"]

    def test_collect_returns_list(self) -> None:
        assert isinstance(MoleculeDataset([_mol("a")]).collect(), list)

    def test_repr_is_lazy(self) -> None:
        assert repr(MoleculeDataset([])) == "MoleculeDataset(<lazy>)"


class TestMap:
    def test_map_transforms(self) -> None:
        out = MoleculeDataset([_mol("a"), _mol("b")]).map(_relabel("!")).collect()
        assert [m.name for m in out] == ["a!", "b!"]

    def test_map_is_lazy(self) -> None:
        seen: list[str] = []

        def fn(m: Molecule) -> Molecule:
            seen.append(m.name)
            return m

        ds = MoleculeDataset([_mol("a"), _mol("b")]).map(fn)
        assert seen == []  # nothing applied until iterated
        ds.collect()
        assert seen == ["a", "b"]

    def test_chained_maps(self) -> None:
        out = MoleculeDataset([_mol("x")]).map(_relabel("-1")).map(_relabel("-2")).collect()
        assert out[0].name == "x-1-2"


class TestTake:
    def test_take_limits(self) -> None:
        out = MoleculeDataset([_mol(str(i)) for i in range(5)]).take(2).collect()
        assert [m.name for m in out] == ["0", "1"]

    def test_take_short_circuits_infinite(self) -> None:
        infinite = (_mol(str(i)) for i in count())
        out = MoleculeDataset(infinite).take(3).collect()
        assert [m.name for m in out] == ["0", "1", "2"]

    def test_take_more_than_available(self) -> None:
        out = MoleculeDataset([_mol("a")]).take(10).collect()
        assert len(out) == 1

    def test_take_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="n >= 0"):
            MoleculeDataset([]).take(-1)

    def test_map_then_take_is_lazy(self) -> None:
        calls: list[str] = []

        def fn(m: Molecule) -> Molecule:
            calls.append(m.name)
            return m

        infinite = (_mol(str(i)) for i in count())
        out = MoleculeDataset(infinite).map(fn).take(2).collect()
        assert [m.name for m in out] == ["0", "1"]
        assert calls == ["0", "1"]  # map ran only for the taken items


class TestReiterability:
    def test_reiterable_over_list(self) -> None:
        ds = MoleculeDataset([_mol("a"), _mol("b")]).map(_relabel("!"))
        assert len(ds.collect()) == 2
        assert len(ds.collect()) == 2  # list source -> repeatable

    def test_single_pass_over_iterator(self) -> None:
        ds = MoleculeDataset(iter([_mol("a"), _mol("b")])).map(_relabel("!"))
        assert len(ds.collect()) == 2
        assert len(ds.collect()) == 0  # one-shot source -> exhausted


class TestValid:
    def test_valid_drops_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_rdkit, "sanitize_ok", lambda m: m.tag != "bad")
        out = MoleculeDataset([_mol("a"), _mol("bad"), _mol("c")]).valid().collect()
        assert [m.name for m in out] == ["a", "c"]

    def test_valid_is_lazy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        checked: list[str] = []

        def ok(m: _FakeMol) -> bool:
            checked.append(m.tag)
            return True

        monkeypatch.setattr(_rdkit, "sanitize_ok", ok)
        ds = MoleculeDataset([_mol("a"), _mol("b")]).valid()
        assert checked == []
        ds.collect()
        assert checked == ["a", "b"]


class TestDedup:
    @pytest.fixture
    def by_inchikey(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_rdkit, "to_inchikey", lambda m: f"KEY-{m.tag}")

    @pytest.fixture
    def by_smiles(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_rdkit, "to_smiles", lambda m: f"SMI-{m.tag}")

    def test_dedup_keeps_first(self, by_inchikey: None) -> None:
        out = MoleculeDataset([_mol("a"), _mol("b"), _mol("a")]).dedup().collect()
        assert [m.name for m in out] == ["a", "b"]

    def test_dedup_streams_lazily(self, by_inchikey: None) -> None:
        # dedup over an unbounded source composed with take() must not hang.
        endless = (_mol("x") for _ in count())
        out = MoleculeDataset(endless).dedup().take(1).collect()
        assert [m.name for m in out] == ["x"]

    def test_dedup_key_smiles(self, by_smiles: None) -> None:
        out = MoleculeDataset([_mol("a"), _mol("a"), _mol("d")]).dedup(key="smiles").collect()
        assert [m.name for m in out] == ["a", "d"]

    def test_dedup_bad_key_raises_eagerly(self) -> None:
        with pytest.raises(ValueError, match="inchikey"):
            MoleculeDataset([_mol("a")]).dedup(key="nope")


class TestFilter:
    def test_filter_by_heavy_atoms(self) -> None:
        ds = MoleculeDataset([_mol("small", heavy=5), _mol("big", heavy=40)])
        out = ds.filter(Criterion.le("n_heavy_atoms", 20)).collect()
        assert [m.name for m in out] == ["small"]

    def test_filter_composed_criterion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_rdkit, "formal_charge", lambda m: 0 if m.tag == "ok" else 2)
        ds = MoleculeDataset([_mol("ok", heavy=10), _mol("charged", heavy=10)])
        crit = Criterion.le("n_heavy_atoms", 20) & Criterion.le("formal_charge", 0)
        assert [m.name for m in ds.filter(crit).collect()] == ["ok"]

    def test_filter_then_take_short_circuits(self) -> None:
        endless = (_mol(str(i), heavy=i) for i in count())
        out = MoleculeDataset(endless).filter(Criterion.ge("n_heavy_atoms", 0)).take(3).collect()
        assert len(out) == 3

    def test_filter_unknown_descriptor_raises_eagerly(self) -> None:
        with pytest.raises(ValueError, match="unknown descriptor"):
            MoleculeDataset([]).filter(Criterion.lt("logp", 5))

    def test_filter_rdkit_absent_on_consume(self) -> None:
        ds = MoleculeDataset([_mol("a")])
        with pytest.raises(RDKitNotInstalledError):
            ds.filter(Criterion.lt("molecular_weight", 500)).collect()


class TestMoleculeAwareRDKitAbsent:
    def test_valid_raises_on_consume(self) -> None:
        with pytest.raises(RDKitNotInstalledError):
            MoleculeDataset([_mol("a")]).valid().collect()

    def test_dedup_raises_on_consume(self) -> None:
        with pytest.raises(RDKitNotInstalledError):
            MoleculeDataset([_mol("a"), _mol("b")]).dedup().collect()
