"""Tests for :mod:`molforge.chem` Bemis-Murcko scaffold extraction.

The default test environment has no RDKit, so the two shim calls a scaffold
needs — decompose, then serialize — are monkeypatched at the
``molforge.core._rdkit`` boundary. The fakes carry the scaffold each molecule
*would* decompose to, which makes the ``generic`` routing, the metadata/name
contract, and the grouping logic assertable without RDKit.

Real chemistry (aspirin's benzene, the acyclic empty-scaffold convention, the
standardize → scaffold composition) is covered by :class:`TestRealChemistry`,
which runs wherever RDKit *is* installed and skips otherwise.
"""

from __future__ import annotations

import importlib.util

import pytest

import molforge.chem as chem
from molforge.chem import MoleculeDataset
from molforge.core import Molecule, RDKitNotInstalledError, _rdkit

_HAS_RDKIT = importlib.util.find_spec("rdkit") is not None

needs_rdkit = pytest.mark.skipif(not _HAS_RDKIT, reason="requires RDKit")
needs_no_rdkit = pytest.mark.skipif(_HAS_RDKIT, reason="asserts the RDKit-absent path")


class _FakeMol:
    """A mol that knows its own SMILES and what it decomposes to."""

    def __init__(self, smiles: str, *, scaffold: str = "", generic: str | None = None) -> None:
        self.smiles = smiles
        self.scaffold = scaffold
        self.generic = scaffold if generic is None else generic

    def GetNumAtoms(self) -> int:
        return len(self.smiles)

    def GetNumHeavyAtoms(self) -> int:
        return len(self.smiles)


def _mol(
    name: str,
    *,
    smiles: str = "CCO",
    scaffold: str = "",
    generic: str | None = None,
    metadata: dict[str, object] | None = None,
) -> Molecule:
    fake = _FakeMol(smiles, scaffold=scaffold, generic=generic)
    return Molecule.from_rdkit(fake, name=name, metadata=metadata)


@pytest.fixture
def murcko_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake decomposition + serialization, so scaffolds are pure bookkeeping."""

    def fake_scaffold(mol: _FakeMol, *, generic: bool = False) -> _FakeMol:
        return _FakeMol(mol.generic if generic else mol.scaffold)

    monkeypatch.setattr(_rdkit, "murcko_scaffold", fake_scaffold)
    monkeypatch.setattr(_rdkit, "to_smiles", lambda mol, **kwargs: mol.smiles)


def _library() -> list[Molecule]:
    """Three scaffolds across five molecules, one of them acyclic."""
    return [
        _mol("aspirin", smiles="CC(=O)Oc1ccccc1C(=O)O", scaffold="c1ccccc1", generic="C1CCCCC1"),
        _mol("paracetamol", smiles="CC(=O)Nc1ccc(O)cc1", scaffold="c1ccccc1", generic="C1CCCCC1"),
        _mol("nicotinamide", smiles="NC(=O)c1cccnc1", scaffold="c1ccncc1", generic="C1CCCCC1"),
        _mol("ethanol", smiles="CCO"),
        _mol("isobutanol", smiles="CC(C)CO"),
    ]


class TestMurckoScaffold:
    def test_returns_the_scaffold_as_a_molecule(self, murcko_ops: None) -> None:
        scaffold = chem.murcko_scaffold(_library()[0])
        assert isinstance(scaffold, Molecule)
        assert scaffold.smiles == "c1ccccc1"

    def test_generic_returns_the_framework(self, murcko_ops: None) -> None:
        assert chem.murcko_scaffold(_library()[0], generic=True).smiles == "C1CCCCC1"

    def test_name_preserved(self, murcko_ops: None) -> None:
        assert chem.murcko_scaffold(_library()[0]).name == "aspirin"

    def test_records_the_step(self, murcko_ops: None) -> None:
        assert chem.murcko_scaffold(_library()[0]).metadata["scaffold"] == "murcko"

    def test_generic_records_a_distinct_step(self, murcko_ops: None) -> None:
        out = chem.murcko_scaffold(_library()[0], generic=True)
        assert out.metadata["scaffold"] == "murcko_generic"

    def test_source_metadata_carried_across(self, murcko_ops: None) -> None:
        base = _mol("drugX", scaffold="c1ccccc1", metadata={"source": "chembl"})
        out = chem.murcko_scaffold(base)
        assert out.metadata["source"] == "chembl"
        assert out.metadata["scaffold"] == "murcko"

    def test_metadata_not_aliased(self, murcko_ops: None) -> None:
        base = _mol("drugX", scaffold="c1ccccc1")
        chem.murcko_scaffold(base)
        assert "scaffold" not in base.metadata  # didn't mutate the input's dict

    def test_input_untouched(self, murcko_ops: None) -> None:
        base = _library()[0]
        chem.murcko_scaffold(base)
        assert base.smiles == "CC(=O)Oc1ccccc1C(=O)O"

    def test_acyclic_gives_an_empty_scaffold(self, murcko_ops: None) -> None:
        """Bemis-Murcko's own convention: no rings, no scaffold."""
        assert chem.murcko_scaffold(_mol("ethanol")).smiles == ""

    def test_composes_with_standardize(self, murcko_ops: None, monkeypatch) -> None:
        """A standardized molecule can be scaffolded — metadata records both."""
        monkeypatch.setattr(_rdkit, "cleanup", lambda m: _FakeMol(m.smiles, scaffold=m.scaffold))
        cleaned = chem.cleanup(_library()[0])
        out = chem.murcko_scaffold(cleaned)
        assert out.metadata["standardized"] == ["cleanup"]
        assert out.metadata["scaffold"] == "murcko"


class TestScaffoldSmilesProperty:
    def test_reports_the_scaffold(self, murcko_ops: None) -> None:
        assert _library()[0].scaffold_smiles == "c1ccccc1"

    def test_empty_for_acyclic(self, murcko_ops: None) -> None:
        assert _mol("ethanol").scaffold_smiles == ""

    def test_delegates_the_underlying_mol(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, object] = {}

        def fake(mol: object, *, generic: bool = False) -> _FakeMol:
            seen["mol"] = mol
            seen["generic"] = generic
            return _FakeMol("c1ccccc1")

        monkeypatch.setattr(_rdkit, "murcko_scaffold", fake)
        monkeypatch.setattr(_rdkit, "to_smiles", lambda mol, **kwargs: mol.smiles)
        molecule = _library()[0]
        assert molecule.scaffold_smiles == "c1ccccc1"
        assert seen["mol"] is molecule.to_rdkit()
        assert seen["generic"] is False  # the property is the plain scaffold


class TestScaffoldDedup:
    def test_unique_keeps_one_per_scaffold(self, murcko_ops: None) -> None:
        kept = chem.unique(_library(), key="scaffold")
        assert [m.name for m in kept] == ["aspirin", "nicotinamide", "ethanol"]

    def test_acyclic_molecules_collapse_together(self, murcko_ops: None) -> None:
        """They all share the empty scaffold — worth knowing before using it."""
        acyclic = [_mol("ethanol"), _mol("isobutanol")]
        assert len(chem.unique(acyclic, key="scaffold")) == 1

    def test_dataset_dedup_by_scaffold(self, murcko_ops: None) -> None:
        out = MoleculeDataset(_library()).dedup(key="scaffold").collect()
        assert [m.name for m in out] == ["aspirin", "nicotinamide", "ethanol"]

    def test_dedup_by_scaffold_is_lazy(self, murcko_ops: None) -> None:
        consumed: list[str] = []

        def source():
            for molecule in _library():
                consumed.append(molecule.name)
                yield molecule

        dataset = MoleculeDataset(source()).dedup(key="scaffold")
        assert consumed == []  # nothing pulled yet
        dataset.collect()
        assert len(consumed) == 5

    def test_unknown_key_lists_scaffold(self) -> None:
        with pytest.raises(ValueError, match="'scaffold'"):
            chem.unique([_mol("x")], key="murcko")

    def test_dataset_unknown_key_raises_eagerly(self) -> None:
        with pytest.raises(ValueError, match="'scaffold'"):
            MoleculeDataset([_mol("x")]).dedup(key="nope")


class TestGroupByScaffold:
    def test_groups_by_scaffold(self, murcko_ops: None) -> None:
        groups = MoleculeDataset(_library()).group_by_scaffold()
        assert {k: [m.name for m in v] for k, v in groups.items()} == {
            "c1ccccc1": ["aspirin", "paracetamol"],
            "c1ccncc1": ["nicotinamide"],
            "": ["ethanol", "isobutanol"],
        }

    def test_keys_in_first_seen_order(self, murcko_ops: None) -> None:
        groups = MoleculeDataset(_library()).group_by_scaffold()
        assert list(groups) == ["c1ccccc1", "c1ccncc1", ""]

    def test_generic_merges_related_frameworks(self, murcko_ops: None) -> None:
        groups = MoleculeDataset(_library()).group_by_scaffold(generic=True)
        assert {k: [m.name for m in v] for k, v in groups.items()} == {
            "C1CCCCC1": ["aspirin", "paracetamol", "nicotinamide"],
            "": ["ethanol", "isobutanol"],
        }

    def test_empty_dataset(self, murcko_ops: None) -> None:
        assert MoleculeDataset([]).group_by_scaffold() == {}

    def test_composes_after_lazy_combinators(self, murcko_ops: None) -> None:
        groups = MoleculeDataset(_library()).take(2).group_by_scaffold()
        assert {k: len(v) for k, v in groups.items()} == {"c1ccccc1": 2}

    def test_consumes_a_single_pass_source(self, murcko_ops: None) -> None:
        dataset = MoleculeDataset(iter(_library()))
        assert len(dataset.group_by_scaffold()) == 3
        assert dataset.group_by_scaffold() == {}  # source exhausted, like collect()


@needs_no_rdkit
class TestRDKitAbsent:
    def test_murcko_scaffold_raises(self) -> None:
        with pytest.raises(RDKitNotInstalledError):
            chem.murcko_scaffold(_mol("x"))

    def test_scaffold_smiles_raises(self) -> None:
        with pytest.raises(RDKitNotInstalledError):
            _ = _mol("x").scaffold_smiles

    def test_group_by_scaffold_raises_on_consume(self) -> None:
        with pytest.raises(RDKitNotInstalledError):
            MoleculeDataset([_mol("x")]).group_by_scaffold()


@needs_rdkit
class TestRealChemistry:
    """The genuine RDKit path — runs in the chem-extra CI job."""

    def test_aspirin_scaffold_is_benzene(self) -> None:
        aspirin = Molecule.from_smiles("CC(=O)OC1=CC=CC=C1C(=O)O", name="aspirin")
        assert chem.murcko_scaffold(aspirin).smiles == "c1ccccc1"
        assert aspirin.scaffold_smiles == "c1ccccc1"

    def test_side_chains_stripped_not_rings(self) -> None:
        caffeine = Molecule.from_smiles("CN1C=NC2=C1C(=O)N(C)C(=O)N2C")
        scaffold = chem.murcko_scaffold(caffeine)
        assert scaffold.n_atoms == 11  # the fused bicycle, N-methyls gone
        assert "c" in scaffold.smiles or "C" in scaffold.smiles

    def test_acyclic_molecule_has_an_empty_scaffold(self) -> None:
        ethanol = Molecule.from_smiles("CCO")
        scaffold = chem.murcko_scaffold(ethanol)
        assert scaffold.n_atoms == 0
        assert scaffold.smiles == ""
        assert ethanol.scaffold_smiles == ""

    def test_generic_collapses_heteroatoms(self) -> None:
        benzene_like = Molecule.from_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
        pyridine_like = Molecule.from_smiles("NC(=O)c1cccnc1")
        assert benzene_like.scaffold_smiles != pyridine_like.scaffold_smiles
        assert (
            chem.murcko_scaffold(benzene_like, generic=True).smiles
            == chem.murcko_scaffold(pyridine_like, generic=True).smiles
        )

    def test_input_molecule_untouched(self) -> None:
        aspirin = Molecule.from_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
        before = aspirin.smiles
        chem.murcko_scaffold(aspirin)
        assert aspirin.smiles == before

    def test_scaffold_of_a_standardized_salt(self) -> None:
        """Regression: the desalted molecule must still be usable, so the
        documented standardize → scaffold pipeline composes."""
        salt = Molecule.from_smiles("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]", name="aspirin-Na")
        cleaned = chem.standardize(salt)
        assert chem.murcko_scaffold(cleaned).smiles == "c1ccccc1"
        assert cleaned.scaffold_smiles == "c1ccccc1"

    def test_group_a_small_library(self) -> None:
        library = [
            Molecule.from_smiles("CC(=O)OC1=CC=CC=C1C(=O)O", name="aspirin"),
            Molecule.from_smiles("CC(=O)Nc1ccc(O)cc1", name="paracetamol"),
            Molecule.from_smiles("CCO", name="ethanol"),
        ]
        groups = MoleculeDataset(library).group_by_scaffold()
        assert [m.name for m in groups["c1ccccc1"]] == ["aspirin", "paracetamol"]
        assert [m.name for m in groups[""]] == ["ethanol"]

    def test_dedup_by_scaffold_on_real_molecules(self) -> None:
        library = [
            Molecule.from_smiles("CC(=O)OC1=CC=CC=C1C(=O)O", name="aspirin"),
            Molecule.from_smiles("CC(=O)Nc1ccc(O)cc1", name="paracetamol"),
            Molecule.from_smiles("NC(=O)c1cccnc1", name="nicotinamide"),
        ]
        kept = chem.unique(library, key="scaffold")
        assert [m.name for m in kept] == ["aspirin", "nicotinamide"]
