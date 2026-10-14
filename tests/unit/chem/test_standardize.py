"""Tests for :mod:`molforge.chem` standardization.

The default test environment has no RDKit, so the MolStandardize ops are
monkeypatched at the ``molforge.core._rdkit`` shim boundary. Each fake op
*tags* the mol, so the pipeline order is directly assertable; the genuine
not-installed path is also checked.

:class:`TestDesaltedMoleculeIsUsable` needs real RDKit (it is about what
RDKit hands back from a fragment deletion) and skips without it, mirroring
the chem-extra CI job.
"""

from __future__ import annotations

import importlib.util

import pytest

import molforge.chem as chem
from molforge.core import Molecule, RDKitNotInstalledError, _rdkit

_HAS_RDKIT = importlib.util.find_spec("rdkit") is not None


class _FakeMol:
    def __init__(self, tag: str) -> None:
        self.tag = tag

    def GetNumAtoms(self) -> int:
        return 3

    def GetNumHeavyAtoms(self) -> int:
        return 3


@pytest.fixture
def tagging_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each op appends its name to the mol's tag."""
    monkeypatch.setattr(_rdkit, "cleanup", lambda m: _FakeMol(m.tag + "/clean"))
    monkeypatch.setattr(_rdkit, "largest_fragment", lambda m: _FakeMol(m.tag + "/frag"))
    monkeypatch.setattr(_rdkit, "uncharge", lambda m: _FakeMol(m.tag + "/uncharge"))
    monkeypatch.setattr(_rdkit, "canonical_tautomer", lambda m: _FakeMol(m.tag + "/taut"))


def _base() -> Molecule:
    return Molecule.from_rdkit(_FakeMol("raw"), name="drugX", metadata={"source": "chembl"})


class TestStandardize:
    def test_default_pipeline(self, tagging_ops: None) -> None:
        out = chem.standardize(_base())
        assert out.to_rdkit().tag == "raw/clean/frag/uncharge"  # cleanup, desalt, neutralize
        assert out.metadata["standardized"] == ["cleanup", "largest_fragment", "neutralize"]

    def test_full_pipeline(self, tagging_ops: None) -> None:
        out = chem.standardize(_base(), tautomer=True)
        assert out.to_rdkit().tag == "raw/clean/frag/uncharge/taut"
        assert out.metadata["standardized"][-1] == "canonical_tautomer"

    def test_cleanup_only(self, tagging_ops: None) -> None:
        out = chem.standardize(_base(), desalt=False, neutralize=False)
        assert out.to_rdkit().tag == "raw/clean"
        assert out.metadata["standardized"] == ["cleanup"]

    def test_input_untouched_and_name_source_preserved(self, tagging_ops: None) -> None:
        base = _base()
        out = chem.standardize(base)
        assert base.to_rdkit().tag == "raw"  # original unmodified
        assert out.name == "drugX"
        assert out.metadata["source"] == "chembl"

    def test_metadata_not_aliased(self, tagging_ops: None) -> None:
        base = _base()
        chem.standardize(base)
        assert "standardized" not in base.metadata  # didn't mutate input's dict


class TestGranularOps:
    def test_each_op(self, tagging_ops: None) -> None:
        base = _base()
        assert chem.cleanup(base).to_rdkit().tag == "raw/clean"
        assert chem.largest_fragment(base).to_rdkit().tag == "raw/frag"
        assert chem.neutralize(base).to_rdkit().tag == "raw/uncharge"
        assert chem.canonical_tautomer(base).to_rdkit().tag == "raw/taut"

    def test_granular_records_step(self, tagging_ops: None) -> None:
        assert chem.neutralize(_base()).metadata["standardized"] == ["neutralize"]


@pytest.mark.skipif(_HAS_RDKIT, reason="asserts the RDKit-absent path")
class TestRDKitAbsent:
    def test_standardize_raises(self) -> None:
        with pytest.raises(RDKitNotInstalledError):
            chem.standardize(_base())

    def test_granular_raises(self) -> None:
        with pytest.raises(RDKitNotInstalledError):
            chem.largest_fragment(_base())


@pytest.mark.skipif(not _HAS_RDKIT, reason="requires RDKit")
class TestDesaltedMoleculeIsUsable:
    """Regression, real RDKit: a desalted molecule must be *usable*.

    ``FragmentParent`` builds the parent by deleting the other fragments'
    atoms and leaves the ring-info / valence caches uninitialized, so every
    ring-aware question about the result — descriptors, scaffolds — used to
    die on a bare RDKit ``RingInfo not initialized`` precondition violation.
    Only multi-fragment inputs hit it, which is exactly the input desalting
    is for.
    """

    _SALT = "CC(=O)Oc1ccccc1C(=O)[O-].[Na+]"

    def test_descriptors_after_largest_fragment(self) -> None:
        parent = chem.largest_fragment(Molecule.from_smiles(self._SALT))
        assert parent.smiles == "CC(=O)Oc1ccccc1C(=O)[O-]"
        assert parent.n_rotatable_bonds == 2
        assert parent.tpsa > 0

    def test_full_descriptor_set_after_standardize(self) -> None:
        cleaned = chem.standardize(Molecule.from_smiles(self._SALT))
        descriptors = chem.molecule_descriptors(cleaned)
        assert descriptors["n_heavy_atoms"] == 13  # aspirin, sodium gone
        assert descriptors["tpsa"] > 0

    def test_ring_info_available_on_the_result(self) -> None:
        cleaned = chem.standardize(Molecule.from_smiles(self._SALT))
        assert cleaned.to_rdkit().GetRingInfo().NumRings() == 1

    def test_single_fragment_input_unaffected(self) -> None:
        aspirin = Molecule.from_smiles("CC(=O)Oc1ccccc1C(=O)O")
        assert chem.standardize(aspirin).smiles == aspirin.smiles
