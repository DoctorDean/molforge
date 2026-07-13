"""Tests for io.fetch_chembl / fetch_chembl_many.

Two boundaries are mocked: urllib.request.urlopen (the ChEMBL REST call) and,
for the success paths, the ``molforge.core._rdkit`` shim (so a Molecule can be
built from SMILES without RDKit). The genuine RDKit-absent path is checked
with only the network mocked.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from molforge.core import Molecule, RDKitNotInstalledError, _rdkit
from molforge.io import fetch_chembl, fetch_chembl_many


class _FakeMol:
    def __init__(self, smiles: str) -> None:
        self.smiles = smiles

    def GetNumAtoms(self) -> int:
        return len(self.smiles)

    def GetNumHeavyAtoms(self) -> int:
        return len(self.smiles)


ASPIRIN = json.dumps(
    {
        "molecule_chembl_id": "CHEMBL25",
        "pref_name": "ASPIRIN",
        "molecule_structures": {"canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O"},
    }
)


def _response(body: str) -> object:
    resp = MagicMock()
    resp.read.return_value = body.encode("utf-8")
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


@pytest.fixture
def fake_smiles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_rdkit, "mol_from_smiles", lambda s, **k: _FakeMol(s))


class TestFetchChembl:
    def test_builds_molecule(self, fake_smiles: None) -> None:
        with patch("urllib.request.urlopen", return_value=_response(ASPIRIN)) as mock:
            mol = fetch_chembl("CHEMBL25")
        assert isinstance(mol, Molecule)
        assert mol.name == "ASPIRIN"
        assert mol.metadata["source"] == "chembl"
        assert mol.metadata["chembl_id"] == "CHEMBL25"
        assert mock.call_args[0][0] == (
            "https://www.ebi.ac.uk/chembl/api/data/molecule/CHEMBL25.json"
        )

    def test_name_falls_back_to_id(self, fake_smiles: None) -> None:
        body = json.dumps(
            {
                "molecule_chembl_id": "CHEMBL999",
                "pref_name": None,
                "molecule_structures": {"canonical_smiles": "C"},
            }
        )
        with patch("urllib.request.urlopen", return_value=_response(body)):
            mol = fetch_chembl("CHEMBL999")
        assert mol.name == "CHEMBL999"

    def test_no_structure_raises(self) -> None:
        body = json.dumps(
            {"molecule_chembl_id": "CHEMBL1201580", "pref_name": "X", "molecule_structures": None}
        )
        with (
            patch("urllib.request.urlopen", return_value=_response(body)),
            pytest.raises(ValueError, match="no small-molecule structure"),
        ):
            fetch_chembl("CHEMBL1201580")

    def test_empty_id_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            fetch_chembl("  ")

    def test_http_error_becomes_oserror(self) -> None:
        import urllib.error

        err = urllib.error.HTTPError(url="u", code=404, msg="NF", hdrs=None, fp=None)  # type: ignore[arg-type]
        with (
            patch("urllib.request.urlopen", side_effect=err),
            pytest.raises(OSError, match="ChEMBL fetch failed"),
        ):
            fetch_chembl("CHEMBL0")

    def test_rdkit_absent_raises(self) -> None:
        # Network mocked, RDKit genuinely absent -> from_smiles raises.
        with (
            patch("urllib.request.urlopen", return_value=_response(ASPIRIN)),
            pytest.raises(RDKitNotInstalledError),
        ):
            fetch_chembl("CHEMBL25")


class TestFetchChemblMany:
    def test_fetches_all(self, fake_smiles: None) -> None:
        with patch("urllib.request.urlopen", return_value=_response(ASPIRIN)):
            mols = fetch_chembl_many(["CHEMBL25", "CHEMBL25"])
        assert len(mols) == 2
        assert all(isinstance(m, Molecule) for m in mols)

    def test_bad_on_error_raises(self) -> None:
        with pytest.raises(ValueError, match="on_error must be"):
            fetch_chembl_many(["CHEMBL25"], on_error="nope")

    def test_skip_drops_failures(self, fake_smiles: None) -> None:
        import urllib.error

        err = urllib.error.HTTPError(url="u", code=404, msg="NF", hdrs=None, fp=None)  # type: ignore[arg-type]
        with patch("urllib.request.urlopen", side_effect=[err, _response(ASPIRIN)]):
            mols = fetch_chembl_many(["BAD", "CHEMBL25"], on_error="skip")
        assert len(mols) == 1
        assert mols[0].name == "ASPIRIN"

    def test_skip_drops_structureless(self, fake_smiles: None) -> None:
        no_struct = json.dumps({"molecule_chembl_id": "CHEMBL_AB", "molecule_structures": None})
        with patch(
            "urllib.request.urlopen", side_effect=[_response(no_struct), _response(ASPIRIN)]
        ):
            mols = fetch_chembl_many(["CHEMBL_AB", "CHEMBL25"], on_error="skip")
        assert [m.name for m in mols] == ["ASPIRIN"]
