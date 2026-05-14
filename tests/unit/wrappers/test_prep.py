"""Tests for receptor / ligand preparation via meeko + RDKit.

These tests don't require meeko or rdkit to be installed. They verify
the wiring: passthrough for already-prepared PDBQT, clean error
messages when the heavy deps are missing, and helper behavior.

End-to-end prep (real meeko execution) is left to integration tests
marked @pytest.mark.slow.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from molforge.core import AtomArray, Protein
from molforge.docking import DockingEngineNotInstalledError
from molforge.wrappers.docking import is_pdbqt_path, prepare_ligand, prepare_receptor


def _meeko_available() -> bool:
    return importlib.util.find_spec("meeko") is not None


def _rdkit_available() -> bool:
    return importlib.util.find_spec("rdkit") is not None


class TestIsPdbqtPath:
    def test_pdbqt_string(self) -> None:
        assert is_pdbqt_path("ligand.pdbqt") is True

    def test_pdbqt_path_object(self) -> None:
        assert is_pdbqt_path(Path("rec.pdbqt")) is True

    def test_non_pdbqt_string(self) -> None:
        assert is_pdbqt_path("ligand.sdf") is False
        assert is_pdbqt_path("rec.pdb") is False

    def test_protein_instance_is_not_path(self) -> None:
        assert is_pdbqt_path(Protein(AtomArray(0))) is False

    def test_non_path_value_handled_gracefully(self) -> None:
        assert is_pdbqt_path(123) is False
        assert is_pdbqt_path(None) is False


class TestReceptorPrepPassthrough:
    """Already-prepared PDBQT receptors should pass through without meeko."""

    def test_pdbqt_input_just_copies(self, tmp_path: Path) -> None:
        src = tmp_path / "rec_in.pdbqt"
        src.write_text("REMARK fake-prepared receptor\n")
        out = tmp_path / "rec_out.pdbqt"
        result = prepare_receptor(src, out)
        assert result == out
        assert out.read_text() == "REMARK fake-prepared receptor\n"


class TestReceptorPrepMissingDeps:
    @pytest.mark.skipif(_meeko_available(), reason="meeko is installed")
    def test_pdb_input_without_meeko_raises(self, tmp_path: Path) -> None:
        src = tmp_path / "rec.pdb"
        src.write_text("HEADER fake\n")
        out = tmp_path / "rec.pdbqt"
        with pytest.raises(DockingEngineNotInstalledError, match="meeko"):
            prepare_receptor(src, out)

    @pytest.mark.skipif(_meeko_available(), reason="meeko is installed")
    def test_protein_input_without_meeko_raises(self, tmp_path: Path) -> None:
        p = Protein(AtomArray(0))
        out = tmp_path / "rec.pdbqt"
        with pytest.raises(DockingEngineNotInstalledError, match="meeko"):
            prepare_receptor(p, out)


class TestLigandPrepPassthrough:
    def test_pdbqt_input_just_copies(self, tmp_path: Path) -> None:
        src = tmp_path / "lig_in.pdbqt"
        src.write_text("REMARK fake-prepared ligand\nATOM      1  C   LIG\n")
        out = tmp_path / "lig_out.pdbqt"
        result = prepare_ligand(src, out)
        assert result == out
        assert out.read_text() == "REMARK fake-prepared ligand\nATOM      1  C   LIG\n"


class TestLigandPrepMissingDeps:
    @pytest.mark.skipif(_rdkit_available(), reason="rdkit is installed")
    def test_sdf_without_rdkit_raises(self, tmp_path: Path) -> None:
        src = tmp_path / "lig.sdf"
        src.write_text("dummy\n")
        out = tmp_path / "lig.pdbqt"
        with pytest.raises(DockingEngineNotInstalledError, match="RDKit"):
            prepare_ligand(src, out)

    @pytest.mark.skipif(_rdkit_available(), reason="rdkit is installed")
    def test_smiles_without_rdkit_raises(self, tmp_path: Path) -> None:
        out = tmp_path / "lig.pdbqt"
        with pytest.raises(DockingEngineNotInstalledError, match="RDKit"):
            prepare_ligand("CCO", out, from_smiles=True)


class TestLigandPrepErrors:
    @pytest.mark.skipif(not _rdkit_available(), reason="rdkit not installed; skip")
    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        # Even with rdkit installed, a .xyz file should be rejected with
        # a clear error rather than crashing.
        src = tmp_path / "lig.xyz"
        src.write_text("dummy\n")
        out = tmp_path / "lig.pdbqt"
        with pytest.raises(ValueError, match="unsupported ligand file extension"):
            prepare_ligand(src, out)


@pytest.mark.slow
@pytest.mark.skipif(
    not (_meeko_available() and _rdkit_available()), reason="meeko/rdkit not installed"
)
class TestEndToEnd:
    """Real prep runs against meeko + RDKit. Skipped unless deps installed."""

    def test_prepare_ligand_from_smiles(self, tmp_path: Path) -> None:
        # ethanol — trivial test molecule
        out = tmp_path / "ethanol.pdbqt"
        result = prepare_ligand("CCO", out, from_smiles=True)
        assert result.exists()
        text = result.read_text()
        # PDBQT should at minimum contain ATOM records
        assert "ATOM" in text or "HETATM" in text
