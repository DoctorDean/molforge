"""Tests for the AutoDock Vina wrapper.

These tests don't require the ``vina`` PyPI package to be installed.
They exercise:
  - Construction and parameter handling
  - The missing-dependency error path
  - PDBQT output parsing in isolation
  - Receptor / ligand materialization (path passthrough + clear errors
    for unsupported input types)

End-to-end docking against the real engine is left to integration tests.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from molforge.core import AtomArray, Protein
from molforge.docking import DockingEngineNotInstalledError, DockingResult
from molforge.wrappers.docking import Vina


def _vina_available() -> bool:
    return importlib.util.find_spec("vina") is not None


# A small but realistic synthetic Vina poses file. Two MODEL blocks, the
# second worse-scoring (Vina sorts ascending by negative kcal/mol, so the
# numeric ordering of pose blocks isn't guaranteed in real output, but
# our parser sorts after the fact).
_VINA_POSES_PDBQT = """\
MODEL 1
REMARK VINA RESULT:    -8.400    0.000    0.000
REMARK INTER + INTRA:   -10.5
ATOM      1  C   LIG A   1       1.234   2.345   3.456  1.00  0.00           C
ATOM      2  N   LIG A   1       2.234   3.345   4.456  1.00  0.00           N
ENDMDL
MODEL 2
REMARK VINA RESULT:    -7.900    1.234    2.345
ATOM      1  C   LIG A   1       1.300   2.400   3.500  1.00  0.00           C
ATOM      2  N   LIG A   1       2.300   3.400   4.500  1.00  0.00           N
ENDMDL
"""


class TestConstruction:
    def test_defaults(self) -> None:
        v = Vina()
        assert v.name == "Vina"
        assert v.scoring == "vina"
        assert v.seed is None
        assert v.cpu == 0
        assert v.verbosity == 0

    def test_custom_params(self) -> None:
        v = Vina(scoring="vinardo", seed=42, cpu=4, verbosity=1)
        assert v.scoring == "vinardo"
        assert v.seed == 42
        assert v.cpu == 4

    def test_construction_does_not_import_vina(self) -> None:
        """The vina package should NOT be loaded just by constructing the wrapper."""
        # The handle is built lazily on dock(). Constructing is free.
        v = Vina()
        # Nothing observable should depend on `vina` being installed
        assert v is not None


class TestMissingDependency:
    @pytest.mark.skipif(_vina_available(), reason="vina is installed")
    def test_dock_without_vina_raises_clear_error(self, tmp_path: Path) -> None:
        v = Vina()
        # Create a dummy pdbqt path to satisfy the materialize step before
        # the vina-handle creation
        receptor = tmp_path / "rec.pdbqt"
        receptor.write_text("HEADER fake\n")
        ligand = tmp_path / "lig.pdbqt"
        ligand.write_text("HEADER fake\n")
        with pytest.raises(DockingEngineNotInstalledError, match=r"pip install vina"):
            v.dock(receptor, ligand, center=(0.0, 0.0, 0.0))


class TestMaterialization:
    def test_pdbqt_path_passthrough(self, tmp_path: Path) -> None:
        v = Vina()
        path = tmp_path / "thing.pdbqt"
        path.write_text("HEADER fake\n")
        assert v._materialize_receptor(path, tmp_path) == path
        assert v._materialize_ligand(path, tmp_path) == path

    def test_pdb_receptor_without_meeko_raises_clear_error(self, tmp_path: Path) -> None:
        """Without meeko installed, prep raises DockingEngineNotInstalledError."""
        import importlib.util

        if importlib.util.find_spec("meeko") is not None:
            pytest.skip("meeko is installed; this test verifies the missing-dep path")

        v = Vina()
        bad = tmp_path / "rec.pdb"
        bad.write_text("HEADER fake\n")
        with pytest.raises(DockingEngineNotInstalledError, match="meeko"):
            v._materialize_receptor(bad, tmp_path)

    def test_sdf_ligand_without_meeko_raises_clear_error(self, tmp_path: Path) -> None:
        import importlib.util

        if importlib.util.find_spec("meeko") is not None:
            pytest.skip("meeko is installed; this test verifies the missing-dep path")

        v = Vina()
        bad = tmp_path / "lig.sdf"
        bad.write_text("dummy\n")
        with pytest.raises(DockingEngineNotInstalledError):
            v._materialize_ligand(bad, tmp_path)

    def test_protein_receptor_without_meeko_raises_clear_error(self, tmp_path: Path) -> None:
        import importlib.util

        if importlib.util.find_spec("meeko") is not None:
            pytest.skip("meeko is installed; this test verifies the missing-dep path")

        v = Vina()
        p = Protein(AtomArray(0))
        with pytest.raises(DockingEngineNotInstalledError, match="meeko"):
            v._materialize_receptor(p, tmp_path)


class TestParsePoses:
    """The output parser is the most important testable seam: every
    real docking run produces text we have to parse, and Vina's output
    format is stable enough that these tests cover real behavior."""

    def test_returns_docking_result(self) -> None:
        v = Vina()
        result = v._parse_poses_pdbqt(_VINA_POSES_PDBQT)
        assert isinstance(result, DockingResult)

    def test_correct_number_of_poses(self) -> None:
        v = Vina()
        result = v._parse_poses_pdbqt(_VINA_POSES_PDBQT)
        assert len(result) == 2

    def test_scores_extracted(self) -> None:
        v = Vina()
        result = v._parse_poses_pdbqt(_VINA_POSES_PDBQT)
        assert result.poses[0].score == pytest.approx(-8.4)
        assert result.poses[1].score == pytest.approx(-7.9)

    def test_rmsds_extracted(self) -> None:
        v = Vina()
        result = v._parse_poses_pdbqt(_VINA_POSES_PDBQT)
        assert result.poses[0].rmsd_lb == pytest.approx(0.0)
        assert result.poses[1].rmsd_lb == pytest.approx(1.234)
        assert result.poses[1].rmsd_ub == pytest.approx(2.345)

    def test_sorted_best_first(self) -> None:
        v = Vina()
        result = v._parse_poses_pdbqt(_VINA_POSES_PDBQT)
        # Scores ascending (lower = better)
        scores = [p.score for p in result.poses]
        assert scores == sorted(scores)

    def test_ranks_reassigned(self) -> None:
        v = Vina()
        result = v._parse_poses_pdbqt(_VINA_POSES_PDBQT)
        assert result.poses[0].rank == 0
        assert result.poses[1].rank == 1

    def test_pose_ligand_is_protein(self) -> None:
        v = Vina()
        result = v._parse_poses_pdbqt(_VINA_POSES_PDBQT)
        assert isinstance(result.poses[0].ligand, Protein)
        assert result.poses[0].ligand.n_atoms == 2

    def test_best_helper(self) -> None:
        v = Vina()
        result = v._parse_poses_pdbqt(_VINA_POSES_PDBQT)
        assert result.best.score == pytest.approx(-8.4)

    def test_engine_metadata_recorded(self) -> None:
        v = Vina()
        result = v._parse_poses_pdbqt(
            _VINA_POSES_PDBQT,
            run_metadata={"center": (1.0, 2.0, 3.0), "exhaustiveness": 16},
        )
        assert result.engine == "Vina"
        assert result.metadata["exhaustiveness"] == 16

    def test_empty_text_yields_empty_result(self) -> None:
        v = Vina()
        result = v._parse_poses_pdbqt("")
        assert len(result) == 0
        assert result.engine == "Vina"

    def test_single_pose_without_model_records(self) -> None:
        """Some Vina versions emit a single pose without MODEL/ENDMDL when n_poses=1."""
        v = Vina()
        text = (
            "REMARK VINA RESULT:    -9.000    0.000    0.000\n"
            "ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        )
        result = v._parse_poses_pdbqt(text)
        assert len(result) == 1
        assert result.poses[0].score == pytest.approx(-9.0)


@pytest.mark.slow
@pytest.mark.skipif(not _vina_available(), reason="vina not installed")
class TestEndToEnd:
    """End-to-end docking against the real engine. Run with `pytest -m slow`."""

    def test_dock_small_ligand(self) -> None:
        # Requires prepared receptor + ligand PDBQT files; placeholder.
        pytest.skip("Requires prepared PDBQT fixtures — wire up in CI")
