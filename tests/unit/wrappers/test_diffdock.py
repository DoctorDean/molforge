"""Tests for the DiffDock wrapper.

These don't require DiffDock (or torch) to be installed. They exercise:
  - Construction with parameter validation
  - Installation-path resolution from env vars and explicit args
  - SDF atom-block parsing and confidence-from-filename parsing
  - The _run_cli subprocess seam, via a mocked subprocess.run
  - Receptor / ligand argument handling

End-to-end docking requires the real model and weights and is out of
scope for the unit suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molforge.docking import DockingEngineNotInstalledError, DockingResult
from molforge.wrappers.docking import DiffDock
from molforge.wrappers.docking.diffdock import (
    _confidence_from_filename,
)

# A minimal valid V2000 SDF — a 3-atom molecule, enough to parse.
_SAMPLE_SDF = """ligand
  molforge generated

  3  2  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.9572    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.2400    0.9270    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1  3  1  0
M  END
$$$$
"""


class TestConstruction:
    def test_defaults(self) -> None:
        engine = DiffDock()
        assert engine.name == "DiffDock"
        assert engine.repo_dir is None
        assert engine.samples_per_complex == 10
        assert engine.inference_steps == 20
        assert engine.batch_size == 10

    def test_custom_settings(self) -> None:
        engine = DiffDock(
            repo_dir="/custom/path",
            samples_per_complex=40,
            inference_steps=10,
            batch_size=8,
        )
        assert engine.repo_dir == Path("/custom/path")
        assert engine.samples_per_complex == 40
        assert engine.inference_steps == 10
        assert engine.batch_size == 8

    def test_invalid_samples_per_complex(self) -> None:
        with pytest.raises(ValueError, match="samples_per_complex"):
            DiffDock(samples_per_complex=0)

    def test_invalid_inference_steps(self) -> None:
        with pytest.raises(ValueError, match="inference_steps"):
            DiffDock(inference_steps=0)

    def test_invalid_batch_size(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            DiffDock(batch_size=-1)

    def test_construction_doesnt_resolve_install(self) -> None:
        """Construction must not probe the filesystem or env vars."""
        DiffDock()


class TestInstallationResolution:
    def test_missing_install_clear_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DIFFDOCK_HOME", raising=False)
        engine = DiffDock()
        with pytest.raises(DockingEngineNotInstalledError, match="cloned"):
            engine._resolve_repo()

    def test_env_var_used(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        repo = tmp_path / "DiffDock"
        repo.mkdir()
        (repo / "inference.py").write_text("# placeholder\n")
        monkeypatch.setenv("DIFFDOCK_HOME", str(repo))
        assert DiffDock()._resolve_repo() == repo

    def test_explicit_dir_overrides_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        env_repo = tmp_path / "env"
        explicit = tmp_path / "explicit"
        for d in (env_repo, explicit):
            d.mkdir()
            (d / "inference.py").write_text("# placeholder\n")
        monkeypatch.setenv("DIFFDOCK_HOME", str(env_repo))
        assert DiffDock(repo_dir=explicit)._resolve_repo() == explicit

    def test_dir_without_inference_script_errors(self, tmp_path: Path) -> None:
        engine = DiffDock(repo_dir=tmp_path)
        with pytest.raises(DockingEngineNotInstalledError, match=r"inference\.py"):
            engine._resolve_repo()

    def test_nonexistent_dir_errors(self, tmp_path: Path) -> None:
        engine = DiffDock(repo_dir=tmp_path / "does_not_exist")
        with pytest.raises(DockingEngineNotInstalledError, match="not a directory"):
            engine._resolve_repo()


class TestConfidenceFromFilename:
    def test_positive_confidence(self) -> None:
        assert _confidence_from_filename("rank1_confidence1.73.sdf") == pytest.approx(1.73)

    def test_negative_confidence(self) -> None:
        assert _confidence_from_filename("rank3_confidence-0.42.sdf") == pytest.approx(-0.42)

    def test_no_confidence_marker_returns_none(self) -> None:
        # DiffDock writes the top pose as a bare rank1.sdf in some versions.
        assert _confidence_from_filename("rank1.sdf") is None

    def test_unparseable_confidence_returns_none(self) -> None:
        assert _confidence_from_filename("rank1_confidenceXYZ.sdf") is None


class TestRunCli:
    """`_run_cli` is the subprocess-driving seam — exercised with a
    mocked ``subprocess.run`` so neither DiffDock nor torch is
    needed."""

    @staticmethod
    def _fake_install(tmp_path: Path) -> Path:
        repo = tmp_path / "DiffDock"
        repo.mkdir()
        (repo / "inference.py").write_text("# placeholder\n")
        return repo

    @staticmethod
    def _receptor_pdb(tmp_path: Path) -> Path:
        pdb = tmp_path / "receptor.pdb"
        pdb.write_text(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n"
        )
        return pdb

    @staticmethod
    def _out_dir_from_cmd(cmd: list[str]) -> Path:
        return Path(cmd[cmd.index("--out_dir") + 1])

    def test_invokes_subprocess_and_parses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = self._fake_install(tmp_path)
        receptor = self._receptor_pdb(tmp_path)
        captured: dict[str, object] = {}

        def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            captured["cmd"] = cmd
            complex_dir = self._out_dir_from_cmd(cmd) / "complex0"
            complex_dir.mkdir(parents=True)
            (complex_dir / "rank1_confidence0.85.sdf").write_text(_SAMPLE_SDF)
            (complex_dir / "rank2_confidence-0.30.sdf").write_text(_SAMPLE_SDF)
            return None

        monkeypatch.setattr("subprocess.run", fake_run)
        result = DiffDock(repo_dir=repo)._run_cli(
            receptor=receptor,
            ligand="CCO",
            repo=repo,
            timeout=None,
        )
        assert isinstance(result, DockingResult)
        assert result.engine == "DiffDock"
        assert len(result) == 2
        cmd = captured["cmd"]
        assert "-m" in cmd and "inference" in cmd
        assert "--protein_path" in cmd
        # SMILES ligand is passed through verbatim.
        assert cmd[cmd.index("--ligand_description") + 1] == "CCO"

    def test_poses_sorted_best_first(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        repo = self._fake_install(tmp_path)
        receptor = self._receptor_pdb(tmp_path)

        def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            cdir = self._out_dir_from_cmd(cmd) / "complex0"
            cdir.mkdir(parents=True)
            # Deliberately write them out of order.
            (cdir / "rank2_confidence-0.30.sdf").write_text(_SAMPLE_SDF)
            (cdir / "rank1_confidence0.85.sdf").write_text(_SAMPLE_SDF)
            return None

        monkeypatch.setattr("subprocess.run", fake_run)
        result = DiffDock(repo_dir=repo)._run_cli(
            receptor=receptor, ligand="CCO", repo=repo, timeout=None
        )
        # Best = highest confidence; score is the negated confidence,
        # so the best pose has the most-negative score and rank 0.
        assert result.best.metadata["confidence"] == pytest.approx(0.85)
        assert result.poses[0].rank == 0
        assert result.poses[1].rank == 1
        assert result.poses[0].score < result.poses[1].score

    def test_ligand_file_passed_as_absolute_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = self._fake_install(tmp_path)
        receptor = self._receptor_pdb(tmp_path)
        ligand_file = tmp_path / "ligand.sdf"
        ligand_file.write_text(_SAMPLE_SDF)
        captured: dict[str, object] = {}

        def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            captured["cmd"] = cmd
            cdir = self._out_dir_from_cmd(cmd) / "complex0"
            cdir.mkdir(parents=True)
            (cdir / "rank1_confidence0.5.sdf").write_text(_SAMPLE_SDF)
            return None

        monkeypatch.setattr("subprocess.run", fake_run)
        DiffDock(repo_dir=repo)._run_cli(
            receptor=receptor, ligand=ligand_file, repo=repo, timeout=None
        )
        cmd = captured["cmd"]
        ligand_arg = cmd[cmd.index("--ligand_description") + 1]
        assert Path(ligand_arg).is_absolute()
        assert Path(ligand_arg) == ligand_file.resolve()

    def test_no_output_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        repo = self._fake_install(tmp_path)
        receptor = self._receptor_pdb(tmp_path)

        def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            return None  # writes nothing

        monkeypatch.setattr("subprocess.run", fake_run)
        with pytest.raises(RuntimeError, match=r"no rank.*\.sdf output"):
            DiffDock(repo_dir=repo)._run_cli(
                receptor=receptor, ligand="CCO", repo=repo, timeout=None
            )

    def test_subprocess_failure_becomes_runtime_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess

        repo = self._fake_install(tmp_path)
        receptor = self._receptor_pdb(tmp_path)

        def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr="diffusion crashed")

        monkeypatch.setattr("subprocess.run", fake_run)
        with pytest.raises(RuntimeError, match="DiffDock failed"):
            DiffDock(repo_dir=repo)._run_cli(
                receptor=receptor, ligand="CCO", repo=repo, timeout=None
            )

    def test_dock_resolves_install_then_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The public entry point: dock() resolves DIFFDOCK_HOME, then
        # delegates to _run_cli.
        repo = self._fake_install(tmp_path)
        receptor = self._receptor_pdb(tmp_path)
        monkeypatch.setenv("DIFFDOCK_HOME", str(repo))

        def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            cdir = self._out_dir_from_cmd(cmd) / "complex0"
            cdir.mkdir(parents=True)
            (cdir / "rank1_confidence0.6.sdf").write_text(_SAMPLE_SDF)
            return None

        monkeypatch.setattr("subprocess.run", fake_run)
        result = DiffDock().dock(receptor=receptor, ligand="CCO")
        assert len(result) == 1
        assert result.best.metadata["confidence"] == pytest.approx(0.6)

    def test_protein_receptor_is_materialized_and_carried_through(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A Protein receptor is written to a PDB for DiffDock, and the
        same Protein is carried onto the DockingResult."""
        from molforge.io import load

        repo = self._fake_install(tmp_path)
        # Reuse an existing fixture as the receptor.
        receptor = load(Path(__file__).parents[2] / "fixtures" / "pdb" / "tripeptide.pdb")
        captured: dict[str, object] = {}

        def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            captured["cmd"] = cmd
            cdir = self._out_dir_from_cmd(cmd) / "complex0"
            cdir.mkdir(parents=True)
            (cdir / "rank1_confidence0.7.sdf").write_text(_SAMPLE_SDF)
            return None

        monkeypatch.setattr("subprocess.run", fake_run)
        result = DiffDock(repo_dir=repo)._run_cli(
            receptor=receptor, ligand="CCO", repo=repo, timeout=None
        )
        cmd = captured["cmd"]
        assert cmd[cmd.index("--protein_path") + 1].endswith(".pdb")
        assert result.receptor is receptor
