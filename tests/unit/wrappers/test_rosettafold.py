"""Tests for the RoseTTAFold All-Atom wrapper.

These don't require RFAA itself to be installed. They exercise
construction, repo-dir detection, sequence validation, Hydra config
construction, command-line assembly, output collection, aux-file
parsing, and PDB → Protein post-processing in isolation.

End-to-end folding against the real engine is gated on the RFAA repo
being available at $RFAA_HOME and is marked ``@pytest.mark.slow``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from molforge.wrappers.folding import (
    FoldingEngineNotInstalledError,
    RoseTTAFold,
)


def _rfaa_available() -> bool:
    home = os.environ.get("RFAA_HOME")
    return home is not None and (Path(home) / "rf2aa").is_dir()


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------


class TestConstruction:
    def test_defaults(self) -> None:
        engine = RoseTTAFold()
        assert engine.name == "RoseTTAFold"
        assert engine.repo_dir is None
        assert engine.python_executable is None
        assert engine.max_cycle is None
        assert engine.job_name == "molforge_prediction"
        assert engine.extra_overrides == []

    def test_custom_settings(self) -> None:
        engine = RoseTTAFold(
            repo_dir="/opt/RFAA",
            python_executable="/opt/conda/envs/RFAA/bin/python",
            max_cycle=10,
            job_name="my_job",
            extra_overrides=["recycling_steps=8"],
        )
        assert engine.repo_dir == "/opt/RFAA"
        assert engine.python_executable == "/opt/conda/envs/RFAA/bin/python"
        assert engine.max_cycle == 10
        assert engine.job_name == "my_job"
        assert engine.extra_overrides == ["recycling_steps=8"]

    def test_extra_overrides_is_copied(self) -> None:
        """Constructor should not hold onto the user's list."""
        overrides = ["a=1", "b=2"]
        engine = RoseTTAFold(extra_overrides=overrides)
        overrides.append("c=3")
        assert engine.extra_overrides == ["a=1", "b=2"]

    def test_construction_does_not_invoke_anything(self) -> None:
        """Construction must not shell out or import RFAA."""
        with patch("subprocess.run") as mock_run:
            RoseTTAFold(repo_dir="/opt/RFAA")
            mock_run.assert_not_called()


# ----------------------------------------------------------------------
# Repo-dir detection
# ----------------------------------------------------------------------


class TestRepoDirDetection:
    def test_repo_dir_constructor_arg(self, tmp_path: Path) -> None:
        """Explicit repo_dir takes precedence."""
        repo = tmp_path / "RFAA"
        (repo / "rf2aa").mkdir(parents=True)
        engine = RoseTTAFold(repo_dir=str(repo))
        assert engine._require_rfaa() == repo

    def test_env_var_fallback(self, tmp_path: Path, monkeypatch) -> None:
        """When repo_dir is None, fall back to RFAA_HOME."""
        repo = tmp_path / "RFAA"
        (repo / "rf2aa").mkdir(parents=True)
        monkeypatch.setenv("RFAA_HOME", str(repo))
        engine = RoseTTAFold()
        assert engine._require_rfaa() == repo

    def test_missing_raises(self, monkeypatch) -> None:
        monkeypatch.delenv("RFAA_HOME", raising=False)
        engine = RoseTTAFold()
        with pytest.raises(FoldingEngineNotInstalledError, match="RFAA_HOME"):
            engine._require_rfaa()

    def test_nonexistent_path_raises(self) -> None:
        engine = RoseTTAFold(repo_dir="/nonexistent/path/to/RFAA")
        with pytest.raises(FoldingEngineNotInstalledError, match="not a directory"):
            engine._require_rfaa()

    def test_directory_without_rf2aa_raises(self, tmp_path: Path) -> None:
        """A directory that exists but doesn't look like the RFAA repo."""
        engine = RoseTTAFold(repo_dir=str(tmp_path))
        with pytest.raises(FoldingEngineNotInstalledError, match="rf2aa"):
            engine._require_rfaa()


# ----------------------------------------------------------------------
# Sequence validation
# ----------------------------------------------------------------------


class TestSequenceValidation:
    def test_empty_sequence_raises(self) -> None:
        engine = RoseTTAFold(repo_dir="/tmp/anything")
        with pytest.raises(ValueError, match="empty"):
            engine.predict("")

    def test_non_letter_raises(self) -> None:
        engine = RoseTTAFold(repo_dir="/tmp/anything")
        with pytest.raises(ValueError, match="non-letter"):
            engine.predict("MKTV*GG")


# ----------------------------------------------------------------------
# Hydra config construction
# ----------------------------------------------------------------------


class TestConfigConstruction:
    def test_includes_base_default(self) -> None:
        engine = RoseTTAFold()
        config = engine._build_config(fasta_path=Path("/tmp/query.fasta"))
        assert "defaults:" in config
        assert "- base" in config

    def test_includes_job_name(self) -> None:
        engine = RoseTTAFold(job_name="my_test")
        config = engine._build_config(fasta_path=Path("/tmp/query.fasta"))
        assert 'job_name: "my_test"' in config

    def test_includes_fasta_path(self) -> None:
        engine = RoseTTAFold()
        config = engine._build_config(fasta_path=Path("/tmp/my_query.fasta"))
        assert "/tmp/my_query.fasta" in config

    def test_chain_id_is_A(self) -> None:
        engine = RoseTTAFold()
        config = engine._build_config(fasta_path=Path("/tmp/query.fasta"))
        assert "  A:" in config

    def test_max_cycle_included_when_set(self) -> None:
        engine = RoseTTAFold(max_cycle=10)
        config = engine._build_config(fasta_path=Path("/tmp/query.fasta"))
        assert "loader_params:" in config
        assert "MAXCYCLE: 10" in config

    def test_max_cycle_omitted_when_unset(self) -> None:
        engine = RoseTTAFold()
        config = engine._build_config(fasta_path=Path("/tmp/query.fasta"))
        assert "MAXCYCLE" not in config


# ----------------------------------------------------------------------
# Command-line assembly
# ----------------------------------------------------------------------


class TestCommandConstruction:
    def test_basic_command_structure(self, tmp_path: Path) -> None:
        engine = RoseTTAFold()
        cmd = engine._build_command(config_dir=tmp_path / "configs")
        # Default python is sys.executable.
        assert cmd[0] == sys.executable
        assert "-m" in cmd
        i = cmd.index("-m")
        assert cmd[i + 1] == "rf2aa.run_inference"

    def test_config_dir_passed(self, tmp_path: Path) -> None:
        engine = RoseTTAFold()
        cfgdir = tmp_path / "cfgs"
        cmd = engine._build_command(config_dir=cfgdir)
        i = cmd.index("--config-dir")
        assert cmd[i + 1] == str(cfgdir)

    def test_config_name_uses_job_name(self, tmp_path: Path) -> None:
        engine = RoseTTAFold(job_name="my_specific_job")
        cmd = engine._build_command(config_dir=tmp_path)
        i = cmd.index("--config-name")
        assert cmd[i + 1] == "my_specific_job"

    def test_custom_python_executable(self, tmp_path: Path) -> None:
        engine = RoseTTAFold(python_executable="/opt/rfaa/bin/python")
        cmd = engine._build_command(config_dir=tmp_path)
        assert cmd[0] == "/opt/rfaa/bin/python"

    def test_extra_overrides_appended(self, tmp_path: Path) -> None:
        engine = RoseTTAFold(extra_overrides=["foo=bar", "baz=qux"])
        cmd = engine._build_command(config_dir=tmp_path)
        assert cmd[-2:] == ["foo=bar", "baz=qux"]

    def test_no_extra_overrides_means_no_trailing_args(self, tmp_path: Path) -> None:
        engine = RoseTTAFold()
        cmd = engine._build_command(config_dir=tmp_path)
        # The last two args should be --config-name and the job name.
        assert cmd[-2] == "--config-name"
        assert cmd[-1] == engine.job_name


# ----------------------------------------------------------------------
# Subprocess invocation
# ----------------------------------------------------------------------


class TestInvoke:
    def test_successful_run_returns_silently(self, tmp_path: Path) -> None:
        engine = RoseTTAFold()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            engine._invoke(
                ["python", "-m", "rf2aa.run_inference"],
                cwd=tmp_path,
                env={},
                repo_dir=tmp_path / "repo",
            )
            mock_run.assert_called_once()

    def test_pythonpath_includes_repo_dir(self, tmp_path: Path) -> None:
        """The subprocess env's PYTHONPATH must include the RFAA repo."""
        engine = RoseTTAFold()
        repo = tmp_path / "RFAA"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            engine._invoke(
                ["python"], cwd=tmp_path, env={}, repo_dir=repo
            )
            _, kwargs = mock_run.call_args
            assert str(repo) in kwargs["env"]["PYTHONPATH"]

    def test_pythonpath_prepends_to_existing(self, tmp_path: Path) -> None:
        """If PYTHONPATH is already set, the RFAA repo should be prepended."""
        engine = RoseTTAFold()
        repo = tmp_path / "RFAA"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            engine._invoke(
                ["python"],
                cwd=tmp_path,
                env={"PYTHONPATH": "/some/existing/path"},
                repo_dir=repo,
            )
            _, kwargs = mock_run.call_args
            pp = kwargs["env"]["PYTHONPATH"]
            assert pp.startswith(str(repo))
            assert "/some/existing/path" in pp

    def test_failed_run_raises_runtime_error(self, tmp_path: Path) -> None:
        """Non-zero exit should surface as RuntimeError with stderr."""
        import subprocess

        engine = RoseTTAFold()
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=1,
                cmd=["python"],
                stderr="hydra: missing required key 'fasta_file'",
                output="",
            )
            with pytest.raises(RuntimeError, match="rf2aa.run_inference.*failed"):
                engine._invoke(
                    ["python"], cwd=tmp_path, env={}, repo_dir=tmp_path
                )

    def test_failure_message_contains_stderr(self, tmp_path: Path) -> None:
        import subprocess

        engine = RoseTTAFold()
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=2,
                cmd=["python"],
                stderr="database path UniRef30 not found",
                output="",
            )
            with pytest.raises(RuntimeError, match="UniRef30 not found"):
                engine._invoke(
                    ["python"], cwd=tmp_path, env={}, repo_dir=tmp_path
                )


# ----------------------------------------------------------------------
# Output collection
# ----------------------------------------------------------------------


class TestOutputCollection:
    def test_finds_job_named_pdb(self, tmp_path: Path) -> None:
        engine = RoseTTAFold(job_name="my_job")
        (tmp_path / "my_job.pdb").write_text("ATOM ...\n")
        (tmp_path / "my_job_aux.pt").write_bytes(b"binary content")

        pdb, aux = engine._collect_outputs(tmp_path)
        assert pdb.name == "my_job.pdb"
        assert aux is not None
        assert aux.name == "my_job_aux.pt"

    def test_falls_back_to_any_pdb(self, tmp_path: Path) -> None:
        engine = RoseTTAFold(job_name="my_job")
        # Job-named PDB absent; some other PDB present.
        (tmp_path / "other.pdb").write_text("ATOM ...\n")

        pdb, aux = engine._collect_outputs(tmp_path)
        assert pdb.name == "other.pdb"
        assert aux is None

    def test_no_pdb_raises(self, tmp_path: Path) -> None:
        engine = RoseTTAFold()
        with pytest.raises(RuntimeError, match="no .pdb output"):
            engine._collect_outputs(tmp_path)

    def test_aux_optional(self, tmp_path: Path) -> None:
        """PDB present, no aux → still works, aux is None."""
        engine = RoseTTAFold()
        (tmp_path / "molforge_prediction.pdb").write_text("ATOM ...\n")
        pdb, aux = engine._collect_outputs(tmp_path)
        assert pdb is not None
        assert aux is None


# ----------------------------------------------------------------------
# Aux file loading
# ----------------------------------------------------------------------


class TestLoadAuxFile:
    def test_missing_torch_returns_empty(self, tmp_path: Path) -> None:
        """If torch isn't importable, return {} so PDB still works."""
        engine = RoseTTAFold()
        aux = tmp_path / "x_aux.pt"
        aux.write_bytes(b"anything")
        with patch.dict("sys.modules", {"torch": None}):
            result = engine._load_aux_file(aux)
        assert result == {}

    def test_malformed_aux_returns_empty(self, tmp_path: Path) -> None:
        """A non-PyTorch file should produce empty metadata, not crash."""
        engine = RoseTTAFold()
        aux = tmp_path / "x_aux.pt"
        aux.write_bytes(b"this is not a torch file")
        # Whether torch is installed or not, malformed → empty.
        result = engine._load_aux_file(aux)
        assert result == {}

    def test_torch_tensors_converted_to_numpy(self, tmp_path: Path) -> None:
        """Tensors in the aux dict should become numpy arrays."""
        pytest.importorskip("torch")
        import torch

        engine = RoseTTAFold()
        aux = tmp_path / "x_aux.pt"
        data = {
            "plddts": torch.tensor([90.0, 85.0, 80.0]),
            "pae": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "mean_plddt": torch.tensor(85.0),
            "pae_inter": torch.tensor(4.5),
            "metadata_str": "some_string",  # non-tensor value should pass through
        }
        torch.save(data, aux)

        result = engine._load_aux_file(aux)
        assert isinstance(result["plddts"], np.ndarray)
        assert isinstance(result["pae"], np.ndarray)
        assert result["pae"].shape == (2, 2)
        assert result["metadata_str"] == "some_string"


# ----------------------------------------------------------------------
# PDB → Protein post-processing
# ----------------------------------------------------------------------


_TINY_PDB = (
    "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 82.50           N  \n"
    "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 82.50           C  \n"
    "ATOM      3  N   GLY A   2       3.000   0.000   0.000  1.00 78.20           N  \n"
    "ATOM      4  CA  GLY A   2       4.000   0.000   0.000  1.00 78.20           C  \n"
    "END\n"
)


class TestParseOutputs:
    def test_attaches_engine_metadata(self) -> None:
        engine = RoseTTAFold(job_name="my_job")
        protein = engine._parse_outputs(
            pdb_text=_TINY_PDB, confidence={}, sequence="AG"
        )
        assert protein.metadata["engine"] == "RoseTTAFold"
        assert protein.metadata["source_sequence"] == "AG"
        assert protein.metadata["job_name"] == "my_job"

    def test_per_residue_confidence(self) -> None:
        engine = RoseTTAFold()
        protein = engine._parse_outputs(
            pdb_text=_TINY_PDB, confidence={}, sequence="AG"
        )
        per_residue = protein.metadata["confidence_per_residue"]
        assert per_residue.shape == (2,)
        np.testing.assert_allclose(per_residue, [82.5, 78.2], atol=0.01)

    def test_mean_confidence(self) -> None:
        engine = RoseTTAFold()
        protein = engine._parse_outputs(
            pdb_text=_TINY_PDB, confidence={}, sequence="AG"
        )
        assert protein.metadata["mean_confidence"] == pytest.approx(80.35, abs=0.01)

    def test_per_atom_confidence(self) -> None:
        engine = RoseTTAFold()
        protein = engine._parse_outputs(
            pdb_text=_TINY_PDB, confidence={}, sequence="AG"
        )
        per_atom = protein.metadata["confidence_per_atom"]
        assert per_atom.shape == (4,)
        np.testing.assert_allclose(per_atom[:2], 82.5)
        np.testing.assert_allclose(per_atom[2:], 78.2)

    def test_pae_inter_surfaced_when_present(self) -> None:
        engine = RoseTTAFold()
        protein = engine._parse_outputs(
            pdb_text=_TINY_PDB,
            confidence={"pae_inter": 4.8, "mean_pae": 5.1, "pae_prot": 4.5},
            sequence="AG",
        )
        assert protein.metadata["pae_inter"] == 4.8
        assert protein.metadata["mean_pae"] == 5.1
        assert protein.metadata["pae_prot"] == 4.5

    def test_pae_matrix_surfaced_when_present(self) -> None:
        engine = RoseTTAFold()
        pae = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        protein = engine._parse_outputs(
            pdb_text=_TINY_PDB, confidence={"pae": pae}, sequence="AG"
        )
        np.testing.assert_array_equal(protein.metadata["pae"], pae)

    def test_missing_pae_does_not_crash(self) -> None:
        """No confidence keys → no PAE in metadata, but PDB-derived stuff still there."""
        engine = RoseTTAFold()
        protein = engine._parse_outputs(
            pdb_text=_TINY_PDB, confidence={}, sequence="AG"
        )
        assert "pae" not in protein.metadata
        assert "pae_inter" not in protein.metadata
        assert "confidence_per_residue" in protein.metadata


# ----------------------------------------------------------------------
# Uniform-confidence contract
# ----------------------------------------------------------------------


class TestUniformConfidenceConvention:
    """RoseTTAFold metadata must share the same key contract as other folders."""

    def test_same_uniform_keys_as_other_folders(self) -> None:
        from molforge.wrappers.folding import AlphaFold, ESMFold

        rf_meta = RoseTTAFold()._parse_outputs(
            pdb_text=_TINY_PDB, confidence={}, sequence="AG"
        ).metadata
        af_meta = AlphaFold()._pdb_to_protein(_TINY_PDB, sequence="AG").metadata
        esm_meta = ESMFold()._pdb_to_protein(_TINY_PDB, sequence="AG").metadata

        for key in (
            "engine",
            "source_sequence",
            "confidence_per_residue",
            "confidence_per_atom",
            "mean_confidence",
        ):
            assert key in rf_meta, f"RoseTTAFold output missing '{key}'"
            assert key in af_meta, f"AlphaFold output missing '{key}'"
            assert key in esm_meta, f"ESMFold output missing '{key}'"


# ----------------------------------------------------------------------
# Deprecated Rosetta alias
# ----------------------------------------------------------------------


class TestDeprecatedRosettaAlias:
    def test_rosetta_is_subclass_of_rosettafold(self) -> None:
        from molforge.wrappers.folding import Rosetta

        assert issubclass(Rosetta, RoseTTAFold)

    def test_rosetta_emits_deprecation_warning(self) -> None:
        from molforge.wrappers.folding import Rosetta

        with pytest.warns(DeprecationWarning, match="RoseTTAFold"):
            Rosetta(repo_dir="/tmp/anything")

    def test_rosetta_name_attribute(self) -> None:
        from molforge.wrappers.folding import Rosetta

        with pytest.warns(DeprecationWarning):
            engine = Rosetta()
        # Name distinguishes it from the non-deprecated class.
        assert engine.name == "Rosetta"


# ----------------------------------------------------------------------
# End-to-end (skipped unless RFAA is set up)
# ----------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not _rfaa_available(), reason="RFAA_HOME not set or invalid")
class TestEndToEnd:
    """Contract test against the real RFAA install. Run with `pytest -m slow`."""

    def test_short_sequence_folds(self) -> None:
        # Requires the full RFAA setup (clone + conda env + databases +
        # weights). Provided as a smoke check for anyone with it.
        engine = RoseTTAFold(max_cycle=4)
        protein = engine.predict("MKTVRQERLKSIVRILERSK")
        assert protein.n_residues == 20
        assert "confidence_per_residue" in protein.metadata
        assert protein.metadata["engine"] == "RoseTTAFold"
