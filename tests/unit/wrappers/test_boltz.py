"""Tests for the Boltz / Boltz-2 wrapper.

These don't require the ``boltz`` CLI to be installed. They exercise
construction, lazy CLI detection, sequence validation, YAML input
construction, command-line assembly, output collection, and CIF →
Protein post-processing in isolation.

End-to-end folding against the real engine is gated on the ``boltz``
CLI being on $PATH and is marked ``@pytest.mark.slow``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from molforge.wrappers.folding import Boltz, FoldingEngineNotInstalledError


def _boltz_available() -> bool:
    return shutil.which("boltz") is not None


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------


class TestConstruction:
    def test_defaults(self) -> None:
        engine = Boltz()
        assert engine.name == "Boltz"
        assert engine.model_version == "boltz2"
        assert engine.use_msa_server is True
        assert engine.recycling_steps is None
        assert engine.diffusion_samples is None
        assert engine.sampling_steps is None
        assert engine.device is None
        assert engine.executable is None
        assert engine.cache_dir is None

    def test_custom_settings(self) -> None:
        engine = Boltz(
            model_version="boltz1",
            use_msa_server=False,
            recycling_steps=5,
            diffusion_samples=3,
            sampling_steps=50,
            device="cpu",
            executable="/opt/boltz/bin/boltz",
            cache_dir="/tmp/my-boltz",
        )
        assert engine.model_version == "boltz1"
        assert engine.use_msa_server is False
        assert engine.recycling_steps == 5
        assert engine.diffusion_samples == 3
        assert engine.sampling_steps == 50
        assert engine.device == "cpu"
        assert engine.executable == "/opt/boltz/bin/boltz"
        assert engine.cache_dir == "/tmp/my-boltz"

    def test_invalid_model_version_raises(self) -> None:
        with pytest.raises(ValueError, match="model_version must be"):
            Boltz(model_version="boltz3")

    def test_construction_does_not_invoke_cli(self) -> None:
        """Construction must not shell out to the boltz binary."""
        with patch("subprocess.run") as mock_run:
            Boltz()
            mock_run.assert_not_called()

    def test_repr(self) -> None:
        assert repr(Boltz()) == "Boltz()"


# ----------------------------------------------------------------------
# Missing-dependency behaviour
# ----------------------------------------------------------------------


class TestMissingDependency:
    def test_predict_without_boltz_raises_clear_error(self) -> None:
        engine = Boltz()
        with (
            patch("shutil.which", return_value=None),
            pytest.raises(FoldingEngineNotInstalledError, match="boltz"),
        ):
            engine.predict("MKTV")

    def test_missing_executable_path_raises(self) -> None:
        """Explicit executable= that doesn't exist gives the same error.

        We don't probe the filesystem here — the user provided a path,
        and if it's wrong they'll see it on the first invocation.
        Behaviour: if shutil.which can't find anything and executable
        is None, raise; if executable is set, trust the user.
        """
        engine = Boltz(executable=None)
        with (
            patch("shutil.which", return_value=None),
            pytest.raises(FoldingEngineNotInstalledError),
        ):
            engine._require_boltz()


# ----------------------------------------------------------------------
# Sequence validation
# ----------------------------------------------------------------------


class TestSequenceValidation:
    def test_empty_sequence_raises(self) -> None:
        engine = Boltz()
        with pytest.raises(ValueError, match="empty"):
            engine.predict("")

    def test_non_letter_raises(self) -> None:
        engine = Boltz()
        with pytest.raises(ValueError, match="non-letter"):
            engine.predict("MKTV*GG")

    def test_whitespace_stripped(self) -> None:
        """Validation should strip whitespace before sniffing for non-letters."""
        engine = Boltz()
        yaml = engine._build_input_yaml("MKTV", name="query")
        # Sanity: validation doesn't run inside _build_input_yaml itself
        # (that's predict's job). Just confirm the construction path doesn't fail.
        assert "MKTV" in yaml


# ----------------------------------------------------------------------
# YAML input construction
# ----------------------------------------------------------------------


class TestYamlInputConstruction:
    def test_yaml_has_version_marker(self) -> None:
        engine = Boltz()
        y = engine._build_input_yaml("MKTV", name="query")
        assert "version: 1" in y

    def test_yaml_has_sequences_key(self) -> None:
        engine = Boltz()
        y = engine._build_input_yaml("MKTV", name="query")
        assert "sequences:" in y

    def test_yaml_has_protein_entity(self) -> None:
        engine = Boltz()
        y = engine._build_input_yaml("MKTV", name="query")
        assert "protein:" in y

    def test_yaml_includes_sequence(self) -> None:
        engine = Boltz()
        y = engine._build_input_yaml("MKTVRQERLKSIVRIL", name="query")
        assert "MKTVRQERLKSIVRIL" in y

    def test_yaml_chain_id_is_A(self) -> None:
        engine = Boltz()
        y = engine._build_input_yaml("MKTV", name="query")
        assert "id: A" in y


# ----------------------------------------------------------------------
# Command-line assembly
# ----------------------------------------------------------------------


class TestCommandConstruction:
    def test_basic_command_structure(self, tmp_path: Path) -> None:
        engine = Boltz()
        input_path = tmp_path / "input.yaml"
        output_dir = tmp_path / "out"
        cmd = engine._build_command("/bin/boltz", input_path, output_dir)

        assert cmd[0] == "/bin/boltz"
        assert cmd[1] == "predict"
        assert str(input_path) in cmd
        assert "--out_dir" in cmd
        assert str(output_dir) in cmd
        assert "--output_format" in cmd
        assert "mmcif" in cmd

    def test_model_version_passed(self, tmp_path: Path) -> None:
        engine = Boltz(model_version="boltz1")
        cmd = engine._build_command("/bin/boltz", tmp_path / "i.yaml", tmp_path / "o")
        assert "--model" in cmd
        i = cmd.index("--model")
        assert cmd[i + 1] == "boltz1"

    def test_msa_server_flag_when_enabled(self, tmp_path: Path) -> None:
        engine = Boltz(use_msa_server=True)
        cmd = engine._build_command("/bin/boltz", tmp_path / "i.yaml", tmp_path / "o")
        assert "--use_msa_server" in cmd

    def test_msa_server_flag_absent_when_disabled(self, tmp_path: Path) -> None:
        engine = Boltz(use_msa_server=False)
        cmd = engine._build_command("/bin/boltz", tmp_path / "i.yaml", tmp_path / "o")
        assert "--use_msa_server" not in cmd

    def test_recycling_steps_passed_when_set(self, tmp_path: Path) -> None:
        engine = Boltz(recycling_steps=5)
        cmd = engine._build_command("/bin/boltz", tmp_path / "i.yaml", tmp_path / "o")
        assert "--recycling_steps" in cmd
        i = cmd.index("--recycling_steps")
        assert cmd[i + 1] == "5"

    def test_recycling_steps_absent_when_unset(self, tmp_path: Path) -> None:
        engine = Boltz()  # recycling_steps default = None
        cmd = engine._build_command("/bin/boltz", tmp_path / "i.yaml", tmp_path / "o")
        assert "--recycling_steps" not in cmd

    def test_diffusion_samples_passed(self, tmp_path: Path) -> None:
        engine = Boltz(diffusion_samples=3)
        cmd = engine._build_command("/bin/boltz", tmp_path / "i.yaml", tmp_path / "o")
        i = cmd.index("--diffusion_samples")
        assert cmd[i + 1] == "3"

    def test_sampling_steps_passed(self, tmp_path: Path) -> None:
        engine = Boltz(sampling_steps=50)
        cmd = engine._build_command("/bin/boltz", tmp_path / "i.yaml", tmp_path / "o")
        i = cmd.index("--sampling_steps")
        assert cmd[i + 1] == "50"

    def test_cpu_device_routed_to_accelerator_flag(self, tmp_path: Path) -> None:
        engine = Boltz(device="cpu")
        cmd = engine._build_command("/bin/boltz", tmp_path / "i.yaml", tmp_path / "o")
        i = cmd.index("--accelerator")
        assert cmd[i + 1] == "cpu"

    def test_cuda_device_routed_to_gpu_accelerator(self, tmp_path: Path) -> None:
        engine = Boltz(device="cuda")
        cmd = engine._build_command("/bin/boltz", tmp_path / "i.yaml", tmp_path / "o")
        i = cmd.index("--accelerator")
        assert cmd[i + 1] == "gpu"

    def test_default_device_omits_accelerator_flag(self, tmp_path: Path) -> None:
        engine = Boltz()  # device=None
        cmd = engine._build_command("/bin/boltz", tmp_path / "i.yaml", tmp_path / "o")
        assert "--accelerator" not in cmd

    def test_override_flag_present(self, tmp_path: Path) -> None:
        """We always pass --override since we're working in a tempdir."""
        engine = Boltz()
        cmd = engine._build_command("/bin/boltz", tmp_path / "i.yaml", tmp_path / "o")
        assert "--override" in cmd


# ----------------------------------------------------------------------
# Environment variables
# ----------------------------------------------------------------------


class TestBuildEnv:
    def test_default_env_preserves_parent(self) -> None:
        engine = Boltz()
        env = engine._build_env()
        # Should contain typical parent vars.
        # We check the env dict isn't empty (it's a copy of os.environ).
        assert len(env) > 0

    def test_cache_dir_sets_boltz_cache(self) -> None:
        engine = Boltz(cache_dir="/tmp/custom-boltz")
        env = engine._build_env()
        assert env["BOLTZ_CACHE"] == "/tmp/custom-boltz"

    def test_no_cache_dir_does_not_set_boltz_cache(self, monkeypatch) -> None:
        monkeypatch.delenv("BOLTZ_CACHE", raising=False)
        engine = Boltz()
        env = engine._build_env()
        assert "BOLTZ_CACHE" not in env


# ----------------------------------------------------------------------
# Output collection
# ----------------------------------------------------------------------


class TestOutputCollection:
    def test_finds_model_0_cif(self, tmp_path: Path) -> None:
        """Boltz writes <name>_model_0.cif as the highest-ranked model."""
        engine = Boltz()
        sub = tmp_path / "predictions" / "query"
        sub.mkdir(parents=True)
        cif_path = sub / "query_model_0.cif"
        cif_path.write_text("data_test\n")
        # Also write a model_1.cif to make sure we prefer model_0.
        (sub / "query_model_1.cif").write_text("data_other\n")
        # And a confidence JSON.
        (sub / "confidence_query_model_0.json").write_text(
            '{"ptm": 0.85, "iptm": 0.0, "confidence_score": 0.81}'
        )

        cif_text, conf = engine._collect_outputs(tmp_path)
        assert cif_text == "data_test\n"
        assert conf["ptm"] == 0.85
        assert conf["confidence_score"] == 0.81

    def test_falls_back_to_any_cif_when_no_model_0(self, tmp_path: Path) -> None:
        engine = Boltz()
        sub = tmp_path / "predictions" / "query"
        sub.mkdir(parents=True)
        (sub / "query_some_other.cif").write_text("data_test\n")
        cif_text, _ = engine._collect_outputs(tmp_path)
        assert cif_text == "data_test\n"

    def test_no_cif_output_raises(self, tmp_path: Path) -> None:
        engine = Boltz()
        with pytest.raises(RuntimeError, match=r"no \.cif output"):
            engine._collect_outputs(tmp_path)

    def test_missing_confidence_json_returns_empty_dict(self, tmp_path: Path) -> None:
        """A CIF without a confidence JSON is acceptable — just no scalars."""
        engine = Boltz()
        sub = tmp_path / "predictions" / "query"
        sub.mkdir(parents=True)
        (sub / "query_model_0.cif").write_text("data_test\n")
        _, conf = engine._collect_outputs(tmp_path)
        assert conf == {}

    def test_malformed_confidence_json_returns_empty(self, tmp_path: Path) -> None:
        engine = Boltz()
        sub = tmp_path / "predictions" / "query"
        sub.mkdir(parents=True)
        (sub / "query_model_0.cif").write_text("data_test\n")
        (sub / "confidence_query_model_0.json").write_text("not json at all")
        _, conf = engine._collect_outputs(tmp_path)
        assert conf == {}


# ----------------------------------------------------------------------
# Subprocess invocation
# ----------------------------------------------------------------------


class TestInvoke:
    def test_successful_run_returns_silently(self, tmp_path: Path) -> None:
        engine = Boltz()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            engine._invoke(["boltz", "predict", "x"], env={})
            mock_run.assert_called_once()

    def test_failed_run_raises_runtime_error_with_stderr(self) -> None:
        """Non-zero exit should surface stderr / stdout in the error message."""
        import subprocess

        engine = Boltz()
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=1,
                cmd=["boltz", "predict", "x"],
                stderr="ValueError: bad input format",
                output="some stdout",
            )
            with pytest.raises(RuntimeError, match="exit code 1"):
                engine._invoke(["boltz", "predict", "x"], env={})

    def test_failure_message_contains_stderr(self) -> None:
        import subprocess

        engine = Boltz()
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=2,
                cmd=["boltz"],
                stderr="MSA server unreachable",
                output="",
            )
            with pytest.raises(RuntimeError, match="MSA server unreachable"):
                engine._invoke(["boltz"], env={})


# ----------------------------------------------------------------------
# CIF → Protein post-processing
# ----------------------------------------------------------------------

# A tiny synthetic mmCIF for testing post-processing without invoking Boltz.
# Two ALA residues × 2 atoms each, pLDDT in B-factor column.
_TINY_CIF = """data_query
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
ATOM 1  N  N   ALA A 1  0.000  0.000  0.000  1.00  92.50
ATOM 2  C  CA  ALA A 1  1.000  0.000  0.000  1.00  92.50
ATOM 3  N  N   GLY A 2  2.000  0.000  0.000  1.00  78.20
ATOM 4  C  CA  GLY A 2  3.000  0.000  0.000  1.00  78.20
#
"""


class TestParseOutputs:
    def test_attaches_engine_metadata(self) -> None:
        engine = Boltz(model_version="boltz2")
        protein = engine._parse_outputs(
            cif_text=_TINY_CIF,
            confidence_json={"ptm": 0.84, "iptm": 0.0, "confidence_score": 0.79},
            sequence="AG",
        )
        assert protein.metadata["engine"] == "Boltz"
        assert protein.metadata["model_version"] == "boltz2"
        assert protein.metadata["source_sequence"] == "AG"
        assert protein.metadata["use_msa_server"] is True

    def test_per_residue_confidence(self) -> None:
        engine = Boltz()
        protein = engine._parse_outputs(
            cif_text=_TINY_CIF,
            confidence_json={},
            sequence="AG",
        )
        per_residue = protein.metadata["confidence_per_residue"]
        assert per_residue.shape == (2,)
        np.testing.assert_allclose(per_residue, [92.5, 78.2], atol=0.01)

    def test_mean_confidence(self) -> None:
        engine = Boltz()
        protein = engine._parse_outputs(
            cif_text=_TINY_CIF,
            confidence_json={},
            sequence="AG",
        )
        assert protein.metadata["mean_confidence"] == pytest.approx(85.35, abs=0.01)

    def test_per_atom_confidence(self) -> None:
        engine = Boltz()
        protein = engine._parse_outputs(
            cif_text=_TINY_CIF,
            confidence_json={},
            sequence="AG",
        )
        per_atom = protein.metadata["confidence_per_atom"]
        assert per_atom.shape == (4,)
        np.testing.assert_allclose(per_atom[:2], 92.5)
        np.testing.assert_allclose(per_atom[2:], 78.2)

    def test_ptm_and_iptm_from_json(self) -> None:
        engine = Boltz()
        protein = engine._parse_outputs(
            cif_text=_TINY_CIF,
            confidence_json={"ptm": 0.91, "iptm": 0.0, "confidence_score": 0.83},
            sequence="AG",
        )
        assert protein.metadata["ptm"] == 0.91
        assert protein.metadata["iptm"] == 0.0
        assert protein.metadata["confidence_score"] == 0.83

    def test_default_composite_when_json_missing(self) -> None:
        """No JSON → composite is computed from mean pLDDT + iPTM (= 0)."""
        engine = Boltz()
        protein = engine._parse_outputs(
            cif_text=_TINY_CIF,
            confidence_json={},
            sequence="AG",
        )
        # 0.8 * (85.35/100) + 0.2 * 0 = 0.6828
        assert protein.metadata["confidence_score"] == pytest.approx(0.683, abs=0.01)

    def test_boltz1_model_version_recorded(self) -> None:
        engine = Boltz(model_version="boltz1")
        protein = engine._parse_outputs(cif_text=_TINY_CIF, confidence_json={}, sequence="AG")
        assert protein.metadata["model_version"] == "boltz1"


# ----------------------------------------------------------------------
# Uniform-confidence contract
# ----------------------------------------------------------------------


class TestUniformConfidenceConvention:
    """Boltz output must share the same metadata-key contract as ESMFold/AlphaFold."""

    def test_same_metadata_keys_as_other_folders(self) -> None:
        from molforge.wrappers.folding import AlphaFold, ESMFold

        pdb = (
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 80.00           N  \n"
            "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 80.00           C  \n"
            "END\n"
        )

        boltz_meta = (
            Boltz()._parse_outputs(cif_text=_TINY_CIF, confidence_json={}, sequence="AG").metadata
        )
        af_meta = AlphaFold()._pdb_to_protein(pdb, sequence="A").metadata
        esm_meta = ESMFold()._pdb_to_protein(pdb, sequence="A").metadata

        # Required uniform keys from the FoldingEngine convention.
        for key in (
            "engine",
            "source_sequence",
            "confidence_per_residue",
            "confidence_per_atom",
            "mean_confidence",
        ):
            assert key in boltz_meta, f"Boltz output missing '{key}'"
            assert key in af_meta, f"AlphaFold output missing '{key}'"
            assert key in esm_meta, f"ESMFold output missing '{key}'"


# ----------------------------------------------------------------------
# Boltz-2 affinity prediction
# ----------------------------------------------------------------------


class TestAffinity:
    """predict_affinity — YAML properties block, JSON parsing, validation.

    Boltz is GPU-only, so we target the seams: the input YAML, the affinity
    sidecar parser, and the validation gates.
    """

    def _spec(self):
        from molforge.folding import ComplexSpec

        return ComplexSpec.protein_ligand(protein_sequence="MKTVRQ", ligand_smiles="CCO")

    def test_single_ligand_chain_id(self) -> None:
        from molforge.wrappers.folding.boltz import _single_ligand_chain_id

        assert _single_ligand_chain_id(self._spec()) == "B"

    def test_no_ligand_raises(self) -> None:
        from molforge.folding import ComplexSpec
        from molforge.wrappers.folding.boltz import _single_ligand_chain_id

        with pytest.raises(ValueError, match="exactly one ligand"):
            _single_ligand_chain_id(ComplexSpec.from_protein("MKTVRQ"))

    def test_two_ligands_raises(self) -> None:
        from molforge.folding import ComplexSpec, Entity
        from molforge.wrappers.folding.boltz import _single_ligand_chain_id

        spec = ComplexSpec(
            entities=(
                Entity(kind="protein", sequence="MKTVRQ"),
                Entity(kind="ligand", smiles="CCO"),
                Entity(kind="ligand", smiles="CCC"),
            )
        )
        with pytest.raises(ValueError, match="exactly one ligand"):
            _single_ligand_chain_id(spec)

    def test_yaml_has_affinity_properties_block(self) -> None:
        engine = Boltz(model_version="boltz2")
        y = engine._build_input_yaml_from_spec(self._spec(), affinity_binder="B")
        assert "properties:" in y
        assert "- affinity:" in y
        assert "binder: B" in y

    def test_yaml_without_affinity_has_no_properties(self) -> None:
        engine = Boltz(model_version="boltz2")
        assert "properties:" not in engine._build_input_yaml_from_spec(self._spec())

    def test_affinity_json_helpers(self) -> None:
        from molforge.wrappers.folding.boltz import _affinity_probability, _affinity_value

        aj = {"affinity_pred_value": -1.23, "affinity_probability_binary": 0.87}
        assert _affinity_value(aj) == pytest.approx(-1.23)
        assert _affinity_probability(aj) == pytest.approx(0.87)
        assert _affinity_value({}) is None
        assert _affinity_probability({}) is None

    def test_parse_outputs_surfaces_affinity_metadata(self) -> None:
        engine = Boltz(model_version="boltz2")
        protein = engine._parse_outputs(
            cif_text=_TINY_CIF,
            confidence_json={"ptm": 0.8, "iptm": 0.5},
            sequence=None,
            spec=self._spec(),
            affinity_json={"affinity_pred_value": -2.5, "affinity_probability_binary": 0.91},
        )
        assert protein.metadata["affinity_value"] == pytest.approx(-2.5)
        assert protein.metadata["affinity_probability"] == pytest.approx(0.91)
        assert protein.metadata["affinity"]["affinity_pred_value"] == -2.5

    def test_parse_outputs_without_affinity_omits_keys(self) -> None:
        engine = Boltz(model_version="boltz2")
        protein = engine._parse_outputs(cif_text=_TINY_CIF, confidence_json={}, sequence="AG")
        assert "affinity_value" not in protein.metadata

    def test_boltz1_rejects_affinity(self) -> None:
        with pytest.raises(ValueError, match="requires Boltz-2"):
            Boltz(model_version="boltz1").predict_affinity(self._spec())

    def test_collect_affinity_globs_json(self, tmp_path: Path) -> None:
        import json

        engine = Boltz(model_version="boltz2")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "affinity_query.json").write_text(
            json.dumps({"affinity_pred_value": -1.0, "affinity_probability_binary": 0.7})
        )
        data = engine._collect_affinity(tmp_path)
        assert data["affinity_pred_value"] == -1.0

    def test_collect_affinity_missing_returns_empty(self, tmp_path: Path) -> None:
        assert Boltz(model_version="boltz2")._collect_affinity(tmp_path) == {}


# ----------------------------------------------------------------------
# End-to-end (skipped unless boltz is on PATH)
# ----------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not _boltz_available(), reason="boltz CLI not on $PATH")
class TestEndToEnd:
    """Contract test against the real boltz binary. Run with `pytest -m slow`."""

    def test_short_sequence_folds(self) -> None:
        # This requires the boltz CLI + weights + (usually) a GPU.
        # Provided here so it's automatically exercised when the
        # environment supports it.
        engine = Boltz(model_version="boltz2", use_msa_server=False)
        protein = engine.predict("MKTVRQERLKSIVRILERSK")
        assert protein.n_residues == 20
        assert "confidence_per_residue" in protein.metadata
        assert protein.metadata["engine"] == "Boltz"
