"""Tests for the RFdiffusion wrapper.

These don't require RFdiffusion (or torch) to be installed. They
exercise:
  - Construction with various parameter combinations
  - Installation-path resolution from env vars and explicit args
  - Hydra arg building in isolation
  - Output parsing in isolation
  - Missing-dependency error paths

End-to-end generation requires the real engine and is gated on
``RFDIFFUSION_HOME`` being set (so it's effectively always skipped in CI).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molforge.generative import GenerativeEngineNotInstalledError
from molforge.wrappers.generative import RFdiffusion


class TestConstruction:
    def test_defaults(self) -> None:
        engine = RFdiffusion()
        assert engine.name == "RFdiffusion"
        assert engine.num_designs == 1
        assert engine.diffusion_steps == 50
        assert engine.config_name == "base"
        assert engine.rfdiffusion_dir is None

    def test_custom_settings(self) -> None:
        engine = RFdiffusion(
            rfdiffusion_dir="/custom/path",
            num_designs=10,
            diffusion_steps=30,
            device="cuda",
            config_name="symmetry",
        )
        assert engine.rfdiffusion_dir == Path("/custom/path")
        assert engine.num_designs == 10
        assert engine.diffusion_steps == 30
        assert engine.device == "cuda"
        assert engine.config_name == "symmetry"

    def test_construction_doesnt_resolve_install(self) -> None:
        """Construction shouldn't probe filesystem or env vars."""
        # If construction resolved the install dir, the test below
        # would fail in CI where RFDIFFUSION_HOME isn't set.
        RFdiffusion()


class TestInstallationResolution:
    def test_missing_install_clear_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RFDIFFUSION_HOME", raising=False)
        engine = RFdiffusion()
        with pytest.raises(GenerativeEngineNotInstalledError, match="RFdiffusion not found"):
            engine._resolve_rfdiffusion_dir()

    def test_env_var_used(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        # Create a fake RFdiffusion install
        fake_install = tmp_path / "RFdiffusion"
        (fake_install / "scripts").mkdir(parents=True)
        (fake_install / "scripts" / "run_inference.py").write_text("# placeholder\n")
        monkeypatch.setenv("RFDIFFUSION_HOME", str(fake_install))
        engine = RFdiffusion()
        assert engine._resolve_rfdiffusion_dir() == fake_install

    def test_explicit_dir_overrides_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        env_install = tmp_path / "EnvInstall"
        (env_install / "scripts").mkdir(parents=True)
        (env_install / "scripts" / "run_inference.py").write_text("# placeholder\n")

        explicit_install = tmp_path / "Explicit"
        (explicit_install / "scripts").mkdir(parents=True)
        (explicit_install / "scripts" / "run_inference.py").write_text("# placeholder\n")

        monkeypatch.setenv("RFDIFFUSION_HOME", str(env_install))
        engine = RFdiffusion(rfdiffusion_dir=explicit_install)
        assert engine._resolve_rfdiffusion_dir() == explicit_install

    def test_explicit_dir_without_script_errors(self, tmp_path: Path) -> None:
        engine = RFdiffusion(rfdiffusion_dir=tmp_path)  # has no scripts/
        with pytest.raises(GenerativeEngineNotInstalledError, match=r"scripts/run_inference\.py"):
            engine._resolve_rfdiffusion_dir()


class TestHydraArgBuilding:
    def test_unconditional_length(self) -> None:
        engine = RFdiffusion(num_designs=4)
        args = engine._build_hydra_args(
            output_prefix=Path("/tmp/design"),
            length=100,
        )
        joined = " ".join(args)
        assert "inference.num_designs=4" in joined
        assert "diffuser.T=50" in joined
        assert "contigmap.contigs=[100-100]" in joined

    def test_motif_scaffolding_contigs(self) -> None:
        engine = RFdiffusion(num_designs=1)
        args = engine._build_hydra_args(
            output_prefix=Path("/tmp/design"),
            target_pdb="/some/path/target.pdb",
            contigs=["10-40/A20-35/10-40"],
        )
        joined = " ".join(args)
        assert "inference.input_pdb=/some/path/target.pdb" in joined
        assert "contigmap.contigs=[10-40/A20-35/10-40]" in joined

    def test_hotspot_residues(self) -> None:
        engine = RFdiffusion(num_designs=1)
        args = engine._build_hydra_args(
            output_prefix=Path("/tmp/design"),
            length=120,
            hotspot_residues=["A32", "A33", "A34"],
        )
        joined = " ".join(args)
        assert "ppi.hotspot_res=[A32,A33,A34]" in joined

    def test_symmetry(self) -> None:
        engine = RFdiffusion(num_designs=1, config_name="symmetry")
        args = engine._build_hydra_args(
            output_prefix=Path("/tmp/design"),
            length=360,
            symmetry="tetrahedral",
        )
        joined = " ".join(args)
        assert "--config-name=symmetry" in joined
        assert "inference.symmetry=tetrahedral" in joined

    def test_extra_args(self) -> None:
        engine = RFdiffusion(num_designs=1)
        args = engine._build_hydra_args(
            output_prefix=Path("/tmp/design"),
            length=100,
            extra={"diffuser.partial_T": "20"},
        )
        joined = " ".join(args)
        assert "diffuser.partial_T=20" in joined


class TestGenerateValidation:
    def test_length_and_contigs_mutually_exclusive(self) -> None:
        engine = RFdiffusion(rfdiffusion_dir="/nonexistent")
        with pytest.raises(ValueError, match="not both"):
            engine.generate(length=100, contigs=["100"])

    def test_needs_at_least_one_spec(self) -> None:
        engine = RFdiffusion(rfdiffusion_dir="/nonexistent")
        with pytest.raises(ValueError, match="at least one"):
            engine.generate()


class TestOutputParsing:
    def test_no_outputs_raises(self, tmp_path: Path) -> None:
        engine = RFdiffusion()
        with pytest.raises(RuntimeError, match="no PDB output"):
            engine._parse_outputs(tmp_path, source_args={})

    def test_parses_pdb_outputs(self, tmp_path: Path) -> None:
        # Drop in two minimal PDBs
        pdb_text = (
            "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00 50.00           N\n"
            "ATOM      2  CA  GLY A   1       1.000   0.000   0.000  1.00 50.00           C\n"
            "ATOM      3  C   GLY A   1       2.000   0.000   0.000  1.00 50.00           C\n"
            "END\n"
        )
        (tmp_path / "design_0.pdb").write_text(pdb_text)
        (tmp_path / "design_1.pdb").write_text(pdb_text)
        engine = RFdiffusion()
        designs = engine._parse_outputs(
            tmp_path,
            source_args={"length": 1, "diffusion_steps": 50, "num_designs": 2},
        )
        assert len(designs) == 2
        for i, d in enumerate(designs):
            assert d.metadata["engine"] == "RFdiffusion"
            assert d.metadata["design_index"] == i
            assert d.metadata["source_args"]["length"] == 1
