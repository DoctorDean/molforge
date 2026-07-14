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
        assert f"inference.input_pdb={Path('/some/path/target.pdb')}" in joined
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


# Minimal valid PDB used by the mocked-subprocess tests below.
_MINI_PDB = (
    "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00 50.00           N\n"
    "ATOM      2  CA  GLY A   1       1.000   0.000   0.000  1.00 50.00           C\n"
    "ATOM      3  C   GLY A   1       2.000   0.000   0.000  1.00 50.00           C\n"
    "END\n"
)


class TestRunCli:
    """`_run_cli` is the subprocess-driving seam — exercised with a
    mocked ``subprocess.run`` so neither RFdiffusion nor torch is
    needed."""

    @staticmethod
    def _fake_install(tmp_path: Path) -> Path:
        rfdir = tmp_path / "RFdiffusion"
        (rfdir / "scripts").mkdir(parents=True)
        (rfdir / "scripts" / "run_inference.py").write_text("# placeholder\n")
        return rfdir

    @staticmethod
    def _output_dir_from_cmd(cmd: list[str]) -> Path:
        """RFdiffusion's `inference.output_prefix=<dir>/design` Hydra
        arg tells the mock where the wrapper expects the PDBs."""
        for tok in cmd:
            if tok.startswith("inference.output_prefix="):
                return Path(tok.split("=", 1)[1]).parent
        raise AssertionError("output_prefix not found in command")

    def test_invokes_subprocess_and_parses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rfdir = self._fake_install(tmp_path)
        captured: dict[str, object] = {}

        def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            captured["cmd"] = cmd
            out_dir = self._output_dir_from_cmd(cmd)
            (out_dir / "design_0.pdb").write_text(_MINI_PDB)
            (out_dir / "design_1.pdb").write_text(_MINI_PDB)
            return None

        monkeypatch.setattr("subprocess.run", fake_run)
        designs = RFdiffusion(num_designs=2)._run_cli(
            rfdiffusion_dir=rfdir,
            length=40,
            target_pdb=None,
            contigs=None,
            hotspot_residues=None,
            symmetry=None,
            extra_hydra_args=None,
            timeout=None,
        )
        assert len(designs) == 2
        for i, d in enumerate(designs):
            assert d.metadata["engine"] == "RFdiffusion"
            assert d.metadata["design_index"] == i
        cmd = captured["cmd"]
        assert str(rfdir / "scripts" / "run_inference.py") in cmd
        # length=40 becomes a contigmap.contigs Hydra arg
        assert any("contigmap.contigs=[40-40]" in tok for tok in cmd)

    def test_no_pdb_output_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        rfdir = self._fake_install(tmp_path)

        def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            return None  # writes nothing — RFdiffusion silently failed

        monkeypatch.setattr("subprocess.run", fake_run)
        with pytest.raises(RuntimeError, match="no PDB output"):
            RFdiffusion()._run_cli(
                rfdiffusion_dir=rfdir,
                length=40,
                target_pdb=None,
                contigs=None,
                hotspot_residues=None,
                symmetry=None,
                extra_hydra_args=None,
                timeout=None,
            )

    def test_subprocess_failure_becomes_runtime_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess

        rfdir = self._fake_install(tmp_path)

        def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr="diffusion blew up")

        monkeypatch.setattr("subprocess.run", fake_run)
        with pytest.raises(RuntimeError, match="RFdiffusion failed"):
            RFdiffusion()._run_cli(
                rfdiffusion_dir=rfdir,
                length=40,
                target_pdb=None,
                contigs=None,
                hotspot_residues=None,
                symmetry=None,
                extra_hydra_args=None,
                timeout=None,
            )

    def test_generate_resolves_install_then_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The public entry point: generate() resolves the install dir
        # from RFDIFFUSION_HOME, then delegates to _run_cli.
        rfdir = self._fake_install(tmp_path)
        monkeypatch.setenv("RFDIFFUSION_HOME", str(rfdir))

        def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            out_dir = self._output_dir_from_cmd(cmd)
            (out_dir / "design_0.pdb").write_text(_MINI_PDB)
            return None

        monkeypatch.setattr("subprocess.run", fake_run)
        designs = RFdiffusion().generate(length=40)
        assert len(designs) == 1
        assert designs[0].metadata["engine"] == "RFdiffusion"

    def test_generate_passes_contigs_and_symmetry_through(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rfdir = self._fake_install(tmp_path)
        monkeypatch.setenv("RFDIFFUSION_HOME", str(rfdir))
        captured: dict[str, object] = {}

        def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            captured["cmd"] = cmd
            out_dir = self._output_dir_from_cmd(cmd)
            (out_dir / "design_0.pdb").write_text(_MINI_PDB)
            return None

        monkeypatch.setattr("subprocess.run", fake_run)
        RFdiffusion().generate(
            contigs=["10-40/A20-35/10-40"],
            symmetry="c2",
        )
        cmd = captured["cmd"]
        assert any("contigmap.contigs=" in tok for tok in cmd)
        assert any("inference.symmetry=c2" in tok for tok in cmd)
