"""Tests for the AMBER MD engine wrapper.

AmberTools (tleap, sander) is rarely available in CI — it's not pip-
installable, the conda install is heavy, and pmemd requires a paid
license. So the test strategy mirrors test_gromacs.py:

- Constructor validation and error paths run in pure Python.
- Source-inspection regression tests verify the wrapper attaches
  Provenance at the right steps and uses the documented templates.
- Subprocess-driven tests mock `_run_subprocess` at the engine's
  single seam so the pipeline logic can be exercised without real
  binaries.
- One end-to-end test is skip-marked for when AmberTools is on PATH.

Together these catch the things that actually break: missing
imports, wrong subprocess args, wrong file paths, mis-typed
Provenance, mis-named Simulation handles. The thing they don't
catch is "tleap rejects this PDB" — that needs a real binary.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from molforge.core import AtomArray, Protein
from molforge.core import metadata_keys as mk
from molforge.core.provenance import Provenance
from molforge.md import MDEngineNotInstalledError, Simulation
from molforge.wrappers.md import AMBER

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _tiny_protein() -> Protein:
    """A two-atom Protein. Real AMBER would refuse it; for testing
    the wrapper's plumbing it's fine — the binary path is mocked."""
    arr = AtomArray(2)
    arr.coords[:] = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float32)
    arr.element[:] = ["N", "C"]
    arr.atom_name[:] = ["N", "CA"]
    arr.residue_name[:] = "ALA"
    arr.residue_id[:] = 1
    arr.chain_id[:] = "A"
    return Protein(arr, name="test_protein")


# ---------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------


class TestConstruction:
    def test_defaults(self) -> None:
        engine = AMBER()
        assert engine.tleap_executable == "tleap"
        assert engine.sander_executable == "sander"
        assert engine.pmemd_executable == "pmemd"
        assert engine.water_model == "tip3p"
        assert engine.box_buffer_a == 10.0
        assert engine.verbose is False

    def test_custom_executables(self) -> None:
        engine = AMBER(
            tleap_executable="/opt/amber/bin/tleap",
            sander_executable="/opt/amber/bin/sander",
            pmemd_executable="/opt/amber/bin/pmemd.cuda",
        )
        assert engine.pmemd_executable == "/opt/amber/bin/pmemd.cuda"

    def test_vacuum_water_model(self) -> None:
        engine = AMBER(water_model="none")
        assert engine.water_model == "none"

    def test_invalid_water_model(self) -> None:
        with pytest.raises(ValueError, match="unknown water_model"):
            AMBER(water_model="bizarre")

    def test_invalid_box_buffer(self) -> None:
        with pytest.raises(ValueError, match="box_buffer_a"):
            AMBER(box_buffer_a=0.0)
        with pytest.raises(ValueError, match="box_buffer_a"):
            AMBER(box_buffer_a=-1.0)

    def test_construction_does_not_resolve_binaries(self) -> None:
        """Construction is cheap: no filesystem touching, so an
        AMBER() instance is creatable even where AmberTools isn't
        installed. The binaries are resolved lazily inside
        prepare/minimize/run."""
        engine = AMBER(
            tleap_executable="/nonexistent/tleap",
            sander_executable="/nonexistent/sander",
        )
        # No exception raised by __init__ even though the paths
        # don't exist.
        assert engine.tleap_executable == "/nonexistent/tleap"


# ---------------------------------------------------------------------
# Force-field and water-model resolution
# ---------------------------------------------------------------------


class TestForceFieldValidation:
    def test_prepare_rejects_unknown_force_field(self) -> None:
        with pytest.raises(ValueError, match="unknown force_field"):
            AMBER().prepare(_tiny_protein(), force_field="nonsense_ff")

    def test_prepare_without_tleap_raises(self) -> None:
        with pytest.raises(MDEngineNotInstalledError, match="tleap"):
            AMBER(tleap_executable="/nonexistent/tleap").prepare(_tiny_protein())

    def test_error_points_at_openmm_fallback(self) -> None:
        """The friendly error tells the user about OpenMM as an
        alternative — so a user who tried AMBER first doesn't think
        they need to install AmberTools to do MD."""
        try:
            AMBER(tleap_executable="/nonexistent/tleap").prepare(_tiny_protein())
        except MDEngineNotInstalledError as e:
            assert "OpenMM" in str(e)
        else:
            raise AssertionError("Expected MDEngineNotInstalledError")


# ---------------------------------------------------------------------
# Subprocess seam (mocked)
# ---------------------------------------------------------------------


class TestSubprocessSeam:
    """Exercise the pipeline logic with `_run_subprocess` mocked.
    The mock lets us assert what AMBER would call without needing
    real binaries."""

    def _make_engine_with_mocked_binaries(self) -> AMBER:
        """Return an AMBER engine where shutil.which returns plausible
        paths so the binary-resolution checks pass."""
        return AMBER()

    @patch("molforge.wrappers.md.amber.shutil.which")
    @patch.object(AMBER, "_run_subprocess")
    def test_prepare_invokes_tleap(
        self,
        mock_run: MagicMock,
        mock_which: MagicMock,
        tmp_path: Path,
    ) -> None:
        """prepare() invokes tleap with the right script."""
        mock_which.return_value = "/usr/bin/tleap"

        # Side effect: when tleap "runs", touch its output files so
        # the post-condition check (prmtop + inpcrd must exist)
        # passes. We also need the .inpcrd reader to work — mock
        # mdtraj.load for that.
        def fake_tleap(*args: Any, **kwargs: Any) -> None:
            cwd = kwargs["cwd"]
            (cwd / "system.prmtop").write_text("fake prmtop")
            (cwd / "system.inpcrd").write_text("fake inpcrd")

        mock_run.side_effect = fake_tleap

        with patch(
            "molforge.wrappers.md.amber._read_inpcrd_coordinates",
            return_value=np.zeros((2, 3), dtype=np.float32),
        ):
            engine = self._make_engine_with_mocked_binaries()
            sim = engine.prepare(_tiny_protein(), force_field="ff14SB")

        # Verify tleap was called once.
        assert mock_run.call_count == 1
        call_args = mock_run.call_args
        cmd = call_args[0][0]  # first positional arg is the cmd list
        assert cmd[0] == "/usr/bin/tleap"
        assert "-f" in cmd
        assert "tleap.in" in cmd

        # The Simulation has the right shape.
        assert isinstance(sim, Simulation)
        assert sim.force_field == "ff14SB"
        assert isinstance(sim.engine_handle, Path)
        assert sim.engine_handle.is_dir()

        # Run dir contains the generated tleap script (and we can
        # peek at it to verify the right template was used).
        tleap_script = (sim.engine_handle / "tleap.in").read_text()
        assert "source leaprc.protein.ff14SB" in tleap_script
        assert "source leaprc.water.tip3p" in tleap_script
        assert "solvateBox mol TIP3PBOX" in tleap_script

    @patch("molforge.wrappers.md.amber.shutil.which")
    @patch.object(AMBER, "_run_subprocess")
    def test_prepare_vacuum_skips_solvate(
        self,
        mock_run: MagicMock,
        mock_which: MagicMock,
    ) -> None:
        """water_model='none' uses the vacuum template — no solvate
        directive should appear in the tleap script."""
        mock_which.return_value = "/usr/bin/tleap"

        def fake_tleap(*args: Any, **kwargs: Any) -> None:
            cwd = kwargs["cwd"]
            (cwd / "system.prmtop").write_text("fake")
            (cwd / "system.inpcrd").write_text("fake")

        mock_run.side_effect = fake_tleap

        with patch(
            "molforge.wrappers.md.amber._read_inpcrd_coordinates",
            return_value=np.zeros((2, 3), dtype=np.float32),
        ):
            sim = AMBER(water_model="none").prepare(_tiny_protein())

        tleap_script = (sim.engine_handle / "tleap.in").read_text()
        assert "solvateBox" not in tleap_script
        assert "leaprc.water" not in tleap_script
        # ff14SB is still loaded.
        assert "source leaprc.protein.ff14SB" in tleap_script

    @patch("molforge.wrappers.md.amber.shutil.which")
    @patch.object(AMBER, "_run_subprocess")
    def test_minimize_invokes_sander(
        self,
        mock_run: MagicMock,
        mock_which: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_which.return_value = "/usr/bin/sander"

        # Build a Simulation that looks like it came from prepare.
        sim = Simulation(
            topology=_tiny_protein(),
            coordinates=np.zeros((2, 3), dtype=np.float32),
            force_field="ff14SB",
            temperature=300.0,
            timestep=0.002,
            engine_handle=tmp_path,
            metadata={
                mk.PROVENANCE: Provenance.from_engine(
                    engine="AMBER.prepare",
                    parameters={"force_field": "ff14SB"},
                    inputs={"protein": "test"},
                ),
            },
        )
        # The handle must be a real directory.
        (tmp_path / "system.prmtop").write_text("fake")
        (tmp_path / "system.inpcrd").write_text("fake")

        def fake_sander(*args: Any, **kwargs: Any) -> None:
            cwd = kwargs["cwd"]
            (cwd / "min.rst7").write_text("fake rst")

        mock_run.side_effect = fake_sander

        with patch(
            "molforge.wrappers.md.amber._read_rst_coordinates",
            return_value=np.zeros((2, 3), dtype=np.float32),
        ):
            engine = AMBER()
            sim2 = engine.minimize(sim, max_iterations=500)

        # sander invoked.
        assert mock_run.call_count == 1
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/usr/bin/sander"
        assert "-O" in cmd
        assert "min.in" in cmd

        # min.in script contains the right maxcyc.
        script = (tmp_path / "min.in").read_text()
        assert "maxcyc=500" in script

        # ncyc heuristic: half of max_iter, capped at 500. 500/2=250.
        assert "ncyc=250" in script

        # Provenance chained.
        prov = sim2.metadata[mk.PROVENANCE]
        assert prov.engine == "AMBER.minimize"
        assert prov.parent is not None
        assert prov.parent.engine == "AMBER.prepare"
        assert prov.parameters["max_iterations"] == 500

    @patch("molforge.wrappers.md.amber.shutil.which")
    @patch.object(AMBER, "_run_subprocess")
    def test_minimize_ncyc_capped_at_500(
        self,
        mock_run: MagicMock,
        mock_which: MagicMock,
        tmp_path: Path,
    ) -> None:
        """For very long minimisations the ncyc heuristic must cap
        so the steepest-descent prelude doesn't dominate."""
        mock_which.return_value = "/usr/bin/sander"
        sim = Simulation(
            topology=_tiny_protein(),
            coordinates=np.zeros((2, 3), dtype=np.float32),
            force_field="ff14SB",
            temperature=300.0,
            timestep=0.002,
            engine_handle=tmp_path,
            metadata={},
        )
        (tmp_path / "system.prmtop").write_text("fake")

        def fake_sander(*args: Any, **kwargs: Any) -> None:
            (kwargs["cwd"] / "min.rst7").write_text("fake")

        mock_run.side_effect = fake_sander

        with patch(
            "molforge.wrappers.md.amber._read_rst_coordinates",
            return_value=np.zeros((2, 3), dtype=np.float32),
        ):
            AMBER().minimize(sim, max_iterations=5000)

        script = (tmp_path / "min.in").read_text()
        # 5000 / 2 = 2500, capped to 500.
        assert "ncyc=500" in script


# ---------------------------------------------------------------------
# Run dir resolution and error paths
# ---------------------------------------------------------------------


class TestRunDirValidation:
    def test_minimize_without_run_dir_raises(self) -> None:
        """A Simulation not produced by AMBER.prepare lacks the run
        directory handle. Subsequent calls must fail clearly."""
        bogus = Simulation(
            topology=_tiny_protein(),
            coordinates=np.zeros((2, 3), dtype=np.float32),
            force_field="ff14SB",
            temperature=300.0,
            timestep=0.002,
            engine_handle="not_a_path",  # type: ignore[arg-type]
            metadata={},
        )
        # Mock the shutil.which side so we get past the binary
        # resolution and hit the run-dir validation.
        with (
            patch(
                "molforge.wrappers.md.amber.shutil.which",
                return_value="/usr/bin/sander",
            ),
            pytest.raises(ValueError, match="AMBER run directory"),
        ):
            AMBER().minimize(bogus)

    def test_run_picks_pmemd_when_available(self, tmp_path: Path) -> None:
        """run() resolves pmemd first; if absent, falls back to
        sander. The wrapper returns which binary was used via the
        step name."""

        def fake_which(name: str) -> str | None:
            if name == "pmemd":
                return "/usr/bin/pmemd"
            if name == "sander":
                return "/usr/bin/sander"
            return None

        with patch("molforge.wrappers.md.amber.shutil.which", side_effect=fake_which):
            engine = AMBER()
            binary, step = engine._resolve_md_binary()
        assert binary == "/usr/bin/pmemd"
        assert "pmemd" in step

    def test_run_falls_back_to_sander_when_no_pmemd(self) -> None:
        def fake_which(name: str) -> str | None:
            if name == "pmemd":
                return None
            if name == "sander":
                return "/usr/bin/sander"
            return None

        with patch("molforge.wrappers.md.amber.shutil.which", side_effect=fake_which):
            binary, step = AMBER()._resolve_md_binary()
        assert "sander" in binary
        assert "sander" in step

    def test_run_raises_when_neither_binary_available(self) -> None:
        with (
            patch("molforge.wrappers.md.amber.shutil.which", return_value=None),
            pytest.raises(MDEngineNotInstalledError, match="Neither pmemd"),
        ):
            AMBER()._resolve_md_binary()


# ---------------------------------------------------------------------
# Source-inspection regression net
# ---------------------------------------------------------------------


class TestSourceInspection:
    """Tighter than nothing for the things mocking can't fully cover.
    Mirrors the GROMACS-side wiring tests."""

    def test_module_has_three_pipeline_engine_strings(self) -> None:
        """The three Provenance step engine strings must be present
        in the source — catches future refactors that drop one."""
        from molforge.wrappers.md import amber

        src = Path(amber.__file__).read_text()
        assert 'engine="AMBER.prepare"' in src
        assert 'engine="AMBER.minimize"' in src
        assert 'engine="AMBER.run"' in src

    def test_parent_provenance_helper_used(self) -> None:
        """Every chained step uses _parent_provenance(...) to extract
        the parent — not the raw .get() which would skip the type
        narrowing."""
        from molforge.wrappers.md import amber

        src = Path(amber.__file__).read_text()
        assert "_parent_provenance(" in src

    def test_subprocess_routes_through_seam(self) -> None:
        """All subprocess invocations go through _run_subprocess so
        tests can mock at a single point — catches a regression
        where someone adds a direct subprocess.run call."""
        from molforge.wrappers.md import amber

        src = Path(amber.__file__).read_text()
        # subprocess.run appears exactly once — inside _run_subprocess.
        # Other invocations should call self._run_subprocess instead.
        assert src.count("subprocess.run(") == 1


# ---------------------------------------------------------------------
# End-to-end (skipped without real binary)
# ---------------------------------------------------------------------


@pytest.mark.skipif(
    shutil.which("tleap") is None or shutil.which("sander") is None,
    reason="AmberTools (tleap + sander) not installed",
)
class TestRealAmber:
    """Exercises the full AMBER pipeline end-to-end. Skipped when
    AmberTools isn't available."""

    def test_prepare_minimize_run_chain(self, tmp_path: Path) -> None:
        from molforge.io import fetch
        from molforge.prep import prepare_for_md

        protein = fetch("1AKE")
        ready = prepare_for_md(protein)

        engine = AMBER(water_model="tip3p")
        sim = engine.prepare(ready, force_field="ff14SB")
        sim = engine.minimize(sim, max_iterations=50)
        traj = engine.run(sim, n_steps=20, save_every=5)

        # Provenance chain reads correctly.
        engines = [s.engine for s in traj.metadata[mk.PROVENANCE].chain()]
        # Last three are AMBER.prepare / minimize / run, in order.
        # (The prep functions add 4 steps earlier in the chain.)
        assert engines[-3:] == [
            "AMBER.prepare",
            "AMBER.minimize",
            "AMBER.run",
        ]
        assert traj.n_frames > 0
