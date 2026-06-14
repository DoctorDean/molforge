"""Tests for the GROMACS wrapper.

These do not require a GROMACS install. They exercise:
  - Construction with parameter validation
  - `gmx`-executable resolution
  - The .gro / multi-model-PDB / .xvg parsing helpers
  - The prepare / minimize / run pipeline, driven by a mocked
    subprocess.run that writes the files each `gmx` step would produce

A real GROMACS run is an integration concern and is out of scope for
the unit suite.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from molforge.core import AtomArray, Protein
from molforge.md import MDEngineNotInstalledError, Simulation, Trajectory
from molforge.wrappers.md import GROMACS
from molforge.wrappers.md.gromacs import (
    _read_gro_coordinates,
    _read_multimodel_pdb_coordinates,
    _read_xvg_column,
)

# --- fixtures / sample data ------------------------------------------

_SAMPLE_GRO = """molforge test system
 3
    1ALA      N    1   1.000   2.000   3.000
    1ALA     CA    2   1.500   2.500   3.500
    1ALA      C    3   2.000   3.000   4.000
   5.00000   5.00000   5.00000
"""

_SAMPLE_FRAMES_PDB = """MODEL        1
ATOM      1  N   ALA A   1      10.000  20.000  30.000  1.00  0.00           N
ATOM      2  CA  ALA A   1      15.000  25.000  35.000  1.00  0.00           C
ATOM      3  C   ALA A   1      20.000  30.000  40.000  1.00  0.00           C
ENDMDL
MODEL        2
ATOM      1  N   ALA A   1      10.500  20.500  30.500  1.00  0.00           N
ATOM      2  CA  ALA A   1      15.500  25.500  35.500  1.00  0.00           C
ATOM      3  C   ALA A   1      20.500  30.500  40.500  1.00  0.00           C
ENDMDL
"""

_SAMPLE_XVG = """# This file was created by gmx energy
@    title "GROMACS Energies"
@    xaxis  label "Time (ps)"
    0.000000  -1234.567
    1.000000  -1250.123
    2.000000  -1255.890
"""


def _topology(n: int = 3) -> Protein:
    arr = AtomArray(n)
    arr.element[:] = "C"
    arr.atom_name[:] = "CA"
    arr.residue_name[:] = "ALA"
    arr.residue_id[:] = np.arange(1, n + 1)
    arr.chain_id[:] = "A"
    return Protein(arr)


# --- construction -----------------------------------------------------


class TestConstruction:
    def test_defaults(self) -> None:
        engine = GROMACS()
        assert engine.name == "GROMACS"
        assert engine.gmx_executable == "gmx"
        assert engine.water_model == "none"
        assert engine.box_margin == pytest.approx(1.0)
        assert engine.box_type == "cubic"
        assert engine.verbose is False

    def test_custom_settings(self) -> None:
        engine = GROMACS(
            gmx_executable="gmx_mpi",
            water_model="tip3p",
            box_margin=1.5,
            box_type="dodecahedron",
            verbose=True,
        )
        assert engine.gmx_executable == "gmx_mpi"
        assert engine.water_model == "tip3p"
        assert engine.box_margin == pytest.approx(1.5)
        assert engine.box_type == "dodecahedron"
        assert engine.verbose is True

    def test_invalid_water_model(self) -> None:
        with pytest.raises(ValueError, match="water_model"):
            GROMACS(water_model="not-a-water-model")

    def test_invalid_box_margin(self) -> None:
        with pytest.raises(ValueError, match="box_margin"):
            GROMACS(box_margin=0.0)

    def test_construction_does_not_resolve_gmx(self) -> None:
        """Constructing must not probe PATH for the gmx binary."""
        GROMACS(gmx_executable="definitely-not-installed-xyz")


# --- gmx resolution ---------------------------------------------------


class TestGmxResolution:
    def test_missing_gmx_raises_clear_error(self) -> None:
        engine = GROMACS()
        with (
            patch("shutil.which", return_value=None),
            pytest.raises(MDEngineNotInstalledError, match="GROMACS"),
        ):
            engine._require_gmx()

    def test_resolved_gmx_path_returned(self) -> None:
        engine = GROMACS()
        with patch("shutil.which", return_value="/usr/bin/gmx"):
            assert engine._require_gmx() == "/usr/bin/gmx"

    def test_missing_gmx_error_points_at_openmm(self) -> None:
        engine = GROMACS()
        with (
            patch("shutil.which", return_value=None),
            pytest.raises(MDEngineNotInstalledError, match="OpenMM"),
        ):
            engine._require_gmx()


# --- .gro parsing -----------------------------------------------------


class TestGroParsing:
    def test_parses_coordinates_in_angstrom(self) -> None:
        # .gro is in nm; the parser scales to Å (x10).
        coords = _read_gro_coordinates(_write(_SAMPLE_GRO, ".gro"))
        assert coords.shape == (3, 3)
        assert tuple(coords[0]) == pytest.approx((10.0, 20.0, 30.0))
        assert tuple(coords[2]) == pytest.approx((20.0, 30.0, 40.0))

    def test_too_short_raises(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            _read_gro_coordinates(_write("just one line\n", ".gro"))

    def test_unparseable_count_raises(self) -> None:
        bad = "title\n  NOT-A-NUMBER\n    1ALA      N    1   1.0   2.0   3.0\n"
        with pytest.raises(ValueError, match="atom count"):
            _read_gro_coordinates(_write(bad, ".gro"))

    def test_truncated_atom_block_raises(self) -> None:
        bad = "title\n 5\n    1ALA      N    1   1.000   2.000   3.000\n"
        with pytest.raises(ValueError, match="truncated"):
            _read_gro_coordinates(_write(bad, ".gro"))


# --- multi-model PDB parsing -----------------------------------------


class TestMultiModelPdbParsing:
    def test_parses_all_frames(self) -> None:
        coords = _read_multimodel_pdb_coordinates(_SAMPLE_FRAMES_PDB)
        assert coords.shape == (2, 3, 3)
        assert tuple(coords[0, 0]) == pytest.approx((10.0, 20.0, 30.0))
        assert tuple(coords[1, 2]) == pytest.approx((20.5, 30.5, 40.5))

    def test_single_frame_without_model_records(self) -> None:
        single = "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N\n"
        coords = _read_multimodel_pdb_coordinates(single)
        assert coords.shape == (1, 1, 3)

    def test_empty_text_yields_empty_array(self) -> None:
        coords = _read_multimodel_pdb_coordinates("REMARK nothing here\n")
        assert coords.shape[0] == 0


# --- .xvg parsing -----------------------------------------------------


class TestXvgParsing:
    def test_reads_value_column(self) -> None:
        values = _read_xvg_column(_SAMPLE_XVG)
        assert values is not None
        assert values.shape == (3,)
        assert values[0] == pytest.approx(-1234.567)
        assert values[2] == pytest.approx(-1255.890)

    def test_comment_and_metadata_lines_skipped(self) -> None:
        values = _read_xvg_column("# c\n@ s0 legend\n5.0  -1.0\n")
        assert values is not None
        assert values.tolist() == pytest.approx([-1.0])

    def test_no_data_returns_none(self) -> None:
        assert _read_xvg_column("# only a comment\n@ only metadata\n") is None


# --- the prepare / minimize / run pipeline ---------------------------


def _gmx_step_mock(extra: dict[str, str] | None = None):  # type: ignore[no-untyped-def]
    """Build a subprocess.run replacement that writes the output files
    each `gmx` subcommand is expected to produce.

    Each call inspects the gmx subcommand (cmd[1]) and the cwd, and
    writes plausible output files there, so the wrapper's own parsing
    and bookkeeping logic runs for real.
    """

    def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        sub = cmd[1]
        cwd = Path(kwargs["cwd"])
        if sub == "pdb2gmx":
            (cwd / "conf.gro").write_text(_SAMPLE_GRO)
            (cwd / "topol.top").write_text("; topology\n")
        elif sub == "editconf":
            (cwd / "boxed.gro").write_text(_SAMPLE_GRO)
        elif sub == "solvate":
            (cwd / "solvated.gro").write_text(_SAMPLE_GRO)
        elif sub == "grompp":
            out = cmd[cmd.index("-o") + 1]
            (cwd / out).write_text("fake tpr")
        elif sub == "mdrun":
            deffnm = cmd[cmd.index("-deffnm") + 1]
            (cwd / f"{deffnm}.gro").write_text(_SAMPLE_GRO)
            if deffnm == "md":
                (cwd / "md.xtc").write_text("fake xtc")
                (cwd / "md.edr").write_text("fake edr")
        elif sub == "trjconv":
            (cwd / "md_frames.pdb").write_text(_SAMPLE_FRAMES_PDB)
        elif sub == "energy":
            (cwd / "energy.xvg").write_text(_SAMPLE_XVG)
        for name, content in (extra or {}).items():
            (cwd / name).write_text(content)

        class _Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Result()

    return fake_run


class TestPreparePipeline:
    def test_prepare_runs_and_returns_simulation(self) -> None:
        engine = GROMACS()
        with (
            patch("shutil.which", return_value="/usr/bin/gmx"),
            patch("subprocess.run", side_effect=_gmx_step_mock()),
        ):
            sim = engine.prepare(_topology(), force_field="amber99sb-ildn")
        try:
            assert isinstance(sim, Simulation)
            assert sim.force_field == "amber99sb-ildn"
            assert sim.coordinates.shape == (3, 3)
            # The run directory is carried as the engine handle.
            assert isinstance(sim.engine_handle, Path)
            assert sim.engine_handle.is_dir()
            assert sim.metadata["run_dir"] == str(sim.engine_handle)
        finally:
            shutil.rmtree(sim.engine_handle, ignore_errors=True)

    def test_prepare_rejects_unknown_force_field(self) -> None:
        engine = GROMACS()
        with (
            patch("shutil.which", return_value="/usr/bin/gmx"),
            pytest.raises(ValueError, match="force_field"),
        ):
            engine.prepare(_topology(), force_field="not-a-force-field")

    def test_prepare_without_gmx_raises(self) -> None:
        engine = GROMACS()
        with (
            patch("shutil.which", return_value=None),
            pytest.raises(MDEngineNotInstalledError),
        ):
            engine.prepare(_topology(), force_field="amber99sb")

    def test_solvate_step_runs_when_water_requested(self) -> None:
        engine = GROMACS(water_model="tip3p")
        calls: list[str] = []

        base = _gmx_step_mock()

        def tracking_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            calls.append(cmd[1])
            return base(cmd, **kwargs)

        with (
            patch("shutil.which", return_value="/usr/bin/gmx"),
            patch("subprocess.run", side_effect=tracking_run),
        ):
            sim = engine.prepare(_topology(), force_field="amber99sb")
        try:
            assert "solvate" in calls
        finally:
            shutil.rmtree(sim.engine_handle, ignore_errors=True)

    def test_no_solvate_step_for_vacuum(self) -> None:
        engine = GROMACS(water_model="none")
        calls: list[str] = []
        base = _gmx_step_mock()

        def tracking_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            calls.append(cmd[1])
            return base(cmd, **kwargs)

        with (
            patch("shutil.which", return_value="/usr/bin/gmx"),
            patch("subprocess.run", side_effect=tracking_run),
        ):
            sim = engine.prepare(_topology(), force_field="amber99sb")
        try:
            assert "solvate" not in calls
        finally:
            shutil.rmtree(sim.engine_handle, ignore_errors=True)


class TestMinimize:
    def _prepared(self, engine: GROMACS) -> Simulation:
        with (
            patch("shutil.which", return_value="/usr/bin/gmx"),
            patch("subprocess.run", side_effect=_gmx_step_mock()),
        ):
            return engine.prepare(_topology(), force_field="amber99sb")

    def test_minimize_updates_state(self) -> None:
        engine = GROMACS()
        sim = self._prepared(engine)
        try:
            with (
                patch("shutil.which", return_value="/usr/bin/gmx"),
                patch("subprocess.run", side_effect=_gmx_step_mock()),
            ):
                out = engine.minimize(sim, max_iterations=100)
            assert out.metadata["minimized"] is True
            assert out.coordinates.shape == (3, 3)
        finally:
            shutil.rmtree(sim.engine_handle, ignore_errors=True)

    def test_minimize_without_run_dir_raises(self) -> None:
        engine = GROMACS()
        # A hand-built Simulation has no GROMACS run directory.
        sim = Simulation(
            topology=_topology(),
            coordinates=np.zeros((3, 3), dtype=np.float32),
            engine_handle=None,
        )
        with (
            patch("shutil.which", return_value="/usr/bin/gmx"),
            pytest.raises(ValueError, match="run directory"),
        ):
            engine.minimize(sim)

    def test_grompp_failure_becomes_runtime_error(self) -> None:
        import subprocess

        engine = GROMACS()
        sim = self._prepared(engine)
        try:

            def failing_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
                if cmd[1] == "grompp":
                    raise subprocess.CalledProcessError(
                        returncode=1, cmd=cmd, stderr="grompp blew up"
                    )
                return _gmx_step_mock()(cmd, **kwargs)

            with (
                patch("shutil.which", return_value="/usr/bin/gmx"),
                patch("subprocess.run", side_effect=failing_run),
                pytest.raises(RuntimeError, match="grompp"),
            ):
                engine.minimize(sim)
        finally:
            shutil.rmtree(sim.engine_handle, ignore_errors=True)


class TestRun:
    def _prepared(self, engine: GROMACS) -> Simulation:
        with (
            patch("shutil.which", return_value="/usr/bin/gmx"),
            patch("subprocess.run", side_effect=_gmx_step_mock()),
        ):
            return engine.prepare(_topology(), force_field="amber99sb")

    def test_run_produces_trajectory(self) -> None:
        engine = GROMACS()
        sim = self._prepared(engine)
        try:
            with (
                patch("shutil.which", return_value="/usr/bin/gmx"),
                patch("subprocess.run", side_effect=_gmx_step_mock()),
            ):
                traj = engine.run(sim, n_steps=10, save_every=5)
            assert isinstance(traj, Trajectory)
            assert traj.n_frames == 2
            assert traj.coordinates.shape == (2, 3, 3)
            assert traj.times is not None
            assert traj.times.shape == (2,)
            # Energies came from the mocked .xvg.
            assert traj.energies is not None
            assert traj.energies[0] == pytest.approx(-1234.567)
            assert traj.metadata["engine"] == "GROMACS"
            assert traj.metadata["n_steps"] == 10
        finally:
            shutil.rmtree(sim.engine_handle, ignore_errors=True)

    def test_run_rejects_bad_n_steps(self) -> None:
        engine = GROMACS()
        sim = self._prepared(engine)
        try:
            with (
                patch("shutil.which", return_value="/usr/bin/gmx"),
                pytest.raises(ValueError, match="n_steps"),
            ):
                engine.run(sim, n_steps=0)
        finally:
            shutil.rmtree(sim.engine_handle, ignore_errors=True)

    def test_run_rejects_bad_save_every(self) -> None:
        engine = GROMACS()
        sim = self._prepared(engine)
        try:
            with (
                patch("shutil.which", return_value="/usr/bin/gmx"),
                pytest.raises(ValueError, match="save_every"),
            ):
                engine.run(sim, n_steps=10, save_every=0)
        finally:
            shutil.rmtree(sim.engine_handle, ignore_errors=True)

    def test_run_without_run_dir_raises(self) -> None:
        engine = GROMACS()
        sim = Simulation(
            topology=_topology(),
            coordinates=np.zeros((3, 3), dtype=np.float32),
            engine_handle=None,
        )
        with (
            patch("shutil.which", return_value="/usr/bin/gmx"),
            pytest.raises(ValueError, match="run directory"),
        ):
            engine.run(sim, n_steps=10)

    def test_trajectory_without_energy_file_still_valid(self) -> None:
        """If the .edr is missing, energies are None but the trajectory
        is still returned."""
        engine = GROMACS()
        sim = self._prepared(engine)
        try:

            def no_edr_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
                result = _gmx_step_mock()(cmd, **kwargs)
                cwd = Path(kwargs["cwd"])
                # Remove the .edr that the mdrun mock just wrote.
                edr = cwd / "md.edr"
                if edr.exists():
                    edr.unlink()
                return result

            with (
                patch("shutil.which", return_value="/usr/bin/gmx"),
                patch("subprocess.run", side_effect=no_edr_run),
            ):
                traj = engine.run(sim, n_steps=10, save_every=5)
            assert traj.n_frames == 2
            assert traj.energies is None
        finally:
            shutil.rmtree(sim.engine_handle, ignore_errors=True)

    def test_mdrun_failure_becomes_runtime_error(self) -> None:
        import subprocess

        engine = GROMACS()
        sim = self._prepared(engine)
        try:

            def failing_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
                if cmd[1] == "mdrun":
                    raise subprocess.CalledProcessError(
                        returncode=1, cmd=cmd, stderr="mdrun crashed"
                    )
                return _gmx_step_mock()(cmd, **kwargs)

            with (
                patch("shutil.which", return_value="/usr/bin/gmx"),
                patch("subprocess.run", side_effect=failing_run),
                pytest.raises(RuntimeError, match="mdrun"),
            ):
                engine.run(sim, n_steps=10)
        finally:
            shutil.rmtree(sim.engine_handle, ignore_errors=True)

    def test_missing_frames_pdb_raises(self) -> None:
        """If trjconv writes no frames PDB, run() raises a clear error."""
        engine = GROMACS()
        sim = self._prepared(engine)
        try:

            def no_frames_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
                result = _gmx_step_mock()(cmd, **kwargs)
                if cmd[1] == "trjconv":
                    # Remove the frames PDB the mock just wrote.
                    fp = Path(kwargs["cwd"]) / "md_frames.pdb"
                    if fp.exists():
                        fp.unlink()
                return result

            with (
                patch("shutil.which", return_value="/usr/bin/gmx"),
                patch("subprocess.run", side_effect=no_frames_run),
                pytest.raises(RuntimeError, match="no trajectory frames"),
            ):
                engine.run(sim, n_steps=10)
        finally:
            shutil.rmtree(sim.engine_handle, ignore_errors=True)

    def test_empty_trajectory_raises(self) -> None:
        """A frames PDB with no parseable models raises a clear error."""
        engine = GROMACS()
        sim = self._prepared(engine)
        try:

            def empty_frames_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
                result = _gmx_step_mock()(cmd, **kwargs)
                if cmd[1] == "trjconv":
                    (Path(kwargs["cwd"]) / "md_frames.pdb").write_text("REMARK empty trajectory\n")
                return result

            with (
                patch("shutil.which", return_value="/usr/bin/gmx"),
                patch("subprocess.run", side_effect=empty_frames_run),
                pytest.raises(RuntimeError, match="no frames"),
            ):
                engine.run(sim, n_steps=10)
        finally:
            shutil.rmtree(sim.engine_handle, ignore_errors=True)

    def test_verbose_mode_does_not_capture_output(self) -> None:
        """In verbose mode the subprocess call passes capture_output=False."""
        engine = GROMACS(verbose=True)
        sim = self._prepared(engine)
        try:
            seen: list[bool] = []
            base = _gmx_step_mock()

            def recording_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
                seen.append(kwargs.get("capture_output", True))
                return base(cmd, **kwargs)

            with (
                patch("shutil.which", return_value="/usr/bin/gmx"),
                patch("subprocess.run", side_effect=recording_run),
            ):
                engine.run(sim, n_steps=10, save_every=5)
            # verbose=True means capture_output=False on every call.
            assert seen and not any(seen)
        finally:
            shutil.rmtree(sim.engine_handle, ignore_errors=True)


class TestGroMalformedAtomLine:
    def test_malformed_atom_line_raises(self) -> None:
        bad = "title\n 1\n    1ALA      N    1   not-a-number   here   bad\n   5.0   5.0   5.0\n"
        with pytest.raises(ValueError, match=r"malformed \.gro atom line"):
            _read_gro_coordinates(_write(bad, ".gro"))


# --- helpers ----------------------------------------------------------


def _write(text: str, suffix: str) -> Path:
    """Write text to a temp file and return its path."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as fh:
        fh.write(text)
        return Path(fh.name)
