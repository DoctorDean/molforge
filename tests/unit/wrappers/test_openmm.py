"""Tests for the OpenMM wrapper.

Two halves:

  - Dependency-free tests (construction, the bundled force-field
    registry, the missing-dependency error path) run everywhere.
  - ``TestRealOpenMM`` exercises ``prepare`` / ``minimize`` / ``run``
    against a real OpenMM install and a chemically complete tripeptide
    fixture. These are the tests that actually cover the wrapper's
    system-building and integration-loop logic; they're skipped when
    openmm isn't installed.

The real-engine tests are intentionally *not* marked ``slow`` — the
tripeptide is tiny and a 20-step run is sub-second, so they earn their
place in the normal suite wherever openmm is present.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from molforge.core import AtomArray, Protein
from molforge.io import load
from molforge.md import MDEngineNotInstalledError, Simulation, Trajectory
from molforge.wrappers.md import OpenMM


def _openmm_available() -> bool:
    return importlib.util.find_spec("openmm") is not None


def _topology(n: int = 3) -> Protein:
    arr = AtomArray(n)
    arr.element[:] = "C"
    arr.atom_name[:] = "CA"
    arr.residue_name[:] = "ALA"
    arr.residue_id[:] = np.arange(1, n + 1)
    arr.chain_id[:] = "A"
    return Protein(arr)


@pytest.fixture(scope="module")
def tripeptide(fixtures_dir: Path) -> Protein:
    """A chemically complete heavy-atom ALA-ALA-ALA tripeptide.

    Every standard heavy atom is present (plus the C-terminal OXT), so
    OpenMM's AMBER force field can build a system from it once
    hydrogens are added.
    """
    return load(fixtures_dir / "pdb" / "ala_tripeptide_heavy.pdb")


class TestConstruction:
    def test_defaults(self) -> None:
        engine = OpenMM()
        assert engine.name == "OpenMM"
        assert engine.platform is None
        assert engine.precision == "mixed"
        assert engine.nonbonded_cutoff == pytest.approx(1.0)
        assert engine.nonbonded_method == "NoCutoff"
        assert engine.constraints == "HBonds"
        assert engine.add_hydrogens is True

    def test_custom_settings(self) -> None:
        engine = OpenMM(
            platform="CUDA",
            precision="single",
            nonbonded_cutoff=1.2,
            nonbonded_method="PME",
            constraints="AllBonds",
            add_hydrogens=False,
        )
        assert engine.platform == "CUDA"
        assert engine.precision == "single"
        assert engine.nonbonded_cutoff == pytest.approx(1.2)
        assert engine.nonbonded_method == "PME"
        assert engine.constraints == "AllBonds"
        assert engine.add_hydrogens is False

    def test_constraints_none(self) -> None:
        engine = OpenMM(constraints=None)
        assert engine.constraints is None

    def test_construction_does_not_import_openmm(self) -> None:
        """Constructing must not trigger heavy imports."""
        OpenMM()
        OpenMM(platform="CUDA", precision="single")


class TestMissingDependency:
    @pytest.mark.skipif(_openmm_available(), reason="openmm is installed")
    def test_prepare_without_openmm_raises_clear_error(self) -> None:
        engine = OpenMM()
        with pytest.raises(MDEngineNotInstalledError, match="OpenMM"):
            engine.prepare(_topology(), force_field="amber14-all")

    @pytest.mark.skipif(_openmm_available(), reason="openmm is installed")
    def test_minimize_without_openmm_raises(self) -> None:
        """minimize() imports openmm before the engine-handle check, so
        the missing-dependency error is what surfaces."""
        engine = OpenMM()
        sim = Simulation(
            topology=_topology(),
            coordinates=np.zeros((3, 3), dtype=np.float32),
        )
        with pytest.raises(MDEngineNotInstalledError):
            engine.minimize(sim)

    @pytest.mark.skipif(_openmm_available(), reason="openmm is installed")
    def test_run_without_openmm_raises(self) -> None:
        engine = OpenMM()
        sim = Simulation(
            topology=_topology(),
            coordinates=np.zeros((3, 3), dtype=np.float32),
            engine_handle=None,
        )
        with pytest.raises(MDEngineNotInstalledError):
            engine.run(sim, n_steps=-5)


class TestForceFieldRegistry:
    """The bundled force-field name lookup is dependency-free."""

    def test_known_force_field_resolves(self) -> None:
        from molforge.wrappers.md.openmm import _FORCE_FIELD_FILES

        assert "amber14-all" in _FORCE_FIELD_FILES
        assert "charmm36" in _FORCE_FIELD_FILES
        for name, files in _FORCE_FIELD_FILES.items():
            assert isinstance(files, list)
            assert all(isinstance(f, str) and f.endswith(".xml") for f in files), name

    def test_amber14_includes_water_model(self) -> None:
        from molforge.wrappers.md.openmm import _FORCE_FIELD_FILES

        assert any("tip3p" in f.lower() for f in _FORCE_FIELD_FILES["amber14-all"])


@pytest.mark.skipif(not _openmm_available(), reason="openmm not installed")
class TestRealOpenMM:
    """Exercise prepare / minimize / run against a real OpenMM install.

    These cover the wrapper's actual system-building and
    integration-loop code — the parts that the dependency-free tests
    above cannot reach.
    """

    def test_prepare_builds_runnable_simulation(self, tripeptide: Protein) -> None:
        sim = OpenMM(platform="CPU").prepare(tripeptide, force_field="amber14-all")
        assert isinstance(sim, Simulation)
        assert sim.engine_handle is not None
        assert sim.force_field == "amber14-all"
        assert sim.coordinates.shape[1] == 3
        # Metadata records the engine configuration.
        assert sim.metadata["nonbonded_method"] == "NoCutoff"

    def test_prepare_adds_hydrogens_by_default(self, tripeptide: Protein) -> None:
        """The fixture is heavy-atom only (16 atoms). With add_hydrogens
        on, prepare() protonates it and the returned Simulation's
        topology and coordinates agree on the larger atom count."""
        assert tripeptide.atom_array.n_atoms == 16
        sim = OpenMM(platform="CPU").prepare(tripeptide)
        assert sim.n_atoms > 16  # hydrogens were added
        # topology (the molforge Protein) must match the coordinates.
        assert sim.topology.atom_array.n_atoms == sim.n_atoms

    def test_prepare_without_hydrogens_fails_on_heavy_atoms(self, tripeptide: Protein) -> None:
        """With add_hydrogens=False a heavy-atom structure has no
        explicit H, so the force field cannot template it."""
        engine = OpenMM(platform="CPU", add_hydrogens=False)
        with pytest.raises(ValueError, match=r"[Tt]emplate"):
            engine.prepare(tripeptide, force_field="amber14-all")

    def test_minimize_updates_coordinates(self, tripeptide: Protein) -> None:
        engine = OpenMM(platform="CPU")
        sim = engine.prepare(tripeptide)
        before = sim.coordinates.copy()
        sim = engine.minimize(sim, max_iterations=50)
        assert sim.coordinates.shape == before.shape
        # Minimization should move at least some atoms.
        assert not np.allclose(sim.coordinates, before)

    def test_minimize_without_handle_raises(self, tripeptide: Protein) -> None:
        """A Simulation built by hand (no engine_handle) is rejected."""
        engine = OpenMM(platform="CPU")
        sim = Simulation(
            topology=tripeptide,
            coordinates=np.zeros((16, 3), dtype=np.float32),
            engine_handle=None,
        )
        with pytest.raises(ValueError, match="prepare"):
            engine.minimize(sim)

    def test_run_produces_trajectory(self, tripeptide: Protein) -> None:
        engine = OpenMM(platform="CPU")
        sim = engine.prepare(tripeptide)
        sim = engine.minimize(sim, max_iterations=50)
        traj = engine.run(sim, n_steps=20, save_every=10)
        assert isinstance(traj, Trajectory)
        # n_frames = n_steps // save_every + 1 (the initial frame).
        assert traj.n_frames == 3
        assert traj.coordinates.shape == (3, sim.n_atoms, 3)
        assert traj.times.shape == (3,)
        assert traj.energies.shape == (3,)
        # Times advance monotonically from the simulation start.
        assert traj.times[0] == pytest.approx(0.0)
        assert traj.times[1] < traj.times[2]
        assert traj.metadata["engine"] == "OpenMM"
        assert traj.metadata["n_steps"] == 20

    def test_run_rejects_negative_n_steps(self, tripeptide: Protein) -> None:
        engine = OpenMM(platform="CPU")
        sim = engine.prepare(tripeptide)
        with pytest.raises(ValueError, match="n_steps"):
            engine.run(sim, n_steps=-5)

    def test_run_rejects_bad_save_every(self, tripeptide: Protein) -> None:
        engine = OpenMM(platform="CPU")
        sim = engine.prepare(tripeptide)
        with pytest.raises(ValueError, match="save_every"):
            engine.run(sim, n_steps=10, save_every=0)

    def test_run_without_handle_raises(self, tripeptide: Protein) -> None:
        engine = OpenMM(platform="CPU")
        sim = Simulation(
            topology=tripeptide,
            coordinates=np.zeros((16, 3), dtype=np.float32),
            engine_handle=None,
        )
        with pytest.raises(ValueError, match="prepare"):
            engine.run(sim, n_steps=10)
