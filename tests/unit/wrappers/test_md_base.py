"""Tests for the MDEngine ABC, Simulation, and Trajectory dataclasses."""

from __future__ import annotations

import numpy as np
import pytest

from molforge.core import AtomArray, Protein
from molforge.md import MDEngine, Simulation, Trajectory


def _make_topology(n_atoms: int = 5) -> Protein:
    arr = AtomArray(n_atoms)
    arr.element[:] = "C"
    arr.atom_name[:] = "CA"
    arr.residue_name[:] = "ALA"
    arr.residue_id[:] = np.arange(1, n_atoms + 1)
    arr.chain_id[:] = "A"
    for i in range(n_atoms):
        arr.coords[i] = [float(i), 0.0, 0.0]
    return Protein(arr)


class TestTrajectory:
    def test_basic_attributes(self) -> None:
        top = _make_topology(5)
        coords = np.random.default_rng(0).normal(size=(10, 5, 3)).astype(np.float32)
        traj = Trajectory(topology=top, coordinates=coords)
        assert traj.n_frames == 10
        assert traj.n_atoms == 5
        assert len(traj) == 10

    def test_optional_arrays_default_none(self) -> None:
        top = _make_topology(3)
        coords = np.zeros((5, 3, 3), dtype=np.float32)
        traj = Trajectory(topology=top, coordinates=coords)
        assert traj.times is None
        assert traj.energies is None
        assert traj.temperatures is None
        assert traj.metadata == {}

    def test_frame_returns_protein_snapshot(self) -> None:
        top = _make_topology(4)
        coords = np.zeros((3, 4, 3), dtype=np.float32)
        coords[1] = 7.5  # mark frame 1 distinctly
        traj = Trajectory(topology=top, coordinates=coords)
        snap = traj.frame(1)
        assert isinstance(snap, Protein)
        assert snap.n_atoms == 4
        # frame returns a deep copy; mutating it must not change the trajectory
        snap.atom_array.coords[0] = [99, 99, 99]
        assert traj.coordinates[1, 0, 0] == pytest.approx(7.5)

    def test_iteration(self) -> None:
        top = _make_topology(2)
        coords = np.zeros((4, 2, 3), dtype=np.float32)
        traj = Trajectory(topology=top, coordinates=coords)
        frames = list(traj)
        assert len(frames) == 4
        assert all(isinstance(f, Protein) for f in frames)


class TestSimulation:
    def test_basic_construction(self) -> None:
        top = _make_topology(3)
        coords = np.zeros((3, 3), dtype=np.float32)
        sim = Simulation(topology=top, coordinates=coords)
        assert sim.n_atoms == 3
        assert sim.time == 0.0
        assert sim.temperature == 300.0
        assert sim.timestep == pytest.approx(0.002)
        assert sim.engine_handle is None

    def test_custom_settings(self) -> None:
        top = _make_topology(2)
        coords = np.zeros((2, 3), dtype=np.float32)
        sim = Simulation(
            topology=top,
            coordinates=coords,
            time=5.0,
            force_field="amber14-all",
            temperature=310.0,
            timestep=0.001,
        )
        assert sim.time == 5.0
        assert sim.force_field == "amber14-all"
        assert sim.temperature == 310.0
        assert sim.timestep == pytest.approx(0.001)


class _DummyMDEngine(MDEngine):
    """Minimal concrete engine for testing the ABC contract."""

    name = "Dummy"

    def prepare(self, protein: Protein, *, force_field: str, **kwargs: object) -> Simulation:
        return Simulation(
            topology=protein,
            coordinates=protein.atom_array.coords.copy(),
            force_field=force_field,
        )

    def minimize(
        self,
        simulation: Simulation,
        *,
        max_iterations: int = 1000,
        tolerance: float = 10.0,
        **kwargs: object,
    ) -> Simulation:
        return simulation

    def run(
        self,
        simulation: Simulation,
        *,
        n_steps: int,
        save_every: int = 1,
        **kwargs: object,
    ) -> Trajectory:
        n_frames = n_steps // save_every + 1
        coords = np.tile(simulation.coordinates[None, :, :], (n_frames, 1, 1)).astype(np.float32)
        return Trajectory(topology=simulation.topology, coordinates=coords)


class TestEngineContract:
    def test_abstract_class_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            MDEngine()  # type: ignore[abstract]

    def test_subclass_can_be_instantiated(self) -> None:
        engine = _DummyMDEngine()
        assert isinstance(engine, MDEngine)

    def test_dummy_full_flow(self) -> None:
        engine = _DummyMDEngine()
        top = _make_topology(3)
        sim = engine.prepare(top, force_field="amber99sb")
        assert sim.force_field == "amber99sb"
        sim = engine.minimize(sim)
        traj = engine.run(sim, n_steps=10, save_every=2)
        # 10/2 + 1 = 6 frames
        assert traj.n_frames == 6
        assert traj.n_atoms == 3

    def test_repr(self) -> None:
        assert repr(_DummyMDEngine()) == "_DummyMDEngine()"


class TestGROMACSStub:
    """GROMACS is a committed-but-unimplemented stub. It must be a
    *coherent* stub: instantiable, satisfying the MDEngine ABC, and
    failing loud with a clear message — not a cryptic abstract-class
    TypeError or a bare NotImplementedError."""

    def test_instantiates(self) -> None:
        """GROMACS must be instantiable — it implements the ABC methods."""
        from molforge.wrappers.md import GROMACS

        engine = GROMACS()
        assert isinstance(engine, MDEngine)

    def test_name(self) -> None:
        from molforge.wrappers.md import GROMACS

        assert GROMACS().name == "GROMACS"

    def test_prepare_raises_with_hint(self) -> None:
        from molforge.wrappers.md import GROMACS

        top = _make_topology(3)
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            GROMACS().prepare(top, force_field="amber99sb")

    def test_minimize_raises_with_hint(self) -> None:
        from molforge.wrappers.md import GROMACS

        with pytest.raises(NotImplementedError, match="not yet implemented"):
            GROMACS().minimize(None)  # type: ignore[arg-type]

    def test_run_raises_with_hint(self) -> None:
        from molforge.wrappers.md import GROMACS

        with pytest.raises(NotImplementedError, match="not yet implemented"):
            GROMACS().run(None, n_steps=10)  # type: ignore[arg-type]

    def test_error_points_at_openmm(self) -> None:
        """The error message should steer users to the working engine."""
        from molforge.wrappers.md import GROMACS

        with pytest.raises(NotImplementedError, match="OpenMM"):
            GROMACS().run(None, n_steps=1)  # type: ignore[arg-type]
