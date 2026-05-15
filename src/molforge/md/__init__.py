"""Molecular dynamics: trajectory I/O, analysis, and engine wrappers.

A *molecular dynamics simulation* numerically integrates Newton's
equations of motion for a system of atoms subject to a force field,
typically over picosecond-to-microsecond timescales.

In molforge:

- :class:`Trajectory` is a sequence of coordinate snapshots indexed by
  frame, with per-frame metadata (time, energy, temperature).
- :class:`Simulation` is the snapshot of an in-progress simulation:
  topology, current coordinates, current velocities, the integrator's
  parameters. You can extend it with :meth:`Simulation.run` to
  produce more :class:`Trajectory` frames.
- :class:`MDEngine` (in this package) is the abstract base for MD
  engine wrappers (OpenMM, GROMACS, ...). Concrete engines live under
  :mod:`molforge.wrappers.md`.

By convention, every MD engine wrapper exposes:
  - :meth:`prepare(protein, force_field, ...) -> Simulation`
  - :meth:`minimize(simulation, ...) -> Simulation`
  - :meth:`run(simulation, n_steps, ...) -> Trajectory`

So users get the same call structure regardless of which engine they're
running under.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from molforge.core import Protein


@dataclass
class Trajectory:
    """A frame-indexed MD trajectory.

    Attributes:
        topology: A :class:`molforge.core.Protein` defining the atoms,
            their elements/names/connectivity. The topology is the same
            across all frames.
        coordinates: ``(n_frames, n_atoms, 3)`` float32 array of
            per-frame coordinates in Å.
        times: ``(n_frames,)`` float array of simulation time per
            frame, in picoseconds. ``None`` if not recorded.
        energies: ``(n_frames,)`` float array of potential energies
            (kJ/mol). ``None`` if not recorded.
        temperatures: ``(n_frames,)`` float array of instantaneous
            temperatures (K). ``None`` if not recorded.
        metadata: engine-specific extras (force field name, integrator,
            timestep, etc.).
    """

    topology: Protein
    coordinates: NDArray[np.float32]
    times: NDArray[np.float64] | None = None
    energies: NDArray[np.float64] | None = None
    temperatures: NDArray[np.float64] | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        """Number of frames in the trajectory."""
        return int(self.coordinates.shape[0])

    @property
    def n_atoms(self) -> int:
        """Number of atoms (same across all frames)."""
        return int(self.coordinates.shape[1])

    def __len__(self) -> int:
        return self.n_frames

    def frame(self, i: int) -> Protein:
        """Return frame ``i`` as a :class:`Protein` snapshot.

        The returned ``Protein`` shares the topology of this trajectory
        but has its own coordinate array.
        """
        from copy import deepcopy

        snapshot = deepcopy(self.topology)
        snapshot.atom_array.coords[:] = self.coordinates[i]
        return snapshot

    def __iter__(self):  # type: ignore[no-untyped-def]
        for i in range(self.n_frames):
            yield self.frame(i)


@dataclass
class Simulation:
    """The state of an in-progress MD simulation.

    Attributes:
        topology: The system's :class:`Protein` (atoms + connectivity).
        coordinates: ``(n_atoms, 3)`` float32 current positions in Å.
        velocities: ``(n_atoms, 3)`` float32 current velocities. ``None``
            until the simulation has been initialized with a thermostat
            target.
        time: Current simulation time (ps).
        force_field: Force-field name (e.g. ``"amber99sb"``,
            ``"amber14-all"``).
        temperature: Thermostat target temperature (K).
        timestep: Integrator timestep (ps).
        engine_handle: Opaque engine-specific state (e.g. an OpenMM
            ``Simulation`` object). Not serialized; rebuilt on resume
            via the engine wrapper.
        metadata: engine-specific extras.
    """

    topology: Protein
    coordinates: NDArray[np.float32]
    velocities: NDArray[np.float32] | None = None
    time: float = 0.0
    force_field: str = ""
    temperature: float = 300.0
    timestep: float = 0.002  # picoseconds
    engine_handle: object | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def n_atoms(self) -> int:
        return int(self.coordinates.shape[0])


class MDEngine(ABC):
    """Abstract base for MD engines (OpenMM, GROMACS, ...).

    Subclasses live under :mod:`molforge.wrappers.md` and must implement
    :meth:`prepare`, :meth:`minimize`, and :meth:`run`. The contract is
    deliberately small so users can swap engines without rewriting their
    pipeline.

    Attributes:
        name: Human-readable engine name (set by subclasses).
    """

    name: str = "MDEngine"

    @abstractmethod
    def prepare(
        self,
        protein: Protein,
        *,
        force_field: str,
        **kwargs: object,
    ) -> Simulation:
        """Build a :class:`Simulation` from a protein structure.

        Concrete engines handle the engine-specific setup: parameterize
        the system against the force field, build the topology, place
        the structure in a (possibly periodic) simulation box, add
        solvent if requested, etc.
        """

    @abstractmethod
    def minimize(
        self,
        simulation: Simulation,
        *,
        max_iterations: int = 1000,
        tolerance: float = 10.0,
        **kwargs: object,
    ) -> Simulation:
        """Energy-minimize the system in place and return it.

        Args:
            simulation: A :class:`Simulation` (typically just returned
                from :meth:`prepare`).
            max_iterations: Limit on minimizer steps.
            tolerance: Convergence tolerance (kJ/mol/nm).

        Returns:
            The same :class:`Simulation` with updated coordinates.
        """

    @abstractmethod
    def run(
        self,
        simulation: Simulation,
        *,
        n_steps: int,
        save_every: int = 1,
        **kwargs: object,
    ) -> Trajectory:
        """Integrate the simulation for ``n_steps`` and return a
        :class:`Trajectory` containing the recorded frames.

        Args:
            simulation: A :class:`Simulation`.
            n_steps: Number of integrator steps to run.
            save_every: Record a frame every ``save_every`` steps.
                A trajectory has ``n_steps // save_every + 1`` frames
                (the +1 is the initial state).

        Returns:
            A :class:`Trajectory`.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class MDEngineNotInstalledError(ImportError):
    """Raised when an MD engine's heavy dependencies aren't installed."""


__all__ = [
    "MDEngine",
    "MDEngineNotInstalledError",
    "Simulation",
    "Trajectory",
]
