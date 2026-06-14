"""OpenMM wrapper.

[OpenMM](https://openmm.org/) (Eastman et al. 2017, *PLoS Comput Biol*)
is a high-performance, Python-first MD engine. It's the natural choice
for molforge because it's installable via pip (no GROMACS-style binary
compile), runs on GPU out of the box, and supports the major modern
force fields (AMBER, CHARMM, OPLS).

The molforge wrapper exposes a small uniform interface:

    engine = OpenMM()
    sim    = engine.prepare(protein, force_field="amber14-all")
    sim    = engine.minimize(sim, max_iterations=1000)
    traj   = engine.run(sim, n_steps=50000, save_every=500)
    # traj.coordinates is (n_frames, n_atoms, 3) of recorded snapshots

What we do **not** handle automatically (yet):

- **Adding solvent** (water box, ions). The unit cell defaults to vacuum.
  Users who want explicit-solvent simulations should set ``solvent="tip3p"``
  (planned) or do prep in OpenMM Modeller upstream and pass the resulting
  PDB through.
- **Custom force fields**. Only the AMBER and CHARMM force-field files
  shipped with OpenMM are accepted by name.
- **Long-range electrostatics** (Particle Mesh Ewald). The default is
  reaction-field; PME is wired up but requires periodic boundary
  conditions, which means you need solvent.

For everything more involved than the standard implicit-solvent
minimization + production run, drop down to OpenMM's API directly via
``sim.engine_handle``.

Installation::

    pip install 'molforge[md]'

OpenMM is heavy (compiled binaries) but installs cleanly from PyPI on
Linux and macOS. On Windows, install via conda-forge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from molforge.md import (
    MDEngine,
    MDEngineNotInstalledError,
    Simulation,
    Trajectory,
)

if TYPE_CHECKING:
    from molforge.core import Protein


# Common OpenMM force-field XML names. We expose a tiny curated subset;
# users can pass any name the installed openmm.app.ForceField accepts.
_FORCE_FIELD_FILES: dict[str, list[str]] = {
    "amber14-all": ["amber14-all.xml", "amber14/tip3p.xml"],
    "amber99sb": ["amber99sb.xml", "tip3p.xml"],
    "charmm36": ["charmm36.xml", "charmm36/water.xml"],
    "amber99sb-ildn": ["amber99sbildn.xml", "tip3p.xml"],
}


class OpenMM(MDEngine):
    """OpenMM MD engine wrapper.

    Args:
        platform: ``"CUDA"``, ``"CPU"``, ``"OpenCL"``, or ``None`` to
            let OpenMM pick the fastest available. CUDA is by far the
            fastest on supported NVIDIA GPUs.
        precision: ``"mixed"`` (default) or ``"single"`` / ``"double"``.
            Mixed is the standard choice — accurate enough for biology,
            ~2x faster than double on GPU.
        nonbonded_cutoff: Cutoff distance for nonbonded interactions in
            nanometers. Default 1.0 nm. Increase for larger boxes;
            leave alone for typical small-protein simulations.
        nonbonded_method: ``"NoCutoff"`` (default, suitable for implicit
            solvent / vacuum), ``"CutoffNonPeriodic"``, ``"PME"``
            (requires periodic box).
        constraints: ``"HBonds"`` (default — bonds to H are constrained,
            allowing 2-fs timestep), ``"AllBonds"``, or ``None``.
        add_hydrogens: When ``True`` (default), missing hydrogens are
            added with OpenMM's ``Modeller.addHydrogens`` during
            :meth:`prepare`. This is what makes a heavy-atom structure
            — the normal output of folding and docking engines —
            usable as-is: a force field needs explicit hydrogens, and
            without this step ``prepare`` fails with a cryptic
            "no template found" error. The step is idempotent, so a
            structure that already has hydrogens is unaffected. Set
            ``False`` only if you have pre-protonated the structure
            yourself and want OpenMM to use exactly those atoms.

    Example:
        >>> from molforge.wrappers.md import OpenMM
        >>> engine = OpenMM(platform="CUDA")
        >>> sim = engine.prepare(my_protein, force_field="amber14-all")
        >>> sim = engine.minimize(sim)
        >>> traj = engine.run(sim, n_steps=50_000, save_every=500)
        >>> traj.n_frames
        101
    """

    name = "OpenMM"

    def __init__(
        self,
        *,
        platform: str | None = None,
        precision: str = "mixed",
        nonbonded_cutoff: float = 1.0,
        nonbonded_method: str = "NoCutoff",
        constraints: str | None = "HBonds",
        add_hydrogens: bool = True,
    ) -> None:
        self.platform = platform
        self.precision = precision
        self.nonbonded_cutoff = nonbonded_cutoff
        self.nonbonded_method = nonbonded_method
        self.constraints = constraints
        self.add_hydrogens = add_hydrogens

    # ------------------------------------------------------------------
    # Lazy import
    # ------------------------------------------------------------------
    def _require_openmm(self) -> Any:
        """Import OpenMM or raise a clean MDEngineNotInstalledError."""
        try:
            import openmm
            import openmm.app as app
            import openmm.unit as unit
        except ImportError as e:
            raise MDEngineNotInstalledError(
                "OpenMM is required for the OpenMM wrapper. Install with:\n"
                "    pip install 'molforge[md]'\n"
                "On Windows, prefer conda: `conda install -c conda-forge openmm`.\n"
                f"Underlying error: {e}"
            ) from e
        return openmm, app, unit

    # ------------------------------------------------------------------
    # Prepare
    # ------------------------------------------------------------------
    def prepare(  # type: ignore[override]  # engine-specific kwargs refine the **kwargs ABC contract
        self,
        protein: Protein,
        *,
        force_field: str = "amber14-all",
        temperature: float = 300.0,
        timestep: float = 0.002,
        **_kwargs: object,
    ) -> Simulation:
        """Build an OpenMM simulation from a :class:`Protein`.

        Args:
            protein: input structure (no solvent — vacuum / implicit by
                default).
            force_field: name in :data:`_FORCE_FIELD_FILES` or any
                XML filename OpenMM can find.
            temperature: thermostat target in K (default 300).
            timestep: integrator timestep in picoseconds (default 0.002
                = 2 fs; compatible with HBonds constraints).

        Returns:
            A :class:`Simulation` whose ``engine_handle`` is the OpenMM
            ``Simulation`` object — drop down to OpenMM's API via that
            attribute for anything not exposed by molforge.
        """
        openmm, app, unit = self._require_openmm()
        import tempfile
        from pathlib import Path

        # Write the protein to a temp PDB; OpenMM's PDBFile is the
        # path of least resistance for getting topology + positions in.
        from molforge.io import write_pdb

        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as fh:
            tmp_pdb = Path(fh.name)
        try:
            write_pdb(protein, tmp_pdb)
            pdb = app.PDBFile(str(tmp_pdb))
        finally:
            tmp_pdb.unlink(missing_ok=True)

        # Build the System from the requested force field.
        ff_files = _FORCE_FIELD_FILES.get(force_field, [force_field])
        forcefield = app.ForceField(*ff_files)

        # A force field needs explicit hydrogens. Heavy-atom structures
        # — the normal output of folding and docking engines, and what
        # most PDB files on disk contain — would otherwise fail
        # createSystem() with a cryptic "no template found" error.
        # Modeller.addHydrogens() places any missing H; it is
        # idempotent, so an already-protonated structure is untouched.
        #
        # When hydrogens are added the atom count changes, so the
        # molforge-side Protein attached to the returned Simulation is
        # rebuilt from the protonated structure — otherwise its
        # topology (heavy atoms) would disagree with the coordinate
        # array (heavy + H).
        topology = pdb.topology
        positions = pdb.positions
        sim_topology = protein
        if self.add_hydrogens:
            modeller = app.Modeller(pdb.topology, pdb.positions)
            modeller.addHydrogens(forcefield)
            topology = modeller.topology
            positions = modeller.positions
            if topology.getNumAtoms() != protein.atom_array.n_atoms:
                # Hydrogens were actually added — rebuild the Protein so
                # Simulation.topology matches the protonated coordinates.
                with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as h_fh:
                    protonated_pdb = Path(h_fh.name)
                try:
                    app.PDBFile.writeFile(topology, positions, str(protonated_pdb))
                    from molforge.io import read_pdb

                    sim_topology = read_pdb(protonated_pdb)
                finally:
                    protonated_pdb.unlink(missing_ok=True)

        nbm = getattr(app, self.nonbonded_method)
        constraints = getattr(app, self.constraints) if self.constraints else None
        system = forcefield.createSystem(
            topology,
            nonbondedMethod=nbm,
            nonbondedCutoff=self.nonbonded_cutoff * unit.nanometer,
            constraints=constraints,
        )

        # Integrator: standard Langevin thermostat.
        integrator = openmm.LangevinMiddleIntegrator(
            temperature * unit.kelvin,
            1.0 / unit.picosecond,
            timestep * unit.picosecond,
        )
        platform = openmm.Platform.getPlatformByName(self.platform) if self.platform else None
        sim_kwargs: dict[str, Any] = {}
        if platform is not None:
            sim_kwargs["platform"] = platform
            if self.precision and self.platform in ("CUDA", "OpenCL"):
                sim_kwargs["platformProperties"] = {"Precision": self.precision}

        omm_simulation = app.Simulation(
            topology,
            system,
            integrator,
            **sim_kwargs,
        )
        omm_simulation.context.setPositions(positions)
        omm_simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin)

        # Build the molforge-side Simulation snapshot.
        snapshot_coords = (
            omm_simulation.context.getState(getPositions=True)
            .getPositions(asNumpy=True)
            .value_in_unit(unit.angstrom)
        )
        return Simulation(
            topology=sim_topology,
            coordinates=np.asarray(snapshot_coords, dtype=np.float32),
            time=0.0,
            force_field=force_field,
            temperature=temperature,
            timestep=timestep,
            engine_handle=omm_simulation,
            metadata={
                "platform": self.platform or "auto",
                "nonbonded_method": self.nonbonded_method,
                "constraints": self.constraints,
            },
        )

    # ------------------------------------------------------------------
    # Minimize
    # ------------------------------------------------------------------
    def minimize(
        self,
        simulation: Simulation,
        *,
        max_iterations: int = 1000,
        tolerance: float = 10.0,
        **_kwargs: object,
    ) -> Simulation:
        """Energy-minimize ``simulation``'s current configuration in place.

        Args:
            simulation: a :class:`Simulation` from :meth:`prepare`.
            max_iterations: cap on minimizer steps. 0 = unlimited.
            tolerance: convergence threshold in kJ/mol/nm.

        Returns:
            The same simulation with updated coordinates and zero velocities
            (OpenMM resets velocities after minimization).
        """
        _, _, unit = self._require_openmm()
        handle = simulation.engine_handle
        if handle is None:
            raise ValueError("simulation has no engine_handle — was prepare() called?")
        # engine_handle is typed `object` (opaque by contract); inside the
        # OpenMM wrapper we know prepare() put an openmm.app.Simulation
        # there. openmm ships no stubs, so Any is the honest static type.
        handle = cast("Any", handle)
        handle.minimizeEnergy(
            tolerance=tolerance * unit.kilojoule_per_mole / unit.nanometer,
            maxIterations=max_iterations,
        )
        positions = (
            handle.context.getState(getPositions=True)
            .getPositions(asNumpy=True)
            .value_in_unit(unit.angstrom)
        )
        simulation.coordinates = np.asarray(positions, dtype=np.float32)
        return simulation

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(
        self,
        simulation: Simulation,
        *,
        n_steps: int,
        save_every: int = 100,
        **_kwargs: object,
    ) -> Trajectory:
        """Integrate ``simulation`` for ``n_steps`` and return a Trajectory.

        Args:
            simulation: a :class:`Simulation` (should be minimized first).
            n_steps: number of MD steps to run. With the default 2 fs
                timestep, 50,000 steps = 100 ps.
            save_every: record a frame every N steps. Default 100; a
                50,000-step run with save_every=100 gives 501 frames.

        Returns:
            A :class:`Trajectory` whose ``coordinates`` is shape
            ``(n_frames, n_atoms, 3)`` and ``times`` is the
            corresponding picosecond timestamps.
        """
        _, _, unit = self._require_openmm()
        handle = simulation.engine_handle
        if handle is None:
            raise ValueError("simulation has no engine_handle — was prepare() called?")
        # engine_handle is typed `object` (opaque by contract); inside the
        # OpenMM wrapper we know prepare() put an openmm.app.Simulation
        # there. openmm ships no stubs, so Any is the honest static type.
        handle = cast("Any", handle)
        if n_steps < 0:
            raise ValueError(f"n_steps must be >= 0, got {n_steps}")
        if save_every < 1:
            raise ValueError(f"save_every must be >= 1, got {save_every}")

        # Always save the initial frame (n_frames = n_steps // save_every + 1)
        n_frames = n_steps // save_every + 1
        n_atoms = simulation.n_atoms
        coords = np.empty((n_frames, n_atoms, 3), dtype=np.float32)
        times = np.empty(n_frames, dtype=np.float64)
        energies = np.empty(n_frames, dtype=np.float64)

        # Frame 0 = initial state
        state = handle.context.getState(getPositions=True, getEnergy=True)
        coords[0] = np.asarray(
            state.getPositions(asNumpy=True).value_in_unit(unit.angstrom),
            dtype=np.float32,
        )
        times[0] = simulation.time
        energies[0] = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

        for frame in range(1, n_frames):
            handle.step(save_every)
            state = handle.context.getState(getPositions=True, getEnergy=True)
            coords[frame] = np.asarray(
                state.getPositions(asNumpy=True).value_in_unit(unit.angstrom),
                dtype=np.float32,
            )
            times[frame] = simulation.time + (frame * save_every) * simulation.timestep
            energies[frame] = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

        # Update the simulation snapshot to the final frame.
        simulation.coordinates = coords[-1]
        simulation.time = times[-1]

        return Trajectory(
            topology=simulation.topology,
            coordinates=coords,
            times=times,
            energies=energies,
            metadata={
                "engine": "OpenMM",
                "force_field": simulation.force_field,
                "timestep_ps": simulation.timestep,
                "save_every": save_every,
                "n_steps": n_steps,
                "temperature_K": simulation.temperature,
            },
        )
