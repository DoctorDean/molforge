"""GROMACS MD-engine wrapper.

[GROMACS](https://www.gromacs.org/) is one of the most widely used
molecular-dynamics engines — fast, heavily optimized, and the de-facto
standard in much of the biomolecular-simulation community.

GROMACS is a command-line program (``gmx``), not a Python library, so
this wrapper drives it as a subprocess. A single ``prepare`` /
``minimize`` / ``run`` cycle maps onto the standard GROMACS workflow:

    prepare()   gmx pdb2gmx   -> topology (.top) + coordinates (.gro)
                gmx editconf  -> place the solute in a box
                gmx solvate   -> (optional) fill the box with water
    minimize()  gmx grompp    -> assemble a .tpr run input
                gmx mdrun     -> steepest-descent energy minimization
    run()       gmx grompp    -> assemble the production .tpr
                gmx mdrun     -> integrate; writes .xtc + .edr
                gmx trjconv   -> dump frames as a multi-model PDB
                gmx energy    -> extract per-frame potential energy

All GROMACS state for a simulation lives in a single run directory.
That directory's path is carried on :attr:`Simulation.engine_handle`
so :meth:`minimize` and :meth:`run` can continue from whatever
:meth:`prepare` produced. The directory is created under the system
temp location and is the caller's to clean up — its path is also
recorded in :attr:`Simulation.metadata` under ``"run_dir"``.

Trajectory frames are read back by asking GROMACS itself
(``gmx trjconv``) to convert its binary ``.xtc`` to a multi-model PDB,
which molforge's own PDB reader then parses — the wrapper deliberately
does not depend on a third-party binary-trajectory library.

What the wrapper does **not** do:
  - install GROMACS — ``gmx`` must be on ``PATH`` or passed explicitly;
  - parameterize non-standard residues or ligands — ``pdb2gmx`` only
    knows the force field's built-in residue templates.

Installation: GROMACS is distributed by most package managers
(``apt install gromacs``, ``conda install -c conda-forge gromacs``) or
built from source. For working MD without that, use
:class:`molforge.wrappers.md.OpenMM`.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from molforge.md import MDEngineNotInstalledError, Simulation, Trajectory
from molforge.wrappers.md._base import MDEngine

if TYPE_CHECKING:
    from molforge.core import Protein


# GROMACS force-field names accepted by `pdb2gmx -ff`. This is not
# exhaustive — GROMACS ships many — but it documents the common ones
# and lets `prepare` validate early with a clear error rather than
# letting pdb2gmx fail deep in a subprocess.
_KNOWN_FORCE_FIELDS: frozenset[str] = frozenset(
    {
        "amber03",
        "amber94",
        "amber96",
        "amber99",
        "amber99sb",
        "amber99sb-ildn",
        "amberGS",
        "charmm27",
        "gromos43a1",
        "gromos43a2",
        "gromos45a3",
        "gromos53a5",
        "gromos53a6",
        "gromos54a7",
        "oplsaa",
    }
)

# Water models accepted by `pdb2gmx -water`.
_KNOWN_WATER_MODELS: frozenset[str] = frozenset(
    {"none", "spc", "spce", "tip3p", "tip4p", "tip5p", "tips3p"}
)


class GROMACS(MDEngine):
    """Wrapper around the GROMACS MD engine.

    Args:
        gmx_executable: Name or path of the GROMACS driver binary.
            Defaults to ``"gmx"``; set this when GROMACS is installed
            under a different name (e.g. ``"gmx_mpi"``) or not on
            ``PATH``. Resolution is lazy — construction never touches
            the filesystem, so a ``GROMACS()`` instance is cheap to
            create even where GROMACS is not installed.
        water_model: Water model passed to ``pdb2gmx -water``. Use
            ``"none"`` (the default) for a vacuum simulation; any other
            value triggers a ``gmx solvate`` step in :meth:`prepare`.
        box_margin: Minimum distance (nm) between the solute and the
            box edge, passed to ``editconf -d``.
        box_type: Box shape for ``editconf -bt`` (``"cubic"``,
            ``"dodecahedron"``, ``"octahedron"``, ...).
        verbose: When ``True``, GROMACS subprocess stdout/stderr is not
            captured, so it streams to the console. Useful for
            debugging a failing run.

    Example:
        >>> from molforge.wrappers.md import GROMACS
        >>> import molforge as mf
        >>>
        >>> protein = mf.load("protein.pdb")
        >>> engine = GROMACS(water_model="tip3p")
        >>> sim = engine.prepare(protein, force_field="amber99sb-ildn")
        >>> sim = engine.minimize(sim, max_iterations=500)
        >>> traj = engine.run(sim, n_steps=5000, save_every=500)
        >>> traj.n_frames
        11
    """

    name = "GROMACS"

    def __init__(
        self,
        *,
        gmx_executable: str = "gmx",
        water_model: str = "none",
        box_margin: float = 1.0,
        box_type: str = "cubic",
        verbose: bool = False,
    ) -> None:
        if water_model not in _KNOWN_WATER_MODELS:
            raise ValueError(
                f"unknown water_model {water_model!r}; "
                f"expected one of {sorted(_KNOWN_WATER_MODELS)}"
            )
        if box_margin <= 0:
            raise ValueError(f"box_margin must be > 0, got {box_margin}")
        self.gmx_executable = gmx_executable
        self.water_model = water_model
        self.box_margin = box_margin
        self.box_type = box_type
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def prepare(  # type: ignore[override]  # engine-specific kwargs refine the **kwargs ABC contract
        self,
        protein: Protein,
        *,
        force_field: str = "amber99sb-ildn",
        temperature: float = 300.0,
        timestep: float = 0.002,
        **_kwargs: object,
    ) -> Simulation:
        """Build a GROMACS :class:`Simulation` from a protein structure.

        Runs ``pdb2gmx`` → ``editconf`` → (optionally) ``solvate`` in a
        fresh run directory.

        Args:
            protein: Input structure. ``pdb2gmx`` needs every heavy
                atom of every residue it is asked to parameterize, and
                only knows the force field's standard residues.
            force_field: A GROMACS force-field name (see
                :data:`_KNOWN_FORCE_FIELDS`).
            temperature: Thermostat target (K), stored on the returned
                :class:`Simulation` and used by :meth:`run`.
            timestep: Integrator timestep (ps), likewise stored and
                used by :meth:`run`.

        Returns:
            A :class:`Simulation` whose ``engine_handle`` and
            ``metadata["run_dir"]`` both hold the run-directory path.

        Raises:
            MDEngineNotInstalledError: If ``gmx`` cannot be found.
            ValueError: If ``force_field`` is not recognized.
            RuntimeError: If any GROMACS step fails.
        """
        gmx = self._require_gmx()
        if force_field not in _KNOWN_FORCE_FIELDS:
            raise ValueError(
                f"unknown GROMACS force_field {force_field!r}; "
                f"expected one of {sorted(_KNOWN_FORCE_FIELDS)}"
            )

        run_dir = Path(tempfile.mkdtemp(prefix="molforge_gmx_"))
        from molforge.io import write_pdb

        input_pdb = run_dir / "input.pdb"
        write_pdb(protein, input_pdb)

        # 1. pdb2gmx: build topology + processed coordinates.
        self._run_gmx(
            gmx,
            [
                "pdb2gmx",
                "-f",
                str(input_pdb),
                "-o",
                "conf.gro",
                "-p",
                "topol.top",
                "-ff",
                force_field,
                "-water",
                self.water_model,
            ],
            cwd=run_dir,
        )

        # 2. editconf: centre the solute in a box.
        self._run_gmx(
            gmx,
            [
                "editconf",
                "-f",
                "conf.gro",
                "-o",
                "boxed.gro",
                "-c",
                "-d",
                str(self.box_margin),
                "-bt",
                self.box_type,
            ],
            cwd=run_dir,
        )
        current_gro = "boxed.gro"

        # 3. solvate (only when a water model was requested).
        if self.water_model != "none":
            self._run_gmx(
                gmx,
                [
                    "solvate",
                    "-cp",
                    "boxed.gro",
                    "-o",
                    "solvated.gro",
                    "-p",
                    "topol.top",
                ],
                cwd=run_dir,
            )
            current_gro = "solvated.gro"

        # Record which .gro is "current" so minimize/run know where to
        # start. Symlink-free: just copy to a stable name.
        shutil.copy(run_dir / current_gro, run_dir / "current.gro")
        coords = _read_gro_coordinates(run_dir / "current.gro")

        return Simulation(
            topology=protein,
            coordinates=coords,
            force_field=force_field,
            temperature=temperature,
            timestep=timestep,
            engine_handle=run_dir,
            metadata={
                "engine": "GROMACS",
                "run_dir": str(run_dir),
                "water_model": self.water_model,
                "box_type": self.box_type,
            },
        )

    def minimize(
        self,
        simulation: Simulation,
        *,
        max_iterations: int = 1000,
        tolerance: float = 10.0,
        **_kwargs: object,
    ) -> Simulation:
        """Energy-minimize the system with steepest descent.

        Writes an EM ``.mdp``, assembles a ``.tpr`` with ``grompp``, and
        runs ``mdrun``. Returns the same :class:`Simulation` with its
        coordinates updated to the minimized structure.

        Args:
            simulation: A :class:`Simulation` from :meth:`prepare`.
            max_iterations: Cap on steepest-descent steps (``nsteps``).
            tolerance: Convergence tolerance in kJ/mol/nm (``emtol``).

        Raises:
            MDEngineNotInstalledError: If ``gmx`` cannot be found.
            ValueError: If the simulation has no GROMACS run directory.
            RuntimeError: If a GROMACS step fails.
        """
        gmx = self._require_gmx()
        run_dir = self._run_dir(simulation)

        mdp = run_dir / "em.mdp"
        mdp.write_text(
            _EM_MDP_TEMPLATE.format(emtol=tolerance, nsteps=max_iterations),
            encoding="utf-8",
        )
        self._run_gmx(
            gmx,
            [
                "grompp",
                "-f",
                "em.mdp",
                "-c",
                "current.gro",
                "-p",
                "topol.top",
                "-o",
                "em.tpr",
                "-maxwarn",
                "2",
            ],
            cwd=run_dir,
        )
        self._run_gmx(gmx, ["mdrun", "-deffnm", "em"], cwd=run_dir)

        shutil.copy(run_dir / "em.gro", run_dir / "current.gro")
        simulation.coordinates = _read_gro_coordinates(run_dir / "current.gro")
        simulation.metadata = {
            **simulation.metadata,
            "minimized": True,
            "emtol": tolerance,
        }
        return simulation

    def run(
        self,
        simulation: Simulation,
        *,
        n_steps: int,
        save_every: int = 1,
        **_kwargs: object,
    ) -> Trajectory:
        """Integrate the system and return a :class:`Trajectory`.

        Writes a production MD ``.mdp``, assembles the ``.tpr``, runs
        ``mdrun``, then reads the frames back by converting the ``.xtc``
        to a multi-model PDB with ``trjconv`` and the energies with
        ``gmx energy``.

        Args:
            simulation: A :class:`Simulation` from :meth:`prepare`
                (typically after :meth:`minimize`).
            n_steps: Number of integrator steps.
            save_every: Record a frame every ``save_every`` steps. The
                trajectory has ``n_steps // save_every + 1`` frames
                (the +1 is the initial frame).

        Raises:
            MDEngineNotInstalledError: If ``gmx`` cannot be found.
            ValueError: If ``n_steps`` / ``save_every`` are invalid or
                the simulation has no run directory.
            RuntimeError: If a GROMACS step fails or produces no frames.
        """
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        if save_every < 1:
            raise ValueError(f"save_every must be >= 1, got {save_every}")
        gmx = self._require_gmx()
        run_dir = self._run_dir(simulation)

        mdp = run_dir / "md.mdp"
        mdp.write_text(
            _MD_MDP_TEMPLATE.format(
                nsteps=n_steps,
                nstxout=save_every,
                nstenergy=save_every,
                dt=simulation.timestep,
                ref_t=simulation.temperature,
            ),
            encoding="utf-8",
        )
        self._run_gmx(
            gmx,
            [
                "grompp",
                "-f",
                "md.mdp",
                "-c",
                "current.gro",
                "-p",
                "topol.top",
                "-o",
                "md.tpr",
                "-maxwarn",
                "2",
            ],
            cwd=run_dir,
        )
        self._run_gmx(gmx, ["mdrun", "-deffnm", "md"], cwd=run_dir)

        # Convert the binary trajectory to a multi-model PDB so
        # molforge's own PDB reader can parse it — no dependency on a
        # third-party binary-trajectory library. trjconv prompts for a
        # group on stdin; "System" selects everything.
        self._run_gmx(
            gmx,
            ["trjconv", "-f", "md.xtc", "-s", "md.tpr", "-o", "md_frames.pdb"],
            cwd=run_dir,
            stdin="System\n",
        )
        frames_pdb = run_dir / "md_frames.pdb"
        if not frames_pdb.is_file():
            raise RuntimeError(
                f"GROMACS produced no trajectory frames in {run_dir}. "
                "Check the mdrun log (md.log) for the actual error."
            )

        coordinates = _read_multimodel_pdb_coordinates(frames_pdb.read_text(encoding="utf-8"))
        if coordinates.shape[0] == 0:
            raise RuntimeError(f"GROMACS trajectory in {run_dir} contained no frames.")

        energies = self._extract_energies(gmx, run_dir)
        n_frames = coordinates.shape[0]
        times = np.arange(n_frames, dtype=np.float64) * (simulation.timestep * save_every)

        return Trajectory(
            topology=simulation.topology,
            coordinates=coordinates,
            times=times,
            energies=energies if energies is not None else None,
            metadata={
                "engine": "GROMACS",
                "n_steps": n_steps,
                "save_every": save_every,
                "run_dir": str(run_dir),
            },
        )

    # ------------------------------------------------------------------
    # Installation resolution
    # ------------------------------------------------------------------
    def _require_gmx(self) -> str:
        """Resolve the ``gmx`` executable or raise a clean error."""
        resolved = shutil.which(self.gmx_executable)
        if resolved is None:
            raise MDEngineNotInstalledError(
                f"GROMACS executable {self.gmx_executable!r} was not found "
                "on PATH.\n"
                "Install GROMACS (e.g. `apt install gromacs`, "
                "`conda install -c conda-forge gromacs`) or pass "
                "gmx_executable=... to the constructor.\n"
                "For working MD without GROMACS, use "
                "molforge.wrappers.md.OpenMM."
            )
        return resolved

    @staticmethod
    def _run_dir(simulation: Simulation) -> Path:
        """Return the GROMACS run directory carried by a Simulation.

        Raises a clear error if the Simulation did not come from this
        engine's :meth:`prepare`.
        """
        handle = simulation.engine_handle
        if not isinstance(handle, Path) or not handle.is_dir():
            raise ValueError(
                "This Simulation has no GROMACS run directory. "
                "minimize() and run() must be given a Simulation "
                "produced by GROMACS.prepare()."
            )
        return handle

    # ------------------------------------------------------------------
    # Subprocess seam
    # ------------------------------------------------------------------
    def _run_gmx(
        self,
        gmx: str,
        args: list[str],
        *,
        cwd: Path,
        stdin: str | None = None,
    ) -> None:
        """Run a single ``gmx`` subcommand, raising on failure.

        The single choke point for every GROMACS invocation — tests
        mock this (or ``subprocess.run`` beneath it) to exercise the
        pipeline logic without a real GROMACS install.
        """
        cmd = [gmx, *args]
        try:
            subprocess.run(
                cmd,
                check=True,
                cwd=str(cwd),
                input=stdin,
                capture_output=not self.verbose,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr or "(stderr not captured)"
            raise RuntimeError(
                f"GROMACS step `gmx {args[0]}` failed (exit {e.returncode}).\n"
                f"command: {' '.join(cmd)}\n"
                f"stderr:\n{stderr}"
            ) from e

    def _extract_energies(self, gmx: str, run_dir: Path) -> np.ndarray | None:
        """Pull per-frame potential energy from the ``.edr`` via ``gmx energy``.

        Returns ``None`` if the energy file is absent or unparseable —
        energies are reported as a best-effort extra, and a trajectory
        is still valid without them.
        """
        if not (run_dir / "md.edr").is_file():
            return None
        try:
            self._run_gmx(
                gmx,
                ["energy", "-f", "md.edr", "-o", "energy.xvg"],
                cwd=run_dir,
                stdin="Potential\n",
            )
        except RuntimeError:
            return None
        xvg = run_dir / "energy.xvg"
        if not xvg.is_file():
            return None
        return _read_xvg_column(xvg.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# .mdp parameter-file templates
# ----------------------------------------------------------------------
_EM_MDP_TEMPLATE = """\
; molforge GROMACS energy-minimization parameters
integrator  = steep
emtol       = {emtol}
emstep      = 0.01
nsteps      = {nsteps}
nstlist     = 10
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
"""

_MD_MDP_TEMPLATE = """\
; molforge GROMACS production-MD parameters
integrator  = md
nsteps      = {nsteps}
dt          = {dt}
nstxout     = {nstxout}
nstvout     = {nstxout}
nstenergy   = {nstenergy}
nstlog      = {nstenergy}
nstlist     = 10
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
tcoupl      = V-rescale
tc-grps     = System
tau-t       = 0.1
ref-t       = {ref_t}
gen-vel     = yes
gen-temp    = {ref_t}
"""


# ----------------------------------------------------------------------
# GROMACS output parsing helpers
# ----------------------------------------------------------------------
def _read_gro_coordinates(path: Path) -> np.ndarray:
    """Parse atom coordinates from a GROMACS ``.gro`` file.

    The ``.gro`` format is fixed-layout text: a title line, an
    atom-count line, one line per atom, then a box-vector line. Atom
    coordinates occupy columns 20-44 in nanometres; molforge works in
    Å, so they are scaled by 10.

    Raises:
        ValueError: If the file is too short or the count line is
            unparseable.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"{path} is too short to be a .gro file")
    try:
        n_atoms = int(lines[1].strip())
    except ValueError as e:
        raise ValueError(f"could not parse atom count from .gro line {lines[1]!r}") from e
    if len(lines) < 2 + n_atoms:
        raise ValueError(f"{path} declares {n_atoms} atoms but the atom block is truncated")
    coords = np.zeros((n_atoms, 3), dtype=np.float32)
    for i in range(n_atoms):
        atom_line = lines[2 + i]
        try:
            x = float(atom_line[20:28])
            y = float(atom_line[28:36])
            z = float(atom_line[36:44])
        except ValueError as e:
            raise ValueError(f"malformed .gro atom line: {atom_line!r}") from e
        # .gro is in nm; molforge coordinates are in Å.
        coords[i] = (x * 10.0, y * 10.0, z * 10.0)
    return coords


def _read_multimodel_pdb_coordinates(text: str) -> np.ndarray:
    """Extract per-frame coordinates from a multi-MODEL PDB.

    ``gmx trjconv`` writes each trajectory frame as a ``MODEL`` /
    ``ENDMDL`` block. Returns an ``(n_frames, n_atoms, 3)`` array. The
    atom count is taken from the first model; later models are assumed
    to match (they always do for a fixed-topology trajectory).
    """
    frames: list[list[tuple[float, float, float]]] = []
    current: list[tuple[float, float, float]] = []
    in_model = False
    for line in text.splitlines():
        if line.startswith("MODEL"):
            in_model = True
            current = []
            continue
        if line.startswith("ENDMDL"):
            if current:
                frames.append(current)
            in_model = False
            continue
        if line.startswith(("ATOM", "HETATM")):
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            current.append((x, y, z))
    # A single-frame trajectory may have no MODEL records at all.
    if not frames and current:
        frames.append(current)
    if not in_model and current and not frames:
        frames.append(current)
    if not frames:
        return np.zeros((0, 0, 3), dtype=np.float32)
    return np.asarray(frames, dtype=np.float32)


def _read_xvg_column(text: str) -> np.ndarray | None:
    """Read the first data column from a GROMACS ``.xvg`` file.

    ``.xvg`` is Grace plot data: ``#`` and ``@`` lines are comments and
    metadata, the rest are whitespace-separated numeric rows whose
    first column is the x-axis (time/frame) and second the value.
    Returns the value column, or ``None`` if no data rows are found.
    """
    values: list[float] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "@")):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        try:
            values.append(float(parts[1]))
        except ValueError:
            continue
    if not values:
        return None
    return np.asarray(values, dtype=np.float64)
