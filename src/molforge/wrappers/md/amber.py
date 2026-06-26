"""AMBER molecular dynamics engine wrapper.

`AMBER <https://ambermd.org/>`_ is a long-established MD suite. molforge
wraps the freely-available AmberTools components — ``tleap`` for
topology + initial coordinate construction, ``sander`` for
minimisation and dynamics — and optionally uses the faster ``pmemd``
when present (``pmemd`` ships with the commercial Amber package; the
wrapper falls back gracefully when only AmberTools is installed).

The wrapper exposes the same :class:`molforge.md.MDEngine` interface
as :class:`OpenMM` and :class:`GROMACS`:

.. code-block:: python

    from molforge.wrappers.md import AMBER

    engine = AMBER()
    sim = engine.prepare(protein, force_field="ff14SB")
    sim = engine.minimize(sim, max_iterations=1000)
    traj = engine.run(sim, n_steps=50_000, save_every=500)

Same downstream code as the other MD wrappers — ``Simulation`` and
``Trajectory`` types are shared across engines, and the
:class:`molforge.core.Provenance` chain on the returned ``Trajectory``
reads as ``["AMBER.prepare", "AMBER.minimize", "AMBER.run"]``
oldest-first (with any upstream wrapper provenance extending the
chain further back).

AMBER licensing
---------------

AmberTools (``tleap`` + ``sander``) is GPL/LGPL and freely
distributed. The full Amber package (including ``pmemd`` and
``pmemd.cuda``) requires a paid academic / commercial license but
adds large performance gains for production-scale runs. The wrapper
detects ``pmemd`` at run time; users without it get ``sander``
fallback automatically.

What this wrapper does and doesn't do
-------------------------------------

It runs the basic AMBER pipeline: ``tleap`` builds the system in
implicit-solvent or explicit-water mode (default: TIP3P), ``sander``
or ``pmemd`` runs the dynamics. The output trajectory is NetCDF
(``.nc``), which mdtraj reads natively — the wrapper just hands it
back through molforge's existing :func:`molforge.io.read_trajectory`.

It does *not* expose AMBER's full configuration surface: custom
force-field combinations beyond a small curated set, free-energy
methods (TI, MBAR), enhanced sampling (REMD), constraint algorithms
beyond the standard SHAKE on hydrogens. Those are out of scope for
v1; concrete user needs can extend the wrapper.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from molforge.core import Protein
from molforge.core import metadata_keys as mk
from molforge.core.provenance import Provenance
from molforge.md import MDEngineNotInstalledError, Simulation, Trajectory
from molforge.wrappers.md._base import MDEngine


def _parent_provenance(meta: dict[str, Any]) -> Provenance | None:
    """Extract a parent Provenance from a metadata dict, typed-safely.

    Mirrors the helper in the OpenMM and GROMACS wrappers. Per-wrapper
    rather than shared because this is small repeated tooling, not a
    public API surface.
    """
    p = meta.get(mk.PROVENANCE)
    return p if isinstance(p, Provenance) else None


# AMBER force-field names accepted by tleap's `source leaprc.protein.X`
# directive. This is not exhaustive — AMBER ships many — but it's the
# curated set that the wrapper documents and validates upfront so
# users get a clear error rather than a tleap parse failure deep in
# the subprocess.
_KNOWN_FORCE_FIELDS: frozenset[str] = frozenset(
    {
        "ff14SB",  # AMBER ff14SB — the modern protein default
        "ff19SB",  # AMBER ff19SB — successor; OPC water recommended
        "ff99SB",  # ff99SB — older, still widely cited
        "ff99SBildn",  # ildn refinement of ff99SB
    }
)


# Water models accepted by `tleap`'s `loadAmberParams` + the
# corresponding `source leaprc.water.X` directive.
_KNOWN_WATER_MODELS: frozenset[str] = frozenset({"none", "tip3p", "tip4pew", "opc", "spce"})


# ---------------------------------------------------------------------
# tleap and sander input templates
# ---------------------------------------------------------------------


# tleap script for explicit-water solvation. {force_field} and
# {water_model} are substituted; {input_pdb}, {prmtop}, {inpcrd}
# are paths. Box buffer is {box_buffer_a} Å.
_TLEAP_SOLVATED_TEMPLATE = """\
source leaprc.protein.{force_field}
source leaprc.water.{water_model}
mol = loadpdb {input_pdb}
solvateBox mol {tleap_box_name} {box_buffer_a}
addions mol Na+ 0
addions mol Cl- 0
saveAmberParm mol {prmtop} {inpcrd}
quit
"""


# tleap script for vacuum / implicit-solvent setup. No solvation
# step; the system is just the protein with the chosen force field.
_TLEAP_VACUUM_TEMPLATE = """\
source leaprc.protein.{force_field}
mol = loadpdb {input_pdb}
saveAmberParm mol {prmtop} {inpcrd}
quit
"""


# Standard sander minimisation input. {max_iter} and {ncyc} get
# substituted; ncyc is the steepest-descent prelude before conjugate
# gradient takes over.
_SANDER_MIN_TEMPLATE = """\
Minimize
 &cntrl
  imin=1,
  maxcyc={max_iter},
  ncyc={ncyc},
  ntb=1,
  ntr=0,
  cut=10.0,
 /
"""


# Standard sander/pmemd production MD input. {n_steps},
# {timestep_ps}, {save_every}, {temperature_k} substituted.
# ntwx=save_every emits coords to mdcrd; we configure ioutfm=1 to
# get NetCDF (.nc) directly.
_SANDER_PROD_TEMPLATE = """\
Production
 &cntrl
  imin=0,
  ntx=1,
  irest=0,
  nstlim={n_steps},
  dt={timestep_ps},
  ntc=2, ntf=2,
  cut=10.0,
  ntb=1,
  ntp=0,
  ntt=3,
  gamma_ln=2.0,
  temp0={temperature_k},
  tempi={temperature_k},
  ntpr={save_every},
  ntwx={save_every},
  ntwr={save_every},
  ioutfm=1,
 /
"""


class AMBER(MDEngine):
    """Wrapper around the AMBER molecular-dynamics suite.

    Uses ``tleap`` for topology / coordinate setup, ``sander`` (always)
    for minimisation, and ``pmemd`` when available (falling back to
    ``sander``) for dynamics. AmberTools alone is enough for the
    wrapper to function; ``pmemd`` simply makes it faster.

    Args:
        tleap_executable: Name or path of the tleap binary. Defaults
            to ``"tleap"``; set this when AmberTools is installed
            under a different name or not on ``$PATH``. Resolution
            is lazy — construction never touches the filesystem.
        sander_executable: Name or path of the sander binary.
            Defaults to ``"sander"``.
        pmemd_executable: Name or path of pmemd. Defaults to
            ``"pmemd"``; the wrapper detects whether it's actually
            installed at run time and falls back to sander if not.
        water_model: Water model for ``tleap``'s solvation step.
            Use ``"none"`` for a vacuum simulation; any other value
            triggers an explicit-water solvation in :meth:`prepare`.
            Defaults to ``"tip3p"``.
        box_buffer_a: Minimum distance (Å) between solute and box
            edge, passed to ``solvateBox``. Ignored when water_model
            is ``"none"``. Defaults to 10 Å.
        verbose: When ``True``, AMBER subprocess stdout/stderr is
            not captured; streams to the console for debugging.
    """

    name = "AMBER"

    def __init__(
        self,
        *,
        tleap_executable: str = "tleap",
        sander_executable: str = "sander",
        pmemd_executable: str = "pmemd",
        water_model: str = "tip3p",
        box_buffer_a: float = 10.0,
        verbose: bool = False,
    ) -> None:
        if water_model not in _KNOWN_WATER_MODELS:
            raise ValueError(
                f"unknown water_model {water_model!r}; "
                f"expected one of {sorted(_KNOWN_WATER_MODELS)}"
            )
        if box_buffer_a <= 0:
            raise ValueError(f"box_buffer_a must be > 0, got {box_buffer_a}")
        self.tleap_executable = tleap_executable
        self.sander_executable = sander_executable
        self.pmemd_executable = pmemd_executable
        self.water_model = water_model
        self.box_buffer_a = box_buffer_a
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(  # type: ignore[override]  # engine-specific kwargs refine the **kwargs ABC contract
        self,
        protein: Protein,
        *,
        force_field: str = "ff14SB",
        temperature: float = 300.0,
        timestep: float = 0.002,
        **_kwargs: object,
    ) -> Simulation:
        """Build an AMBER :class:`Simulation` from a protein structure.

        Writes a ``tleap`` script in a temp directory and runs it to
        produce the ``.prmtop`` topology and ``.inpcrd`` coordinates
        that ``sander`` / ``pmemd`` consume. The Simulation's
        ``engine_handle`` is the run directory; subsequent calls to
        :meth:`minimize` and :meth:`run` use the same directory.

        Args:
            protein: The protein to simulate. Must be MD-ready
                (capped, protonated, no unusual heterogens). Use
                :func:`molforge.prep.prepare_for_md` first.
            force_field: AMBER force-field name (e.g. ``"ff14SB"``,
                ``"ff19SB"``). See :data:`_KNOWN_FORCE_FIELDS`.
            temperature: Simulation temperature in K (default 300).
            timestep: Integration timestep in ps (default 0.002 =
                2 fs, which assumes SHAKE constraints on hydrogens).

        Raises:
            MDEngineNotInstalledError: If ``tleap`` isn't installed.
            ValueError: If ``force_field`` isn't in the supported set.
            RuntimeError: If tleap fails (parse error, missing
                residue template, etc.).
        """
        if force_field not in _KNOWN_FORCE_FIELDS:
            raise ValueError(
                f"unknown force_field {force_field!r}; "
                f"expected one of {sorted(_KNOWN_FORCE_FIELDS)}"
            )
        tleap = self._require_tleap()

        # Set up a long-lived run directory (the Simulation needs to
        # outlive this call so minimize/run can find the files).
        run_dir = Path(tempfile.mkdtemp(prefix="molforge_amber_"))
        input_pdb = run_dir / "input.pdb"
        prmtop = run_dir / "system.prmtop"
        inpcrd = run_dir / "system.inpcrd"

        # Lazy import so this module imports cheaply without
        # pulling the I/O dispatch table eagerly.
        from molforge.io import save

        save(protein, input_pdb)

        # Build the tleap script. "TIP3PBOX" / "TIP4PBOX" / "OPCBOX"
        # are the standard box names that tleap recognises after the
        # corresponding source leaprc.water.X directive.
        if self.water_model == "none":
            script = _TLEAP_VACUUM_TEMPLATE.format(
                force_field=force_field,
                input_pdb=input_pdb.name,
                prmtop=prmtop.name,
                inpcrd=inpcrd.name,
            )
        else:
            box_name = {
                "tip3p": "TIP3PBOX",
                "tip4pew": "TIP4PEWBOX",
                "opc": "OPCBOX",
                "spce": "TIP3PBOX",  # SPC/E reuses TIP3P box shape
            }[self.water_model]
            script = _TLEAP_SOLVATED_TEMPLATE.format(
                force_field=force_field,
                water_model=self.water_model,
                tleap_box_name=box_name,
                box_buffer_a=self.box_buffer_a,
                input_pdb=input_pdb.name,
                prmtop=prmtop.name,
                inpcrd=inpcrd.name,
            )

        script_path = run_dir / "tleap.in"
        script_path.write_text(script, encoding="utf-8")
        self._run_subprocess(
            [tleap, "-s", "-f", "tleap.in"],
            cwd=run_dir,
            step="AMBER.prepare:tleap",
        )

        if not prmtop.is_file() or not inpcrd.is_file():
            raise RuntimeError(
                f"tleap did not produce {prmtop.name} / {inpcrd.name} "
                f"in {run_dir}. Check tleap's log for the actual error."
            )

        # Read the initial coordinates back into a NumPy array. We
        # do this through mdtraj since it knows the .inpcrd format;
        # the coords are what go into the Simulation snapshot.
        coords = _read_inpcrd_coordinates(inpcrd, prmtop)

        return Simulation(
            topology=protein,
            coordinates=coords,
            force_field=force_field,
            temperature=temperature,
            timestep=timestep,
            engine_handle=run_dir,
            metadata={
                mk.PROVENANCE: Provenance.from_engine(
                    engine="AMBER.prepare",
                    parameters={
                        "force_field": force_field,
                        "temperature": temperature,
                        "timestep": timestep,
                        "water_model": self.water_model,
                        "box_buffer_a": self.box_buffer_a,
                        "tleap_executable": self.tleap_executable,
                    },
                    inputs={"protein": protein.name or "<Protein>"},
                    parent=_parent_provenance(protein.metadata),
                ),
                "engine": "AMBER",
                "run_dir": str(run_dir),
                "water_model": self.water_model,
                "prmtop": str(prmtop),
                "inpcrd": str(inpcrd),
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
        """Energy-minimize with ``sander``.

        Uses sander's mixed steepest-descent / conjugate-gradient
        minimisation. ``tolerance`` is recorded on the Simulation
        metadata but isn't directly an input to sander's
        minimisation (sander runs ``max_iterations`` cycles
        unconditionally and reports the final energy).

        Args:
            simulation: A :class:`Simulation` from :meth:`prepare`.
            max_iterations: Maximum minimisation cycles (passed as
                ``maxcyc`` in the sander input).
            tolerance: Recorded on the simulation for downstream
                inspection. sander itself doesn't take a tolerance
                argument the same way OpenMM / GROMACS do.

        Raises:
            MDEngineNotInstalledError: If ``sander`` isn't installed.
            ValueError: If the Simulation has no AMBER run directory.
            RuntimeError: If sander exits non-zero.
        """
        sander = self._require_sander()
        run_dir = self._run_dir(simulation)

        # ncyc = the steepest-descent prelude length before
        # conjugate-gradient takes over. Standard heuristic: half
        # of maxcyc, capped so SD runs don't dominate.
        ncyc = min(max_iterations // 2, 500)
        script = _SANDER_MIN_TEMPLATE.format(
            max_iter=max_iterations,
            ncyc=ncyc,
        )
        (run_dir / "min.in").write_text(script, encoding="utf-8")

        # sander -O overwrites any existing output files. We name
        # the post-min coordinates "min.rst7" (Amber's native
        # restart format) and read them back as the new
        # Simulation.coordinates.
        self._run_subprocess(
            [
                sander,
                "-O",
                "-i",
                "min.in",
                "-o",
                "min.out",
                "-p",
                "system.prmtop",
                "-c",
                "system.inpcrd",
                "-r",
                "min.rst7",
                "-x",
                "min.nc",
            ],
            cwd=run_dir,
            step="AMBER.minimize:sander",
        )

        min_rst = run_dir / "min.rst7"
        if not min_rst.is_file():
            raise RuntimeError(f"sander did not produce {min_rst}; check min.out for errors.")

        # Read the minimised coords back. sander's .rst7 (also
        # called .restrt) is a simple text format that mdtraj reads.
        simulation.coordinates = _read_rst_coordinates(min_rst, run_dir / "system.prmtop")

        simulation.metadata = {
            **simulation.metadata,
            "minimized": True,
            "minimized_rst": str(min_rst),
            "min_tolerance": tolerance,
            mk.PROVENANCE: Provenance.from_engine(
                engine="AMBER.minimize",
                parameters={
                    "max_iterations": max_iterations,
                    "tolerance": tolerance,
                    "ncyc": ncyc,
                },
                parent=_parent_provenance(simulation.metadata),
            ),
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
        """Run production MD and return a :class:`Trajectory`.

        Uses ``pmemd`` when available, falling back to ``sander``.
        The trajectory is written as NetCDF (``.nc``) which mdtraj
        reads natively through :func:`molforge.io.read_trajectory`.

        Args:
            simulation: A :class:`Simulation` from :meth:`prepare`
                (and optionally :meth:`minimize`).
            n_steps: Number of integration steps.
            save_every: Save coordinates every N steps. The number
                of frames in the returned Trajectory is
                ``n_steps // save_every``.

        Raises:
            MDEngineNotInstalledError: If neither pmemd nor sander
                is available.
            ValueError: If the Simulation has no AMBER run directory.
            RuntimeError: If the MD step exits non-zero or produces
                no trajectory file.
        """
        run_dir = self._run_dir(simulation)
        binary, step_name = self._resolve_md_binary()

        # Pick the starting coordinates: prefer minimised if present.
        start_rst = run_dir / "min.rst7"
        if not start_rst.is_file():
            start_rst = run_dir / "system.inpcrd"

        script = _SANDER_PROD_TEMPLATE.format(
            n_steps=n_steps,
            timestep_ps=simulation.timestep,
            save_every=save_every,
            temperature_k=simulation.temperature,
        )
        (run_dir / "prod.in").write_text(script, encoding="utf-8")

        self._run_subprocess(
            [
                binary,
                "-O",
                "-i",
                "prod.in",
                "-o",
                "prod.out",
                "-p",
                "system.prmtop",
                "-c",
                start_rst.name,
                "-r",
                "prod.rst7",
                "-x",
                "prod.nc",
            ],
            cwd=run_dir,
            step=step_name,
        )

        traj_path = run_dir / "prod.nc"
        if not traj_path.is_file():
            raise RuntimeError(
                f"AMBER did not produce trajectory {traj_path}; check prod.out for errors."
            )

        # Read the trajectory back through molforge's mdtraj-backed
        # reader. The topology is the prmtop (which mdtraj reads
        # directly), so we hand the path to read_trajectory.
        from molforge.io import read_trajectory

        prmtop_path = run_dir / "system.prmtop"
        trajectory = read_trajectory(traj_path, topology=prmtop_path)

        # Stamp our metadata + Provenance onto the returned
        # Trajectory. read_trajectory's own metadata becomes a sub-
        # field so we don't lose it.
        trajectory.metadata = {
            **trajectory.metadata,
            mk.PROVENANCE: Provenance.from_engine(
                engine="AMBER.run",
                parameters={
                    "n_steps": n_steps,
                    "save_every": save_every,
                    "timestep_ps": simulation.timestep,
                    "temperature_K": simulation.temperature,
                    "force_field": simulation.force_field,
                    "md_binary": Path(binary).name,
                },
                parent=_parent_provenance(simulation.metadata),
            ),
            "engine": "AMBER",
            "n_steps": n_steps,
            "save_every": save_every,
            "run_dir": str(run_dir),
            "md_binary": Path(binary).name,
        }
        return trajectory

    # ------------------------------------------------------------------
    # Resolution + subprocess helpers
    # ------------------------------------------------------------------

    def _require_tleap(self) -> str:
        resolved = shutil.which(self.tleap_executable)
        if resolved is None:
            raise MDEngineNotInstalledError(
                f"tleap executable {self.tleap_executable!r} was not "
                "found on PATH.\n"
                "Install AmberTools (e.g. `conda install -c conda-forge "
                "ambertools`, or build from https://ambermd.org/AmberTools.php) "
                "or pass tleap_executable=... to the constructor.\n"
                "For working MD without AMBER, use "
                "molforge.wrappers.md.OpenMM."
            )
        return resolved

    def _require_sander(self) -> str:
        resolved = shutil.which(self.sander_executable)
        if resolved is None:
            raise MDEngineNotInstalledError(
                f"sander executable {self.sander_executable!r} was not "
                "found on PATH. sander ships with AmberTools; install "
                "AmberTools alongside tleap."
            )
        return resolved

    def _resolve_md_binary(self) -> tuple[str, str]:
        """Pick the production-MD binary: pmemd if present, else sander.

        Returns ``(binary_path, step_name)`` where ``step_name``
        encodes which binary was used so the provenance step name
        is accurate.
        """
        pmemd = shutil.which(self.pmemd_executable)
        if pmemd is not None:
            return pmemd, "AMBER.run:pmemd"
        sander = shutil.which(self.sander_executable)
        if sander is not None:
            return sander, "AMBER.run:sander"
        raise MDEngineNotInstalledError(
            f"Neither pmemd ({self.pmemd_executable!r}) nor sander "
            f"({self.sander_executable!r}) was found on PATH. "
            "Install AmberTools for sander, or the full Amber package "
            "for pmemd."
        )

    @staticmethod
    def _run_dir(simulation: Simulation) -> Path:
        handle = simulation.engine_handle
        if not isinstance(handle, Path) or not handle.is_dir():
            raise ValueError(
                "This Simulation has no AMBER run directory. "
                "minimize() and run() must be given a Simulation "
                "produced by AMBER.prepare()."
            )
        return handle

    def _run_subprocess(
        self,
        cmd: list[str],
        *,
        cwd: Path,
        step: str,
    ) -> None:
        """Run a single AMBER subprocess, raising with context on failure.

        The single choke point for every AMBER invocation — tests can
        mock this (or :func:`subprocess.run` beneath it) to exercise
        the pipeline logic without a real AmberTools install.
        """
        try:
            subprocess.run(
                cmd,
                check=True,
                cwd=str(cwd),
                capture_output=not self.verbose,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr or "(stderr not captured)"
            raise RuntimeError(
                f"AMBER step `{step}` failed (exit {e.returncode}).\n"
                f"command: {' '.join(cmd)}\n"
                f"stderr:\n{stderr}"
            ) from e


# ---------------------------------------------------------------------
# inpcrd / rst7 readers
# ---------------------------------------------------------------------


def _read_inpcrd_coordinates(inpcrd: Path, prmtop: Path) -> np.ndarray:
    """Read coordinates from an inpcrd file via mdtraj.

    mdtraj recognises ``.inpcrd`` directly when paired with a
    ``.prmtop`` topology; the resulting Trajectory has one frame
    whose ``xyz`` array gives us the coords we need. Units in
    mdtraj are nm, so convert to Å.
    """
    import mdtraj as md

    traj = md.load(str(inpcrd), top=str(prmtop))
    # mdtraj coordinates are in nm; molforge uses Å.
    return np.asarray(traj.xyz[0] * 10.0, dtype=np.float32)


def _read_rst_coordinates(rst: Path, prmtop: Path) -> np.ndarray:
    """Read coordinates from an Amber restart (.rst7 / .restrt) file."""
    import mdtraj as md

    traj = md.load(str(rst), top=str(prmtop))
    return np.asarray(traj.xyz[0] * 10.0, dtype=np.float32)
