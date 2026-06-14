"""MD-engine wrappers.

Concrete engines:
    - :class:`OpenMM` — implemented (Python-first MD, GPU-accelerated)
    - :class:`GROMACS` — implemented (CLI-based; the classic MD workhorse)

Shared:
    - :class:`MDEngine` — abstract base for the engine contract
    - :class:`MDEngineNotInstalledError` — raised when an engine's
      dependencies (OpenMM, or the ``gmx`` executable) aren't found.

All engines expose the same `prepare -> minimize -> run` flow so users
can swap engines without rewriting their pipeline.
"""

from __future__ import annotations

from molforge.md import MDEngine, MDEngineNotInstalledError, Simulation, Trajectory
from molforge.wrappers.md.gromacs import GROMACS
from molforge.wrappers.md.openmm import OpenMM

__all__ = [  # noqa: RUF022 — grouped: base, then engines
    "MDEngine",
    "MDEngineNotInstalledError",
    "Simulation",
    "Trajectory",
    "OpenMM",
    "GROMACS",
]
