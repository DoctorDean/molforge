"""Docking-engine wrappers.

Concrete engines:
    - :class:`Vina` — implemented (AutoDock Vina; small-molecule docking)
    - :class:`DiffDock` — stub (diffusion-based docking)

Shared:
    - :class:`DockingEngine` — abstract base for the engine contract
    - :class:`DockingEngineNotInstalledError` — raised when heavy
      dependencies (the ``vina`` PyPI package) aren't installed.

All engines write poses sorted best-first into
:attr:`DockingResult.poses`.
"""

from __future__ import annotations

from molforge.docking import (
    DockingEngine,
    DockingEngineNotInstalledError,
    DockingResult,
    Pose,
)
from molforge.wrappers.docking.diffdock import DiffDock
from molforge.wrappers.docking.vina import Vina

__all__ = [  # noqa: RUF022 — grouped: base, then engines
    "DockingEngine",
    "DockingEngineNotInstalledError",
    "DockingResult",
    "Pose",
    "Vina",
    "DiffDock",
]
