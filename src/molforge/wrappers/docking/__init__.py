"""Docking-engine wrappers.

Concrete engines:
    - :class:`Vina` — implemented (AutoDock Vina; small-molecule docking)
    - :class:`DiffDock` — implemented (diffusion-based blind docking)
    - :class:`Gnina` — implemented (CNN-rescored Vina; learned scoring)

Shared:
    - :class:`DockingEngine` — abstract base for the engine contract
    - :class:`DockingEngineNotInstalledError` — raised when heavy
      dependencies (the ``vina`` PyPI package, the ``gnina`` binary)
      aren't installed.

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
from molforge.wrappers.docking.gnina import Gnina
from molforge.wrappers.docking.prep import (
    is_pdbqt_path,
    prepare_ligand,
    prepare_receptor,
)
from molforge.wrappers.docking.vina import Vina

__all__ = [  # noqa: RUF022 — grouped: base, then engines, then prep helpers
    "DockingEngine",
    "DockingEngineNotInstalledError",
    "DockingResult",
    "Pose",
    "Vina",
    "DiffDock",
    "Gnina",
    "prepare_receptor",
    "prepare_ligand",
    "is_pdbqt_path",
]
