"""Abstract base class for docking engines.

Concrete docking engines should subclass :class:`molforge.docking.DockingEngine`
and live under :mod:`molforge.wrappers.docking`.
"""

from __future__ import annotations

from molforge.docking import DockingEngine

__all__ = ["DockingEngine"]
