"""Abstract base class for docking engines.

Concrete docking engines should subclass :class:`biocore.docking.DockingEngine`
and live under :mod:`biocore.wrappers.docking`.
"""

from __future__ import annotations

from biocore.docking import DockingEngine

__all__ = ["DockingEngine"]
