"""Re-export the docking ABC for symmetry with folding/md wrappers."""

from __future__ import annotations

from molforge.docking import DockingEngine, DockingEngineNotInstalledError

__all__ = ["DockingEngine", "DockingEngineNotInstalledError"]
