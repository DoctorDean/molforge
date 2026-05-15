"""Re-export the MD ABC for symmetry with folding/docking wrappers."""

from __future__ import annotations

from molforge.md import MDEngine, MDEngineNotInstalledError

__all__ = ["MDEngine", "MDEngineNotInstalledError"]
