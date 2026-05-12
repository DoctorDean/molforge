"""Docking abstractions: receptor/ligand prep, pose handling, scoring."""

from __future__ import annotations

__all__ = ["DockingEngine", "DockingResult", "Pose"]


class Pose:
    """A single docked ligand pose. TODO: implement."""

    def __init__(self) -> None:
        raise NotImplementedError


class DockingResult:
    """Collection of poses returned by a docking engine. TODO: implement."""

    def __init__(self) -> None:
        raise NotImplementedError


class DockingEngine:
    """Abstract base class for docking engines.

    Concrete implementations live under :mod:`biocore.wrappers.docking`.
    """

    def dock(self, receptor: object, ligand: object) -> DockingResult:
        raise NotImplementedError
