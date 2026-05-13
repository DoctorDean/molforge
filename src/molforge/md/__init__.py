"""Molecular dynamics: trajectory I/O, analysis, and engine wrappers.

This subpackage provides a uniform interface over MD engines (OpenMM, GROMACS)
and trajectory analysis utilities.
"""

from __future__ import annotations

__all__ = ["Simulation", "Trajectory"]


class Trajectory:
    """A frame-indexed MD trajectory. TODO: implement."""

    def __init__(self, path: str) -> None:
        raise NotImplementedError


class Simulation:
    """Engine-agnostic MD simulation handle. TODO: implement."""

    def __init__(self, engine: str = "openmm") -> None:
        raise NotImplementedError
