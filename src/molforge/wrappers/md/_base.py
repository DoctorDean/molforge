"""Abstract base class for MD engines."""

from __future__ import annotations

from abc import ABC, abstractmethod


class MDEngine(ABC):
    """Common interface for molecular-dynamics engines."""

    @abstractmethod
    def simulate(self, protein: object, *, steps: int, **kwargs: object) -> object:
        """Run a simulation and return a `Trajectory`."""
