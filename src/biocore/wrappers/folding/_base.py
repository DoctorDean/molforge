"""Abstract base class for folding engines."""

from __future__ import annotations

from abc import ABC, abstractmethod


class FoldingEngine(ABC):
    """Common interface for protein-structure prediction engines."""

    @abstractmethod
    def predict(self, sequence: str, **kwargs: object) -> object:
        """Predict a structure from a sequence and return a `Protein`."""
