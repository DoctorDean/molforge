"""Abstract base class for folding engines.

A *folding engine* is any tool that takes one or more protein sequences
and returns predicted structures as :class:`molforge.core.Protein`
objects. The interface is intentionally narrow:

    sequence(s) -> Protein with model-specific confidence in metadata

Concrete engines (ESMFold, AlphaFold, Boltz, RoseTTAFold, ...) inherit
from :class:`FoldingEngine` and implement :meth:`predict` plus the
``_predict_single`` and ``_predict_batch`` hooks as appropriate.

By convention, every folding engine that produces a per-residue
confidence score should write it to
``protein.metadata["confidence_per_residue"]`` as a ``(n_residues,)``
float32 NumPy array, with an additional ``protein.metadata["mean_confidence"]``
scalar. This gives downstream code (filtering, ranking, ML featurization)
a single field to read regardless of which engine produced the structure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from molforge.core import Protein


class FoldingEngine(ABC):
    """Abstract base for sequence-to-structure prediction engines.

    Subclasses must implement :meth:`predict`. The default implementation
    of :meth:`predict_many` is a simple loop; engines that support
    batching (most do) should override it for efficiency.

    Attributes:
        name: Human-readable engine name (set by subclasses).
    """

    name: str = "FoldingEngine"

    @abstractmethod
    def predict(self, sequence: str, **kwargs: object) -> Protein:
        """Predict a single structure from a sequence.

        Args:
            sequence: One-letter amino-acid sequence. Whitespace is
                stripped; non-letter characters raise :class:`ValueError`.
            **kwargs: Engine-specific options.

        Returns:
            A :class:`molforge.core.Protein` whose ``metadata`` includes
            at minimum ``engine`` (the engine name) and, where the engine
            produces one, ``confidence_per_residue`` and ``mean_confidence``.
        """

    def predict_many(
        self,
        sequences: Sequence[str],
        **kwargs: object,
    ) -> list[Protein]:
        """Predict structures for a batch of sequences.

        The default implementation is a serial loop. Engines with batch
        APIs (almost all of them) should override this.
        """
        return [self.predict(s, **kwargs) for s in sequences]

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def _validate_sequence(sequence: str) -> str:
    """Strip whitespace and validate that the sequence contains only letters.

    Returns the cleaned sequence.

    Raises:
        ValueError: If the cleaned sequence is empty or contains non-letter
            characters.
    """
    cleaned = "".join(c for c in sequence if not c.isspace())
    if not cleaned:
        raise ValueError("sequence is empty after stripping whitespace")
    bad = sorted({c for c in cleaned if not c.isalpha()})
    if bad:
        raise ValueError(
            f"sequence contains non-letter characters: {bad!r}. "
            "Folding engines expect a plain one-letter amino-acid sequence."
        )
    return cleaned.upper()


class FoldingEngineNotInstalledError(ImportError):
    """Raised when a folding engine's heavy dependencies aren't installed.

    The message points at the relevant ``pip install`` extras so users
    can fix it without grepping the docs.
    """
