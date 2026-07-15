"""Generative protein design: abstract engine contract and result types.

This module defines the shared types used by the concrete generative
engines under :mod:`molforge.wrappers.generative`:

- :class:`GenerativeEngine` — abstract base. Subclasses implement
  :meth:`generate` and return either a list of :class:`Protein` (for
  backbone generators) or a list of :class:`DesignedSequence` (for
  sequence designers).
- :class:`DesignedSequence` — a sequence + score + metadata triple
  returned by sequence-design engines (ProteinMPNN, etc.).
- :class:`GenerativeEngineNotInstalledError` — clean error class when
  the engine's heavy deps aren't installed.

The contract is deliberately small. Two main use cases drive the
design:

1. **Backbone generation** (RFdiffusion-like): ``generate(target=...)``
   returns a list of :class:`Protein` backbones.
2. **Sequence design** (ProteinMPNN-like): ``generate(backbone=...)``
   returns a list of :class:`DesignedSequence` instances ranked by score.

Both share lazy-loading semantics — heavy deps and model weights are
only loaded on the first call to :meth:`generate`, not on construction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class GenerativeEngineNotInstalledError(ImportError):
    """Raised when a generative engine's heavy dependencies aren't installed."""


@dataclass
class DesignedSequence:
    """A sequence proposed by a sequence-design engine.

    Attributes:
        sequence: One-letter amino-acid sequence (`/`-separated for
            multi-chain designs, matching ProteinMPNN's convention).
        score: Negative log-likelihood under the model, averaged
            across positions. Lower = better (per the convention used
            by ProteinMPNN; some engines invert this).
        recovery: Optional sequence-recovery fraction relative to a
            native reference (in ``[0, 1]``). Only meaningful when a
            native sequence is available.
        metadata: Engine-specific extras (model name, sampling
            temperature, seed, fixed positions, per-position
            confidence, etc.).
    """

    sequence: str
    score: float
    recovery: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __repr__(self) -> str:
        seq_preview = self.sequence if len(self.sequence) <= 30 else self.sequence[:27] + "..."
        return f"DesignedSequence(sequence={seq_preview!r}, score={self.score:.3f})"


class GenerativeEngine(ABC):
    """Abstract base for generative-design engines.

    Subclasses live under :mod:`molforge.wrappers.generative` and must
    implement :meth:`generate`. The contract is intentionally loose —
    different engine categories (backbone generators vs. sequence
    designers) return different types — but every concrete engine
    follows the same lazy-import / clean-error / uniform-metadata
    pattern as the other molforge wrappers.

    Attributes:
        name: Human-readable engine name (set by subclasses).
    """

    name: str = "GenerativeEngine"

    #: How :func:`molforge.parallel.run_many` batches this engine: ``"serial"``
    #: (default, for GPU engines) or ``"process"`` for CPU / subprocess ones.
    parallelism: str = "serial"

    @abstractmethod
    def generate(self, *args: object, **kwargs: object) -> list[object]:
        """Run the engine and return a list of designs.

        The return type depends on the engine:
            - Backbone generators (RFdiffusion) return ``list[Protein]``.
            - Sequence designers (ProteinMPNN) return
              ``list[DesignedSequence]``.

        Concrete engines document their exact return type.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


__all__ = [
    "DesignedSequence",
    "GenerativeEngine",
    "GenerativeEngineNotInstalledError",
]
