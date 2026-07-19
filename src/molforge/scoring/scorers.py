"""Concrete scorers.

These v1 scorers are all dependency-free — they read already-computed
numbers (folding confidence, a docked pose's score) or wrap a callable —
so scoring never re-runs a heavy engine. Learned scorers that *compute* a
value (ESM perplexity, ProteinMPNN log-likelihood, engine re-scoring) are
follow-ups that implement the same :class:`~molforge.scoring._base.Scorer`
contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from molforge.core import metadata_keys as mk
from molforge.scoring._base import Direction, Score, Scorer

if TYPE_CHECKING:
    from collections.abc import Callable

    from molforge.core import Protein


class ConfidenceScorer(Scorer):
    """Score a :class:`~molforge.core.Protein` by its mean pLDDT-style confidence.

    Reads ``metadata["mean_confidence"]`` (falling back to the mean of
    ``metadata["confidence_per_residue"]``) — the uniform key every folding
    engine writes. Higher is better. No dependencies.
    """

    name = "confidence"
    direction = Direction.HIGHER_IS_BETTER

    def score(self, item: Protein) -> Score:
        value = item.metadata.get(mk.MEAN_CONFIDENCE)
        if value is None:
            per_residue = item.metadata.get(mk.CONFIDENCE_PER_RESIDUE)
            if per_residue is not None and len(per_residue) > 0:
                import numpy as np

                value = float(np.mean(per_residue))
        if value is None:
            raise ValueError(
                "ConfidenceScorer: protein has no 'mean_confidence' or "
                "'confidence_per_residue' in metadata to score."
            )
        return Score(float(value), self.direction, self.name)


class DockingScorer(Scorer):
    """Score a ``Pose`` or ``DockingResult`` by its docked score.

    Docking scores don't share a direction — Vina affinity is
    lower-is-better, a Gnina CNN score is higher-is-better — so the
    direction must be supplied. :meth:`from_engine` reads it from the engine
    that produced the result (its ``score_direction``); otherwise pass
    ``direction`` explicitly. Reads the already-computed ``pose.score``; it
    does not re-dock, so it needs no docking dependencies.
    """

    name = "docking"

    def __init__(self, *, direction: Direction) -> None:
        self.direction = direction

    @classmethod
    def from_engine(cls, engine: Any) -> DockingScorer:
        """Build a scorer whose direction matches ``engine``'s ``score_direction``."""
        raw = getattr(engine, "score_direction", Direction.LOWER_IS_BETTER.value)
        return cls(direction=Direction(raw))

    def score(self, item: Any) -> Score:
        # A DockingResult exposes .best (top pose); a bare Pose is scored directly.
        if hasattr(item, "poses"):
            if not item.poses:
                raise ValueError("DockingScorer: DockingResult has no poses to score.")
            pose = item.best
        else:
            pose = item
        return Score(
            float(pose.score),
            self.direction,
            self.name,
            metadata={"rank": int(getattr(pose, "rank", 0))},
        )


class FunctionScorer(Scorer):
    """Wrap any ``item -> float`` callable as a :class:`Scorer`.

    The escape hatch: plug an engine-rescoring function, an ESM perplexity
    call, or a bespoke composite into anything that consumes a ``Scorer`` —
    including :class:`molforge.design.DesignLoop` — without a dedicated
    class. You supply the :class:`Direction`.
    """

    def __init__(
        self,
        func: Callable[[Any], float],
        *,
        direction: Direction,
        name: str = "function",
    ) -> None:
        self._func = func
        self.direction = direction
        self.name = name

    def score(self, item: Any) -> Score:
        return Score(float(self._func(item)), self.direction, self.name)
