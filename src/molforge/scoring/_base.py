"""Scoring core: a self-describing :class:`Score`, the :class:`Scorer` ABC,
and direction-aware ranking helpers.

Every scorer in molforge — docking affinity, folding confidence, a learned
sequence score — reports a bare float whose "good" direction you have to
*know* out of band: Vina affinity is lower-is-better, a Gnina CNN score is
higher-is-better, ProteinMPNN's score is lower-is-better negative
log-likelihood, pLDDT is higher-is-better. This module makes the direction
explicit, so a score can be ranked, compared, or thresholded without
remembering which way is up — and so any scorer can drop into
:class:`molforge.design.DesignLoop` as an objective.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

    from molforge.parallel import Backend, OnError


class Direction(Enum):
    """Which way is better for a score."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


@dataclass(frozen=True)
class Score:
    """A scalar score plus the direction that makes it comparable.

    Attributes:
        value: The raw score in the scorer's native units.
        direction: Whether higher or lower ``value`` is better.
        scorer: Name of the scorer that produced it.
        metadata: Optional extras (component scores, units, rank, ...).
    """

    value: float
    direction: Direction
    scorer: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ranking_key(self) -> float:
        """A "higher is always better" view of :attr:`value`.

        Lower-is-better scores are negated, so ``ranking_key`` can be
        compared and sorted uniformly across scorers of either direction.
        """
        return self.value if self.direction is Direction.HIGHER_IS_BETTER else -self.value

    def is_better_than(self, other: Score) -> bool:
        """True if this score is strictly better than ``other``.

        A ``nan`` value is never better than a real one (and loses to
        everything), so unscoreable items sink in a ranking rather than
        surfacing spuriously.
        """
        if math.isnan(self.value):
            return False
        if math.isnan(other.value):
            return True
        return self.ranking_key > other.ranking_key


class Scorer(ABC):
    """Abstract base for anything that assigns a :class:`Score` to an item.

    Concrete scorers document what they take (a :class:`~molforge.core.Protein`,
    a ``Pose``, a sequence string, ...) — the contract is deliberately loose,
    like the engine ABCs — and set :attr:`direction`.

    Attributes:
        name: Human-readable scorer name (set by subclasses).
        direction: Whether higher or lower is better for this scorer.
    """

    name: str = "Scorer"
    direction: Direction

    @abstractmethod
    def score(self, item: Any) -> Score:
        """Score a single item."""

    def score_many(
        self,
        items: Iterable[Any],
        *,
        workers: int | None = None,
        backend: Backend = "serial",
        on_error: OnError = "raise",
    ) -> list[Score]:
        """Score many items, in input order.

        Defaults to the ``"serial"`` backend — most scorers are cheap
        metadata reads, and a scorer built from a lambda isn't picklable for
        the process backend. See :func:`molforge.parallel.map_parallel`.
        """
        from molforge.parallel import map_parallel

        return map_parallel(
            self.score, list(items), workers=workers, backend=backend, on_error=on_error
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def rank(items: Iterable[Any], scorer: Scorer) -> list[tuple[Any, Score]]:
    """Score every item and return ``(item, score)`` pairs best-first.

    ``nan`` scores sort last, so unscoreable items never displace real ones.
    """
    scored = [(item, scorer.score(item)) for item in items]
    return sorted(
        scored,
        key=lambda pair: (
            math.isnan(pair[1].value),
            -pair[1].ranking_key if not math.isnan(pair[1].value) else 0.0,
        ),
    )


def best(items: Iterable[Any], scorer: Scorer) -> Any:
    """Return the single best item under ``scorer``.

    Raises:
        ValueError: If ``items`` is empty.
    """
    ranked = rank(items, scorer)
    if not ranked:
        raise ValueError("best() got no items to score")
    return ranked[0][0]
