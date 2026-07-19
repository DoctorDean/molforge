"""Unified scoring: score any structure, pose, or sequence with any scorer.

Every score in molforge is a bare float whose "good" direction you have to
know out of band — Vina affinity is lower-is-better, a Gnina CNN score is
higher-is-better, pLDDT is higher-is-better, ProteinMPNN's log-likelihood is
lower-is-better. :mod:`molforge.scoring` makes the direction explicit so you
can rank, compare, and threshold across sources uniformly:

    from molforge.scoring import ConfidenceScorer, rank

    best_first = rank(structures, ConfidenceScorer())
    top = best_first[0][0]

Every scorer returns a :class:`Score` carrying its :class:`Direction`, and
:attr:`Score.ranking_key` gives a "higher is always better" number. A
:class:`Scorer` also plugs straight into :class:`molforge.design.DesignLoop`
as an objective.

**What's here (v1)** — all dependency-free:

- :class:`ConfidenceScorer` — a folded structure's mean pLDDT.
- :class:`DockingScorer` — a pose / docking result's score, with the
  producing engine's direction (via :meth:`DockingScorer.from_engine`).
- :class:`FunctionScorer` — wrap any ``item -> float`` callable + direction.
- :func:`rank` / :func:`best` — direction-aware ordering.

Learned scorers that *compute* a value (ESM perplexity, ProteinMPNN
log-likelihood, engine re-scoring) are follow-ups implementing the same
:class:`Scorer` ABC.
"""

from __future__ import annotations

from molforge.scoring._base import Direction, Score, Scorer, best, rank
from molforge.scoring.scorers import ConfidenceScorer, DockingScorer, FunctionScorer

__all__ = [
    "ConfidenceScorer",
    "Direction",
    "DockingScorer",
    "FunctionScorer",
    "Score",
    "Scorer",
    "best",
    "rank",
]
