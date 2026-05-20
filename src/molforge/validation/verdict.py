"""Result types for cross-validation runs.

A :class:`Verdict` captures everything we know about a single design
after running it through one or more validators: the design itself,
the metric values measured, which criteria passed, and an overall
pass/fail flag. Verdicts are designed to be:

- **Sortable** by score for ranking designs.
- **Inspectable** for diagnostics (which criterion failed?).
- **Combinable** across validators via :func:`consensus`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


@dataclass
class Verdict:
    """The result of validating a single design.

    Attributes:
        design_id: An identifier for the design (sequence string,
            file path, index, etc.). Used as the join key in
            :func:`consensus` across validators.
        values: Dict of all metric values measured for this design.
            Keys are metric names; values are typically floats but
            can be anything criteria evaluate against.
        criteria_results: Dict mapping criterion name to its pass/fail
            result.
        passed: True iff every criterion passed (the overall verdict).
        score: A single scalar used for ranking. Smaller-is-better by
            convention (matches ProteinMPNN and most folding-confidence
            scores when negated); if you have a larger-is-better metric,
            store ``-value`` here.
        metadata: Engine-specific extras (validator name, runtime, etc.).
    """

    design_id: str
    values: dict[str, Any] = field(default_factory=dict)
    criteria_results: dict[str, bool] = field(default_factory=dict)
    passed: bool = False
    score: float = float("inf")
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def failed_criteria(self) -> list[str]:
        """Names of criteria that failed (empty if everything passed)."""
        return [n for n, ok in self.criteria_results.items() if not ok]

    @property
    def passed_criteria(self) -> list[str]:
        """Names of criteria that passed."""
        return [n for n, ok in self.criteria_results.items() if ok]

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        id_preview = self.design_id if len(self.design_id) <= 24 else self.design_id[:21] + "..."
        return f"Verdict({id_preview!r}, {status}, score={self.score:.3f})"


def rank_verdicts(
    verdicts: Iterable[Verdict],
    *,
    only_passed: bool = False,
    by: str | None = None,
) -> list[Verdict]:
    """Sort verdicts for ranking.

    Args:
        verdicts: Verdicts to sort.
        only_passed: If True, drop verdicts that didn't pass before
            sorting. Useful for "show me the successful designs".
        by: Sort key. ``None`` (default) sorts by ``verdict.score``
            ascending (lower = better). Otherwise sort by the named
            metric value in ``verdict.values``, ascending.

    Returns:
        A new list, sorted. Original order is preserved within
        equal-score groups (stable sort).
    """
    items = list(verdicts)
    if only_passed:
        items = [v for v in items if v.passed]
    if by is None:
        items.sort(key=lambda v: v.score)
    else:
        items.sort(key=lambda v: v.values.get(by, float("inf")))
    return items


__all__ = [
    "Verdict",
    "rank_verdicts",
]
