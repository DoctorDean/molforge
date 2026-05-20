"""Declarative criteria for protein-design quality assessment.

A :class:`Criterion` describes "what counts as a successful design"
in terms of a metric name, a comparison operator, and a threshold.
Criteria compose with logical operators (``&``, ``|``, ``~``) so
complex success conditions can be expressed compactly:

    # Success = pLDDT > 80 AND TM-score > 0.5 AND RMSD < 2.0
    success = (
        Criterion.gt("plddt", 80.0)
        & Criterion.gt("tm_score", 0.5)
        & Criterion.lt("rmsd", 2.0)
    )

    # Pass if either folding model confirms (pLDDT > 80)
    fold_ok = (
        Criterion.gt("esmfold_plddt", 80.0)
        | Criterion.gt("alphafold_plddt", 80.0)
    )

Criteria are evaluated against a flat dict of metric values, so the
same criterion can be reused across designs without per-design setup.
"""

from __future__ import annotations

import operator as _op
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

# Standard operator names paired with their callable + display symbol.
_OP_TABLE: dict[str, tuple[Callable[[Any, Any], bool], str]] = {
    "gt": (_op.gt, ">"),
    "ge": (_op.ge, ">="),
    "lt": (_op.lt, "<"),
    "le": (_op.le, "<="),
    "eq": (_op.eq, "=="),
    "ne": (_op.ne, "!="),
}


class Criterion:
    """Declarative success criterion: a metric name + comparison + threshold.

    Construct via the factory classmethods :meth:`gt`, :meth:`ge`,
    :meth:`lt`, :meth:`le`, :meth:`eq`, :meth:`ne` rather than
    instantiating directly — they make the intent obvious in the
    calling code.

    Compose with the standard logical operators::

        a & b   # both must pass
        a | b   # either must pass
        ~a      # invert (passes when ``a`` would fail)
    """

    __slots__ = ("_describe", "_evaluate", "_metric_names")

    def __init__(
        self,
        evaluate: Callable[[Mapping[str, Any]], bool],
        describe: Callable[[], str],
        metric_names: frozenset[str],
    ) -> None:
        # Internal constructor — most users go through the classmethods.
        self._evaluate = evaluate
        self._describe = describe
        self._metric_names = metric_names

    # ------------------------------------------------------------------
    # Factory classmethods (the public construction API)
    # ------------------------------------------------------------------
    @classmethod
    def _atomic(cls, metric: str, op_name: str, threshold: float) -> Criterion:
        if op_name not in _OP_TABLE:
            raise ValueError(f"unknown operator {op_name!r}; expected one of {sorted(_OP_TABLE)}")
        op_fn, symbol = _OP_TABLE[op_name]

        def evaluate(values: Mapping[str, Any]) -> bool:
            if metric not in values:
                raise KeyError(
                    f"criterion references metric {metric!r} but values "
                    f"dict only contains {sorted(values)}"
                )
            v = values[metric]
            if v is None:
                return False
            return bool(op_fn(v, threshold))

        return cls(
            evaluate=evaluate,
            describe=lambda: f"{metric} {symbol} {threshold}",
            metric_names=frozenset({metric}),
        )

    @classmethod
    def gt(cls, metric: str, threshold: float) -> Criterion:
        """``metric > threshold``."""
        return cls._atomic(metric, "gt", threshold)

    @classmethod
    def ge(cls, metric: str, threshold: float) -> Criterion:
        """``metric >= threshold``."""
        return cls._atomic(metric, "ge", threshold)

    @classmethod
    def lt(cls, metric: str, threshold: float) -> Criterion:
        """``metric < threshold``."""
        return cls._atomic(metric, "lt", threshold)

    @classmethod
    def le(cls, metric: str, threshold: float) -> Criterion:
        """``metric <= threshold``."""
        return cls._atomic(metric, "le", threshold)

    @classmethod
    def eq(cls, metric: str, value: Any) -> Criterion:
        """``metric == value``."""
        return cls._atomic(metric, "eq", value)

    @classmethod
    def ne(cls, metric: str, value: Any) -> Criterion:
        """``metric != value``."""
        return cls._atomic(metric, "ne", value)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, values: Mapping[str, Any]) -> bool:
        """Return True iff ``values`` satisfies this criterion.

        Args:
            values: Dict mapping metric names to their measured values.
                Must contain every metric this criterion references
                (see :attr:`metric_names`).

        Raises:
            KeyError: If a referenced metric is missing from ``values``.
                Missing metrics are treated as a programming error, not
                a failed criterion; if you want "missing = fail", filter
                upstream or use ``None`` explicitly (``None`` always
                fails an atomic comparison).
        """
        return self._evaluate(values)

    @property
    def metric_names(self) -> frozenset[str]:
        """Names of all metrics this criterion references."""
        return self._metric_names

    def __repr__(self) -> str:
        return f"Criterion({self._describe()!r})"

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------
    def __and__(self, other: Criterion) -> Criterion:
        """``self & other`` — both must pass."""
        if not isinstance(other, Criterion):
            return NotImplemented
        return Criterion(
            evaluate=lambda v: self._evaluate(v) and other._evaluate(v),
            describe=lambda: f"({self._describe()}) AND ({other._describe()})",
            metric_names=self._metric_names | other._metric_names,
        )

    def __or__(self, other: Criterion) -> Criterion:
        """``self | other`` — either must pass."""
        if not isinstance(other, Criterion):
            return NotImplemented
        return Criterion(
            evaluate=lambda v: self._evaluate(v) or other._evaluate(v),
            describe=lambda: f"({self._describe()}) OR ({other._describe()})",
            metric_names=self._metric_names | other._metric_names,
        )

    def __invert__(self) -> Criterion:
        """``~self`` — passes when ``self`` would fail."""
        return Criterion(
            evaluate=lambda v: not self._evaluate(v),
            describe=lambda: f"NOT ({self._describe()})",
            metric_names=self._metric_names,
        )


@dataclass(frozen=True)
class NamedCriterion:
    """A criterion paired with a human-readable name.

    Useful when you want diagnostics that say "design passed
    `fold_quality` but failed `solubility`" rather than dumping the
    full criterion expression.
    """

    name: str
    criterion: Criterion
    description: str = ""

    def evaluate(self, values: Mapping[str, Any]) -> bool:
        return self.criterion.evaluate(values)

    @property
    def metric_names(self) -> frozenset[str]:
        return self.criterion.metric_names


@dataclass
class CriteriaSet:
    """A named collection of criteria evaluated together.

    Each criterion is evaluated separately so per-criterion pass/fail
    is available in the resulting :class:`Verdict`. The overall verdict
    passes only if **all** named criteria pass (an implicit AND).
    """

    criteria: dict[str, Criterion] = field(default_factory=dict)

    def add(self, name: str, criterion: Criterion) -> CriteriaSet:
        """Add a named criterion. Returns self for chaining."""
        self.criteria[name] = criterion
        return self

    def evaluate(self, values: Mapping[str, Any]) -> dict[str, bool]:
        """Evaluate every criterion against ``values``.

        Returns a dict mapping criterion name to its pass/fail result.
        """
        return {name: c.evaluate(values) for name, c in self.criteria.items()}

    def passes(self, values: Mapping[str, Any]) -> bool:
        """True iff every criterion passes."""
        return all(self.evaluate(values).values())

    @property
    def metric_names(self) -> frozenset[str]:
        """Union of metric names across all criteria."""
        if not self.criteria:
            return frozenset()
        result: frozenset[str] = frozenset()
        for c in self.criteria.values():
            result = result | c.metric_names
        return result


__all__ = [
    "CriteriaSet",
    "Criterion",
    "NamedCriterion",
]
