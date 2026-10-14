"""A rolled-up structure-quality report — MolProbity in spirit.

:mod:`molforge.structure` ships the individual geometric checks — steric
clashes, Ramachandran classification, Cα chirality, backbone bond lengths.
:func:`report` runs all four, grades each against a threshold, and rolls
them into one :class:`QualityReport` so folding / docking output can be
gated on geometry in a single call::

    from molforge.validation import report

    q = report(protein)
    if not q.passed:
        print(q.summary())

Not a MolProbity *number*: the published MolProbity score combines
clashscore, Ramachandran, and *rotamer* outliers, and molforge has no
rotamer analysis. So :attr:`QualityReport.score` is the fraction of checks
that pass (0-1), and :attr:`QualityReport.passed` (all checks pass) is the
real gate — MolProbity-style, not a spoofed MolProbity value.

Thresholds are documented defaults, overridable per call via
``thresholds=``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from molforge.structure import (
    check_bond_lengths,
    chirality_outliers,
    clash_score,
    classify_ramachandran,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from molforge.core import Protein

__all__ = ["QualityCheck", "QualityReport", "report"]

#: Default pass thresholds, overridable via ``report(..., thresholds=...)``.
DEFAULT_THRESHOLDS: dict[str, float] = {
    # Clashes per 1000 atoms; well-refined structures are near 0.
    "clashscore": 20.0,
    # Fraction of classifiable residues in the disallowed region.
    "ramachandran_outlier_fraction": 0.02,
    # Cα chirality outliers — any inverted centre is a hard error.
    "chirality_outliers": 0.0,
    # Backbone bond-length outliers (> 4σ from ideal).
    "bond_length_outliers": 0.0,
}


@dataclass(frozen=True)
class QualityCheck:
    """One geometric check within a :class:`QualityReport`.

    Attributes:
        name: Check identifier (e.g. ``"clashscore"``).
        value: The raw metric value.
        threshold: The value it was graded against.
        passed: Whether ``value`` is within ``threshold`` (``<=``).
        detail: A short human-readable description of the finding.
    """

    name: str
    value: float
    threshold: float
    passed: bool
    detail: str


@dataclass(frozen=True)
class QualityReport:
    """A rolled-up structure-quality result over several checks.

    Attributes:
        checks: The individual :class:`QualityCheck`s (clashscore,
            ramachandran, chirality, bond_length).
        passed: ``True`` iff every check passed — the geometry gate.
        score: Fraction of checks that passed, in ``[0, 1]``.
    """

    checks: list[QualityCheck]
    passed: bool
    score: float

    def __len__(self) -> int:
        return len(self.checks)

    def __iter__(self) -> Iterator[QualityCheck]:
        return iter(self.checks)

    def __getitem__(self, name: str) -> QualityCheck:
        """Look up a check by name (e.g. ``report["clashscore"]``)."""
        for check in self.checks:
            if check.name == name:
                return check
        raise KeyError(name)

    def summary(self) -> str:
        """A compact multi-line summary — one line per check."""
        head = f"quality: {'PASS' if self.passed else 'FAIL'} (score {self.score:.2f})"
        lines = [head]
        for c in self.checks:
            mark = "ok" if c.passed else "FAIL"
            lines.append(f"  [{mark}] {c.name}: {c.detail}")
        return "\n".join(lines)


def report(protein: Protein, *, thresholds: dict[str, float] | None = None) -> QualityReport:
    """Roll up the structure-quality checks for ``protein``.

    Runs clashscore, Ramachandran (outlier fraction), Cα chirality, and
    backbone bond-length checks; grades each against its threshold; and
    returns a :class:`QualityReport` whose ``passed`` is the all-checks gate.

    Args:
        protein: The structure to grade.
        thresholds: Overrides for any of :data:`DEFAULT_THRESHOLDS`
            (``"clashscore"``, ``"ramachandran_outlier_fraction"``,
            ``"chirality_outliers"``, ``"bond_length_outliers"``). Missing
            keys keep their default.

    Returns:
        A :class:`QualityReport`.
    """
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    checks = [
        _clash_check(protein, limits["clashscore"]),
        _ramachandran_check(protein, limits["ramachandran_outlier_fraction"]),
        _chirality_check(protein, limits["chirality_outliers"]),
        _bond_length_check(protein, limits["bond_length_outliers"]),
    ]
    n_passed = sum(1 for c in checks if c.passed)
    return QualityReport(
        checks=checks,
        passed=all(c.passed for c in checks),
        score=n_passed / len(checks),
    )


# ---------- individual checks ----------


def _clash_check(protein: Protein, threshold: float) -> QualityCheck:
    value = clash_score(protein)
    return QualityCheck(
        name="clashscore",
        value=value,
        threshold=threshold,
        passed=value <= threshold,
        detail=f"{value:.1f} clashes / 1000 atoms (<= {threshold:g})",
    )


def _ramachandran_check(protein: Protein, threshold: float) -> QualityCheck:
    results = classify_ramachandran(protein)
    n = len(results)
    n_outliers = sum(1 for r in results if r.classification == "Outlier")
    n_favored = sum(1 for r in results if r.classification == "Favored")
    fraction = n_outliers / n if n else 0.0
    favored_pct = (n_favored / n * 100.0) if n else 100.0
    return QualityCheck(
        name="ramachandran",
        value=fraction,
        threshold=threshold,
        passed=fraction <= threshold,
        detail=f"{n_outliers}/{n} outliers ({fraction:.1%}), {favored_pct:.0f}% favored",
    )


def _chirality_check(protein: Protein, threshold: float) -> QualityCheck:
    n = len(chirality_outliers(protein))
    return QualityCheck(
        name="chirality",
        value=float(n),
        threshold=threshold,
        passed=n <= threshold,
        detail=f"{n} Cα chirality outlier(s)",
    )


def _bond_length_check(protein: Protein, threshold: float) -> QualityCheck:
    n = len(check_bond_lengths(protein))
    return QualityCheck(
        name="bond_length",
        value=float(n),
        threshold=threshold,
        passed=n <= threshold,
        detail=f"{n} backbone bond-length outlier(s) (> 4 sigma)",
    )
