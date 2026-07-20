"""Cross-validation utilities for protein design.

This subpackage captures the pattern of scoring designs across one
or more validators and combining the results — the natural follow-on
to :mod:`molforge.metrics` for *de novo* design workflows.

Three core building blocks:

**Criteria** — declarative success conditions:
    - :class:`Criterion` — atomic comparison (e.g. ``pLDDT > 80``).
      Compose with ``&``, ``|``, ``~``.
    - :class:`NamedCriterion` — a criterion with a human-readable label.
    - :class:`CriteriaSet` — a named collection of criteria evaluated
      together (implicit AND across criteria; pass/fail captured per
      criterion for diagnostics).

**Verdicts** — per-design results:
    - :class:`Verdict` — the metric values, criterion results, and
      overall pass/fail for one design.
    - :func:`rank_verdicts` — sort verdicts by score, optionally
      filtering to only-passed.

**Orchestration**:
    - :func:`cross_validate` — run a list of designs through one or
      more validators (any callable that returns a metric dict),
      apply criteria, return verdicts.
    - :func:`consensus` — merge verdict lists across validators
      ("ESMFold AND AlphaFold both confirm").

Example::

    from molforge.validation import (
        cross_validate, consensus, rank_verdicts,
        Criterion, CriteriaSet,
    )

    success = (
        CriteriaSet()
        .add("plddt_ok", Criterion.gt("esmfold.plddt", 80))
        .add("tm_ok",    Criterion.gt("esmfold.tm", 0.5))
        .add("rmsd_ok",  Criterion.lt("esmfold.rmsd", 2.0))
    )

    verdicts = cross_validate(
        designs=sequences,
        validators={"esmfold": esmfold_score_fn},
        criteria=success,
        score_metric="esmfold.plddt",
    )

    for v in rank_verdicts(verdicts, only_passed=True):
        print(v.design_id, v.values["esmfold.plddt"])
"""

from __future__ import annotations

from molforge.validation.criteria import (
    CriteriaSet,
    Criterion,
    NamedCriterion,
)
from molforge.validation.orchestration import (
    Validator,
    consensus,
    cross_validate,
)
from molforge.validation.quality import (
    QualityCheck,
    QualityReport,
    report,
)
from molforge.validation.verdict import (
    Verdict,
    rank_verdicts,
)

__all__ = [  # noqa: RUF022 — grouped by concern
    # Criteria
    "Criterion",
    "NamedCriterion",
    "CriteriaSet",
    # Verdicts
    "Verdict",
    "rank_verdicts",
    # Orchestration
    "Validator",
    "cross_validate",
    "consensus",
    # Structure-quality report
    "report",
    "QualityReport",
    "QualityCheck",
]
