# Validation

`molforge.validation` is a small toolkit for deciding whether a
protein candidate — a design, a fold, a docked pose — passes the
bar for downstream work. It's intended for design-loop workflows
where you generate many candidates and need a principled way to
say *"keep these, drop those."*

The three pieces:

- **Criteria** — predicates over a `Protein` (or an analysis result)
  that return pass/fail.
- **Verdicts** — the outcome of running a `CriteriaSet` against a
  candidate.
- **Orchestration** — `cross_validate` runs multiple independent
  validators (e.g. two folding engines) and combines them;
  `consensus` reduces multiple verdicts into one.

## A small example

```python
from molforge.validation import Criterion, CriteriaSet, NamedCriterion

criteria = CriteriaSet([
    NamedCriterion("min_plddt",
                   Criterion(lambda p: p.metadata["confidence_per_residue"].mean() >= 0.7)),
    NamedCriterion("no_long_loops",
                   Criterion(lambda p: max_loop_length(p) <= 20)),
])

verdict = criteria.evaluate(candidate)
if verdict.passed:
    save(candidate, "keepers/")
```

## Cross-engine validation

A common pattern: fold a designed sequence with *two* independent
engines and only keep candidates where both agree it folds well.

```python
from molforge.validation import cross_validate
from molforge.wrappers.folding import ESMFold, AlphaFold

verdicts = cross_validate(
    candidates,
    validators={
        "esmfold":   lambda seq: criteria.evaluate(ESMFold().predict(seq)),
        "alphafold": lambda seq: criteria.evaluate(AlphaFold().predict(seq)),
    },
)
```

See the
[cross-engine validation notebook](https://github.com/DoctorDean/molforge/blob/main/notebooks/examples/cross_engine_validation.ipynb)
for a full walkthrough including consensus rules.

!!! note "API status"
    `cross_validate` defaults to `on_error="record"` — exceptions
    raised by individual validators are captured in the verdict
    rather than propagated. This is friendly for batch jobs but
    can mask bugs in your validators. Pass `on_error="raise"` while
    developing, switch to `"record"` for production runs. The
    default may flip in a future release; this is under
    [API audit](https://github.com/DoctorDean/molforge/issues).

## Reference

- [`molforge.validation`](../reference/validation.md) — full API.
