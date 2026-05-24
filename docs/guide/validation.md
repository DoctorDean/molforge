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

!!! note "Error handling"
    `cross_validate` defaults to `on_error="raise"` — an exception
    raised by any validator propagates immediately and aborts the
    run. This is deliberate: a validator that throws is almost
    always a bug (a misconfigured engine, a missing dependency, a
    bad input), and silently swallowing it would produce a tidy
    list of `passed=False` verdicts that *looks* like a real
    result. For long batch jobs where one bad design shouldn't
    abort the whole screen, pass `on_error="record"` explicitly:
    failures are then captured under
    `verdict.metadata["validator_errors"]` and that verdict is
    marked `passed=False`, but the run continues.

    *Changed in 0.2: the default flipped from `"record"` to
    `"raise"`. Code relying on the old fault-tolerant default must
    now pass `on_error="record"` explicitly.*

## Reference

- [`molforge.validation`](../reference/validation.md) — full API.
