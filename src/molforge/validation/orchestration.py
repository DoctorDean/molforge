"""Cross-validation orchestration: run designs across validators, merge results.

Two main entry points:

- :func:`cross_validate` — given designs and a list of "validators"
  (callables that produce metric dicts), run every design through
  every validator and collect verdicts.
- :func:`consensus` — given verdict lists from multiple validators
  that have already been run, combine them under a chosen rule
  (``all`` / ``any`` / ``majority`` / count threshold).

The contract for a validator: any callable taking a design (whatever
shape — a :class:`Protein`, a :class:`DesignedSequence`, a sequence
string) and returning a flat dict of metric values. Validators can
also raise on bad input; failures are caught and recorded as a Verdict
with ``passed=False`` rather than killing the run.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from molforge.validation.criteria import CriteriaSet
from molforge.validation.verdict import Verdict

# A validator takes a design and returns a dict of metric values.
Validator = Callable[[Any], Mapping[str, Any]]


def cross_validate(
    designs: Iterable[Any],
    *,
    validators: Mapping[str, Validator],
    criteria: CriteriaSet,
    design_id: Callable[[Any], str] | None = None,
    score_metric: str | None = None,
    on_error: str = "record",
) -> list[Verdict]:
    """Run every design through every validator; collect verdicts.

    Args:
        designs: Iterable of designs (typically a list of Protein /
            DesignedSequence / sequence strings).
        validators: Dict mapping validator name to a callable that
            consumes a design and returns a dict of metric values.
            Validator names are prepended to each metric key so values
            from different validators don't collide in the verdict
            dict — e.g. ``validators={"esmfold": fn}`` produces keys
            like ``"esmfold.plddt"``.
        criteria: The :class:`CriteriaSet` to apply. Should reference
            metric keys in their fully-qualified form
            (``"esmfold.plddt"`` not just ``"plddt"``).
        design_id: Optional function to extract a string ID from each
            design. Defaults to ``str(design)`` truncated to 60 chars.
        score_metric: Name of the metric (after qualification) to use
            as the sortable :attr:`Verdict.score`. If ``None``, the
            score is the count of failed criteria (so passed designs
            sort to the front).
        on_error: How to handle validator exceptions:
            - ``"record"`` (default): record the failure in metadata
              and mark the verdict as failed.
            - ``"raise"``: propagate the exception up.

    Returns:
        One :class:`Verdict` per design, in input order.

    Example::

        criteria = (
            CriteriaSet()
            .add("fold_quality", Criterion.gt("esmfold.plddt", 80))
            .add("backbone_match", Criterion.gt("esmfold.tm", 0.5))
            .add("rmsd_ok", Criterion.lt("esmfold.rmsd", 2.0))
        )

        def esmfold_validator(seq):
            predicted = esm_engine.predict(seq)
            return {
                "plddt": predicted.metadata["mean_confidence"],
                "tm":    tm_score(predicted, target_backbone),
                "rmsd":  rmsd(predicted, target_backbone, subset="ca"),
            }

        verdicts = cross_validate(
            designs=sequences,
            validators={"esmfold": esmfold_validator},
            criteria=criteria,
            score_metric="esmfold.plddt",
        )

        for v in rank_verdicts(verdicts, only_passed=True):
            print(v.design_id, v.values["esmfold.plddt"])
    """
    if on_error not in {"record", "raise"}:
        raise ValueError(f"on_error must be 'record' or 'raise', got {on_error!r}")
    if design_id is None:

        def design_id(design: Any) -> str:  # type: ignore[misc]
            s = str(design)
            return s if len(s) <= 60 else s[:57] + "..."

    designs_list = list(designs)
    verdicts: list[Verdict] = []

    for design in designs_list:
        did = design_id(design)
        values: dict[str, Any] = {}
        errors: dict[str, str] = {}

        # Run each validator, namespace its results
        for vname, validator in validators.items():
            try:
                result = validator(design)
            except Exception as e:
                if on_error == "raise":
                    raise
                errors[vname] = f"{type(e).__name__}: {e}"
                continue
            for k, v in result.items():
                values[f"{vname}.{k}"] = v

        # Evaluate criteria. We allow missing metrics to mark relevant
        # criteria as failed (rather than raising) so that a single
        # validator failure doesn't blow up the whole run.
        criteria_results: dict[str, bool] = {}
        for name, c in criteria.criteria.items():
            try:
                criteria_results[name] = c.evaluate(values)
            except KeyError:
                criteria_results[name] = False

        passed = all(criteria_results.values()) and not errors

        # Compute the sortable score
        if score_metric is not None and score_metric in values:
            try:
                score = float(values[score_metric])
            except (TypeError, ValueError):
                score = float("inf")
        else:
            # Number of failed criteria (passed verdicts sort to the front)
            score = float(sum(1 for ok in criteria_results.values() if not ok))

        metadata: dict[str, Any] = {}
        if errors:
            metadata["validator_errors"] = errors

        verdicts.append(
            Verdict(
                design_id=did,
                values=values,
                criteria_results=criteria_results,
                passed=passed,
                score=score,
                metadata=metadata,
            )
        )

    return verdicts


def consensus(
    verdict_lists: Mapping[str, Iterable[Verdict]],
    *,
    mode: str = "all",
    threshold: int | None = None,
) -> list[Verdict]:
    """Combine verdicts from multiple validators into a single consensus list.

    Each input list is the result of running the same set of designs
    through one validator. Designs are joined by ``design_id``, so
    every validator must have produced a verdict for every design
    (use :func:`cross_validate` with multiple validators, or run each
    separately with consistent IDs).

    Args:
        verdict_lists: Dict mapping validator name to its list of
            Verdicts. The dict keys are folded into the merged
            ``values`` so per-validator metrics stay distinguishable.
        mode: Consensus rule:
            - ``"all"`` (default): every validator must mark the
              design as passed.
            - ``"any"``: any validator suffices.
            - ``"majority"``: more than half of validators must pass.
            - ``"threshold"``: at least ``threshold`` validators must
              pass (provide ``threshold``).
        threshold: Required when ``mode="threshold"``; the minimum
            number of validators that must mark the design as passed.

    Returns:
        A list of consensus :class:`Verdict` instances, one per design,
        with metrics from every validator merged into ``values`` and
        ``passed`` set per the chosen rule.

    Raises:
        ValueError: If validator verdict lists disagree on design IDs,
            or for invalid mode/threshold combos.

    Example::

        esm_verdicts = cross_validate(seqs, validators={"esmfold": fn1}, criteria=c)
        af_verdicts  = cross_validate(seqs, validators={"alphafold": fn2}, criteria=c)

        # Only accept designs both folding models agree are good
        joint = consensus(
            {"esm": esm_verdicts, "af": af_verdicts},
            mode="all",
        )
    """
    if mode not in {"all", "any", "majority", "threshold"}:
        raise ValueError(
            f"unknown mode {mode!r}; expected 'all', 'any', 'majority', or 'threshold'"
        )
    if mode == "threshold" and threshold is None:
        raise ValueError("threshold mode requires threshold=<int>")
    if mode != "threshold" and threshold is not None:
        raise ValueError(f"threshold is only valid with mode='threshold' (got mode={mode!r})")

    # Materialize and validate: every list must have the same design IDs
    lists = {name: list(verdicts) for name, verdicts in verdict_lists.items()}
    if not lists:
        return []

    id_sets = {name: {v.design_id for v in vs} for name, vs in lists.items()}
    common = set.intersection(*id_sets.values())
    union = set.union(*id_sets.values())
    if common != union:
        missing = {name: union - ids for name, ids in id_sets.items() if union - ids}
        raise ValueError(
            f"validator verdict lists disagree on design IDs; missing per validator: {missing}"
        )

    # Build a per-validator lookup
    by_id: dict[str, dict[str, Verdict]] = {did: {} for did in common}
    for vname, vs in lists.items():
        for verdict in vs:
            by_id[verdict.design_id][vname] = verdict

    n_validators = len(lists)
    if mode == "majority":
        required = n_validators // 2 + 1
    elif mode == "threshold":
        required = threshold  # type: ignore[assignment]
        if required < 1 or required > n_validators:
            raise ValueError(f"threshold={required} out of range for {n_validators} validators")
    elif mode == "all":
        required = n_validators
    else:  # any
        required = 1

    # Preserve input order: use the first list as the canonical order
    first = next(iter(lists.values()))
    consensus_verdicts: list[Verdict] = []
    for v in first:
        did = v.design_id
        per_validator = by_id[did]

        merged_values: dict[str, Any] = {}
        merged_criteria: dict[str, bool] = {}
        per_validator_passed: list[bool] = []
        for vname, vd in per_validator.items():
            for k, val in vd.values.items():
                merged_values[k] = val
            for cname, ok in vd.criteria_results.items():
                # Namespace criterion results by validator so they don't collide
                merged_criteria[f"{vname}.{cname}"] = ok
            per_validator_passed.append(vd.passed)

        n_passed = sum(per_validator_passed)
        passed = n_passed >= required

        merged_metadata = {
            "consensus_mode": mode,
            "n_validators": n_validators,
            "n_passed": n_passed,
            "required": required,
        }

        # Score: number of validators that *failed* (lower is better)
        score = float(n_validators - n_passed)

        consensus_verdicts.append(
            Verdict(
                design_id=did,
                values=merged_values,
                criteria_results=merged_criteria,
                passed=passed,
                score=score,
                metadata=merged_metadata,
            )
        )

    return consensus_verdicts


__all__ = [
    "Validator",
    "consensus",
    "cross_validate",
]
