# Scoring

Every score in molforge is a bare float whose "good" direction you have to
know out of band:

| Source | Field | Better |
| ------ | ----- | ------ |
| Vina | `Pose.score` (kcal/mol) | **lower** |
| Gnina | `Pose.score` (CNN score) | **higher** |
| ProteinMPNN | `DesignedSequence.score` (NLL) | **lower** |
| Folding engines | `metadata["mean_confidence"]` (pLDDT) | **higher** |

The same `Pose.score` field is lower-is-better for Vina and higher-is-better
for Gnina. [`molforge.scoring`](../reference/scoring.md) makes the direction
explicit so you can rank, compare, and threshold across sources uniformly.

```python
from molforge.scoring import ConfidenceScorer, rank

best_first = rank(structures, ConfidenceScorer())
top = best_first[0][0]
```

## `Score` and `Direction`

A [`Scorer`](../reference/scoring.md) returns a `Score` carrying its
`Direction`. The key method is `ranking_key` — a "higher is always better"
view that negates lower-is-better values, so scores of either direction sort
and compare uniformly:

```python
from molforge.scoring import Score, Direction

affinity = Score(-9.5, Direction.LOWER_IS_BETTER)
plddt    = Score(87.0, Direction.HIGHER_IS_BETTER)

affinity.ranking_key            # 9.5  (negated)
affinity.is_better_than(Score(-7.0, Direction.LOWER_IS_BETTER))   # True
```

A `nan` value is never better than a real one, so unscoreable items sink in
a ranking rather than surfacing spuriously.

## The v1 scorers

All dependency-free — they read already-computed numbers or wrap a callable,
so scoring never re-runs a heavy engine.

**`ConfidenceScorer`** — a folded structure's mean pLDDT (higher is better):

```python
from molforge.scoring import ConfidenceScorer
score = ConfidenceScorer().score(protein)   # reads metadata["mean_confidence"]
```

**`DockingScorer`** — a `Pose` or `DockingResult`'s score, with the correct
direction. Because docking scores don't share a direction, read it from the
engine that produced the result:

```python
from molforge.scoring import DockingScorer

scorer = DockingScorer.from_engine(vina)     # -> lower_is_better
scorer.score(result)                         # scores the best pose
```

The engines expose this via a `score_direction` attribute (Vina / DiffDock
are `"lower_is_better"`; Gnina follows its `sort_order`).

**`FunctionScorer`** — wrap any `item -> float` callable with a direction.
The escape hatch for engine re-scoring, an ESM perplexity call, or a bespoke
composite:

```python
from molforge.scoring import FunctionScorer, Direction

esm = FunctionScorer(esm_perplexity, direction=Direction.LOWER_IS_BETTER, name="esm_ppl")
```

## Ranking helpers

`rank(items, scorer)` returns `(item, score)` pairs best-first; `best(items,
scorer)` returns the single winner. Both are direction-aware and put `nan`
scores last. `Scorer.score_many(items)` scores a batch (serial by default;
see [`molforge.parallel`](../reference/parallel.md)).

## Using a scorer as a design objective

Any `Scorer` plugs straight into
[`DesignLoop`](design.md) as an objective — it grades each candidate's
folded structure by `ranking_key`, so higher is always better regardless of
the scorer's native direction:

```python
from molforge.design import DesignLoop
from molforge.scoring import ConfidenceScorer

loop = DesignLoop(designer=..., folder=..., objective=ConfidenceScorer())
```

## What v1 doesn't do

Learned scorers that *compute* a value — ESM perplexity, ProteinMPNN
log-likelihood, engine re-scoring (re-running Vina/Gnina) — are follow-ups
that implement the same `Scorer` ABC. For now, wrap them with a
`FunctionScorer`.
