# Design loop

Real protein engineering is a **loop**: propose sequences for a scaffold,
predict what they fold to, check whether the prediction actually matches
the scaffold you designed for, keep the winners, and design again from
them. Every piece already exists as a molforge wrapper — a sequence
designer, a folding engine, optionally a docking engine — but nothing
glues them into the loop. [`DesignLoop`](../reference/design.md) is that
glue.

```python
from molforge.design import DesignLoop
from molforge.wrappers.generative import ProteinMPNN
from molforge.wrappers.folding import ESMFold

loop = DesignLoop(designer=ProteinMPNN(), folder=ESMFold(), n_rounds=3)
table = loop.run(backbone)        # a Protein or a path to a PDB
best = table.best                 # highest-scoring design
rows = table.to_records()         # flat dicts → pandas.DataFrame(rows)
```

## The stages

`generate → fold → (dock) → score → iterate`. Each stage is an engine you
already wrap; the loop wires them together, runs the fan-outs in parallel,
logs per-round progress, and returns a ranked table.

| Stage        | Driven by            | Produces                              |
| ------------ | -------------------- | ------------------------------------- |
| **design**   | `designer.generate`  | candidate sequences for the backbone  |
| **fold**     | `folder.predict` (or [`cross_engine_fold`](../reference/ensembles.md)) | a predicted structure per candidate |
| **dock**     | `docker.dock` *(optional)* | a `DockingResult` per candidate |
| **score**    | the objective        | one number per candidate (higher = better) |
| **iterate**  | the loop             | redesign onto the previous round's winners |

## Scoring: the objective

The objective turns a candidate into a single number — higher is better.
Four options:

- **`"self_consistency"`** (default) — fold each designed sequence and
  measure how well it superposes on the backbone it was designed for,
  via `scTM` (and `scRMSD`). This is the metric the RFdiffusion /
  ProteinMPNN / AlphaFold pipelines are graded on: *did the sequence
  refold to the shape you asked for?* It falls straight out of the
  corrected [`tm_score`](../reference/metrics.md).
- **`"plddt"`** — mean folding confidence. No backbone correspondence
  needed; useful when you just want confidently-foldable sequences.
- **`"affinity"`** — the best docking score against `receptor` (negated so
  higher is better). Requires a `docker`.
- **a custom callable** — `Callable[[DesignCandidate], float]`, for
  weighted composites or anything bespoke:

```python
loop = DesignLoop(
    designer=ProteinMPNN(),
    folder=ESMFold(),
    objective=lambda c: c.metrics["sc_tm"] - 0.01 * c.metrics["mpnn_score"],
)
```

- **a [`Scorer`](scoring.md)** — any `molforge.scoring.Scorer` grades each
  candidate's folded structure by its `ranking_key` (higher is always
  better), e.g. `objective=ConfidenceScorer()`.

Every candidate records *all* the metrics it accumulated (`sc_tm`,
`sc_rmsd`, `plddt`, `mpnn_score`, `affinity`, …) in `candidate.metrics`,
regardless of which one the objective used — so you can re-rank or filter
after the fact.

## Cross-engine folding as the fold stage

Pass a **list** of folding engines and each candidate is folded with
[`cross_engine_fold`](ensembles.md#cross-engine-folding): the structure
used for scoring is the cross-engine consensus, and two extra confidence
signals are recorded — `cross_engine_tm_mean` (how much the engines agreed
on this design) and `cross_engine_rmsf_mean` (mean per-residue
disagreement). A design that every engine folds the same way is a more
trustworthy design.

```python
from molforge.wrappers.folding import ESMFold, AlphaFold, Boltz

loop = DesignLoop(designer=ProteinMPNN(), folder=[ESMFold(), AlphaFold(), Boltz()])
```

## Iteration

Iteration is genuine refinement, not just more sampling. Round *r+1*
re-designs onto the **folded structures** of the top `select_top`
candidates from round *r* — the scaffold evolves toward what actually
folds well:

```python
loop = DesignLoop(
    designer=ProteinMPNN(),
    folder=ESMFold(),
    n_designs=8,      # sequences proposed per backbone per round
    n_rounds=4,
    select_top=4,     # winners carried into the next round as new scaffolds
)
```

The [`DesignTable`](../reference/design.md) accumulates every candidate
from every round, ranked best-first — `.best`, `.top_n(k)`, iteration, and
`.to_records()` for a DataFrame.

## What v1 doesn't do

- **No backbone generation.** The `generator` slot (round-0 backbone
  generation, e.g. RFdiffusion) is part of the constructor signature but
  raises `NotImplementedError` — its configuration surface (contigs,
  targets, symmetry) is too engine-specific to wire generically yet. Pass
  a backbone (or a target you've already prepared) to `run()`.
- **The dock stage docks the folded structure against a fixed receptor.**
  It suits ligand / small-molecule design loops and protein-protein
  docking engines; pair a sensible docker with your design type.
