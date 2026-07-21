# End to end: design, cross-check, reproduce

This recipe ties the whole **Identity layer** together — the glue that
sets molforge apart from "pick one engine and run." In one script you:

1. **design** sequences for a backbone,
2. fold each candidate across **several engines at once** and keep the ones
   the engines *agree* on,
3. **score** and rank the survivors with a common yardstick,
4. inspect **where the engines disagree** on the winner, and
5. walk away with a citable, **replayable** record of exactly how you got
   there.

Every step is one object; molforge does the plumbing.

## Requirements

```bash
pip install "molforge[ml]"     # ESMFold (torch, transformers)
pip install "molforge[repro]"  # pipeline.yaml
# Plus the engines you fold/design with (ProteinMPNN, AlphaFold/ColabFold,
# Boltz) — see each wrapper's install notes. GPU strongly recommended.
```

## The recipe

```python
from molforge.io import fetch
from molforge.wrappers.generative import ProteinMPNN
from molforge.wrappers.folding import ESMFold, AlphaFold, Boltz
from molforge.design import DesignLoop
from molforge.scoring import ConfidenceScorer, rank
from molforge.ensembles import cross_engine_fold
from molforge.reproducibility import emit_pipeline

# ── 1. A target backbone ───────────────────────────────────────────────
backbone = fetch("1UBQ")            # or an RFdiffusion output, a native fold

# ── 2. Design → cross-engine fold → score → iterate ────────────────────
# Pass a *list* of folders: each designed sequence is folded by ESMFold AND
# AlphaFold, scored against the cross-engine consensus, and its per-residue
# engine disagreement is recorded. The loop keeps the winners and redesigns
# onto them for the next round.
loop = DesignLoop(
    designer=ProteinMPNN(),
    folder=[ESMFold(), AlphaFold()],       # cross-engine folding
    objective="self_consistency",          # scTM of the refold vs the backbone
    n_designs=8,
    n_rounds=3,
    select_top=4,
)
table = loop.run(backbone)                 # ranked DesignTable, best-first

best = table.best
print(best.sequence)
print({k: round(v, 3) for k, v in best.metrics.items()})
# sc_tm, sc_rmsd, plddt, mpnn_score,
# cross_engine_tm_mean  (how much the engines agreed on this design),
# cross_engine_rmsf_mean (mean per-residue disagreement)
```

The design table is directly rankable, and every candidate records *all*
its metrics — so you can re-rank on any of them.

```python
# ── 3. Score the top candidates on a common, direction-aware scale ─────
# ConfidenceScorer reads mean pLDDT; rank() sorts best-first regardless of
# whether the scorer is higher- or lower-is-better.
survivors = [c.structure for c in table.top_n(5)]
for structure, score in rank(survivors, ConfidenceScorer()):
    print(f"{score.value:5.1f}  {structure.name}")
```

Swap in any `Scorer` — a `DockingScorer` over poses, or a `FunctionScorer`
wrapping an ESM-perplexity call — and the ranking still works because every
`Score` carries its own direction.

```python
# ── 4. Zoom in on the winner: where do the engines agree? ──────────────
ensemble = cross_engine_fold(
    best.sequence,
    engines=[ESMFold(), AlphaFold(), Boltz()],
)
print(ensemble.spread())          # pairwise TM / RMSD summary across engines
disagreement = ensemble.disagreement()   # (L,) per-residue Cα spread
hot = [i for i, d in enumerate(disagreement) if d > 3.0]
print(f"{len(hot)} residues the engines can't agree on: {hot}")
```

`disagreement()` is a model-agnostic confidence signal: the residues all
three engines place in the same spot are the ones to believe.

```python
# ── 5. A reproducible, citable record ──────────────────────────────────
emit_pipeline(best.structure, "pipeline.yaml")
```

The manifest captures the full provenance chain — every engine, version,
and parameter — plus the environment. Anyone can inspect it, cite it, or
re-run it:

```python
from molforge.reproducibility import load_pipeline, replay

manifest = load_pipeline("pipeline.yaml")
print(manifest.describe())
output = replay(manifest)          # re-executes the chain, engines and all
```

## Why this is the point

Each piece exists in other libraries; **the composition is what molforge
owns.** One `Protein` flows through all five steps, one `Provenance` chain
records the whole thing, one cache makes the repeats instant — and you never
wrote a line of format-conversion or orchestration glue. That's the whole
thesis in one script.

- [Cross-engine folding](../guide/ensembles.md)
- [The design loop](../guide/design.md)
- [Scoring](../guide/scoring.md)
- [Reproducibility](../guide/reproducibility.md)
