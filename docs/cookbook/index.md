# Cookbook

Recipes for getting things done with molforge. Each recipe answers a
specific *task-oriented* question — "I want to do X, what do I write?"
— and shows a complete, runnable example. The
[User guide](../guide/data-model.md) covers concepts; this section
covers concrete workflows.

## If you want to...

| Task                                                            | Recipe                                                    |
| --------------------------------------------------------------- | --------------------------------------------------------- |
| Predict a structure from a sequence                             | [Fold a sequence](folding.md)                             |
| Predict a protein + ligand or multi-chain complex               | [Multi-component cofolding](multi-component-folding.md)   |
| Dock a small molecule against a folded receptor                 | [Fold then dock](folding-then-docking.md)                 |
| Get a raw PDB ready for MD simulation                           | [Prepare for MD](prep-for-md.md)                          |
| Run a short MD simulation and analyse it                        | [MD and RMSD](md-and-rmsd.md)                             |
| Design sequences for a backbone, then validate by re-folding    | [Design then refold](design-then-refold.md)               |
| Trace what produced an output across a multi-step workflow      | [Inspect provenance](inspect-provenance.md)               |
| Check a folded or docked structure for quality problems         | [Validate structures](validating-structures.md)           |
| Skip recomputing expensive engine calls you've already run      | [Caching results](caching-results.md)                     |
| Rank a series of analogs by binding affinity (MM/GBSA)          | [Rank binders with MM/GBSA](ranking-binders.md)           |
| Rank analogs by rigorous relative affinity (FEP, via alchemlyb) | [Rank binders with FEP](rank-binders-fep.md)              |
| Compute an absolute binding free energy (ABFE, via alchemlyb)   | [Absolute binding free energy with FEP](absolute-binding-fep.md) |
| Ingest, clean, dedup, and filter a set of small molecules       | [Work with small molecules](small-molecules.md)           |
| Fetch or search structures (RCSB/AlphaFold) and compounds (ChEMBL) | [Fetch and search databases](fetch-and-search.md)      |

## If you're choosing between options...

| Decision                                       | Comparison                                          |
| ---------------------------------------------- | --------------------------------------------------- |
| Which folding engine should I use?             | [Folding engines](choosing-folding.md)              |
| Which docking engine should I use?             | [Docking engines](choosing-docking.md)              |
| Which generative engine for what task?         | [Generative engines](choosing-generative.md)        |

## How these recipes work

Every recipe is **structurally complete** — real imports, real method
signatures, real arguments — and will run as written *if you have the
dependencies for the engine it uses*. Most recipes need optional
extras:

- Folding via ESMFold needs `pip install "molforge[ml]"`, plus
  `torch` and a few GB of weights.
- Docking via Vina needs the `vina` Python package and Open Babel.
- MD via OpenMM needs `pip install "molforge[md,prep]"` and a working
  OpenMM install.

Each recipe states its requirements at the top so you know what you're
in for before you copy the code.

For shorter, more conceptual introductions, see the
[walkthroughs](../walkthroughs/01_sequences.ipynb). For exhaustive
worked examples that combine multiple engines, see the
[examples notebooks](../examples/index.md).
