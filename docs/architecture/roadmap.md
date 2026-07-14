# Roadmap

The goal: molforge as the layer that lets a researcher, a startup, or a
big-enterprise pipeline glue together folding, docking, MD, and generative
design across many engines without rewriting the boilerplate each time.

## Where we are

molforge has **breadth**. Sixteen engine wrappers across six modalities
(folding, docking, MD, generative design, binding free energy, pocket
detection); a canonical data model with `Provenance` and a
content-addressed cache; a from-scratch analysis stack (RMSD, SASA, DSSP,
contacts, dihedrals, superposition, plus TM-score / GDT / lDDT / DockQ);
structure-quality validation (clashes, Ramachandran, chirality,
bond-length); remote data ingestion (RCSB / AlphaFold DB / ChEMBL); and ML
featurization. It ships on PyPI with docs.

So the next chapters are **not more engines.** They are:

1. **Trust** — verify the breadth that already exists. The 0.6.x
   correctness pass found that several from-scratch algorithms *and* the
   green-on-mocks engine tests were hiding real bugs, and that CI wasn't
   even running on the default branch. Verification is now the
   highest-leverage work in the project.
2. **Identity** — the glue nobody else owns: engine-agnostic cross-engine
   ensembles, a real design loop, and reproducible pipeline emission.
   molforge already has every underlying piece; this is integration, not
   new science.

Everything else — more engines, ML layers, performance hotpaths — is
**opportunistic**: pulled in when a real user needs it, not pushed to fill
a matrix.

## Now / Next / Later

- **Now — Trust.** Reference-value tests for the numerical stack; the
  nightly real-engine CI; clear the remaining correctness-audit items.
- **Next — Identity.** `DesignLoop`, cross-engine ensembles, reproducible
  `pipeline.yaml` emission, and a documented plugin path.
- **Later — opportunistic breadth.** More engines, local MSA, ML/data
  layers, performance work — pulled forward by demand, not by the matrix.

## Principles

- **Interop over reimplementation.** Where Biotite, ProDy, BioPython,
  MDAnalysis, or RDKit already solve a problem well, molforge reads and
  writes their formats rather than competing.
- **Heavy deps are opt-in extras.** GROMACS, AMBER, RDKit, torch,
  PyTorch-Geometric come through the appropriate extra; molforge itself
  stays light.
- **GPUs assumed for serious work**, but every GPU-only path carries a
  runtime warning and a documented CPU fallback where one is feasible.

---

## 1. Trust & verification  — *Now*

The dominant risk is not missing features; it's unverified ones. A
structural-bioinformatics library lives or dies on whether its numbers
match the literature and its wrappers still drive the real tools.

- **CI actually runs — done.** The pipeline (lint, strict mypy, the
  3-OS × 3-Python test matrix, build, notebooks) now triggers on `master`;
  for a long stretch it silently ran on a non-existent `main` branch and
  gated nothing. Accumulated lint/format debt was cleared and the ruff
  version pinned to match pre-commit.
- **Reference-value guards — started, extend.** The metric fixes were
  validated against independent oracles (TM-align via `tmtools`, DSSP via
  `mdtraj`) and those golden values are now committed as fast regression
  tests on a real structure (ubiquitin). Extend the same treatment to
  lDDT and DockQ against published reference values, and to the sequence
  aligners.
- **Nightly real-engine smoke tests — started, extend.** An opt-in
  nightly runs the CPU-installable engines against their *real*
  implementations, not mocks — the paths the per-push suite can't reach.
  The docking-prep path (RDKit + meeko + gemmi) is live and already
  earned its keep by exposing a missing runtime dependency. Extend
  coverage to Vina, and pin/version-check the fragile output-parsing
  seams (ESMFold's model→PDB conversion; the filename-glob result parsers
  in Boltz / Chai / DiffDock / ProteinMPNN) that a new engine release
  could silently break.
- **Correctness audit — mostly cleared, finish it.** Fixed in 0.6.x:
  Smith-Waterman affine-gap traceback, PQR charge/radius parsing, DockQ
  Fnat (now residue-level) and iRMS (robust to missing backbone atoms),
  DSSP amide-hydrogen geometry (+ B-bridge assignment), TM-score/GDT
  (now maximized over superpositions, not a single Kabsch fit), and the
  ML featurizer residue-set misalignment. Remaining: the thin `chem`
  descriptor set (add logP / TPSA / HBD / HBA / rotatable-bonds / Ro5 via
  RDKit — cheap and high-value for the comp-bio audience), and the lDDT
  all-atom variant alongside the current CA-only one.

## 2. Distinguishing identity  — *Next*

The long-horizon items that give molforge an identity nobody else owns.
Build them on the now-trustworthy base.

- **Engine-agnostic cross-engine ensembles.** "Fold this with ESMFold,
  AlphaFold, Boltz, and RoseTTAFold; show me the ensemble and the spread."
  Everyone else does "pick one and run." molforge already has the
  wrappers, the common data model, and the metrics — this is the glue.
- **`DesignLoop` tooling.** The protein-engineering loop (generate →
  fold → dock → score → iterate) is what real labs do, and nothing glues
  the parts cleanly. A `DesignLoop` with sane defaults, logging, and a
  ranked design table would be unique — and every piece already exists as
  a wrapper.
- **Reproducibility / `pipeline.yaml`.** Most papers in this space don't
  ship reproducible code. If molforge can run a workflow and emit a file
  that fully describes it — engine versions, weight hashes, parameters,
  the whole provenance chain — that artifact becomes the citable thing.
  Provenance already records the chain; this is the emit/replay layer on
  top.
- **Plugin ecosystem.** `molforge.plugins` exists. Documenting it well,
  shipping a template repo, and writing one or two example third-party
  plugins would seed an ecosystem. Compare napari: its plugin system is
  most of its value.

## 3. Workflow primitives

molforge has good *components*; these make chaining them ergonomic.

- **Parallelism primitives — Next.** `dock_many`, `fold_many`, `run_many`
  taking a list of inputs and a parallelism level, with each engine
  declaring whether it parallelizes across processes (CPU) or within one
  (GPU). Every user writes the same `multiprocessing.Pool` loop today;
  this is a natural precursor to cross-engine ensembles and `DesignLoop`.
- **Provenance — shipped.** `molforge.core.Provenance`: a frozen
  dataclass (engine / version / parameters / inputs / recursive parent),
  JSON-round-trippable, on `metadata[PROVENANCE]`, adopted across every
  wrapper and the prep pipelines. Optional polish (sidecar persistence,
  deeper engine-version introspection) is deferred.
- **Caching — shipped.** Content-addressed result cache keyed on
  `(engine, parameters, inputs, parent_chain)` with cascading
  invalidation; folding, sequence design, and free-energy results
  participate. Docking-engine and MD-trajectory caches remain follow-ups
  (MD trajectories deliberately uncached — multi-GB; use upstream
  checkpointing).
- **A `Pipeline` / DAG builder — deferred, probably don't.** Open
  question below; the default answer is to compose with
  Prefect/Hydra/Snakemake rather than build another runtime.

## 4. Depth where it counts

Deeper, not wider — on the things users actually gate on.

- **Structure-quality validation — shipped.** Clash detection,
  Ramachandran classification, Cα chirality, and backbone bond-length
  checks landed in 0.6.0, so folding/docking output can be gated on
  geometry. A MolProbity-style rolled-up `validate.report(protein)` that
  combines them into one score is the natural next step.
- **Binding free energy — shipped.** MM-PB(GB)SA via AmberTools and
  gmx_MMPBSA (with per-residue decomposition), plus FEP/TI ingestion via
  alchemlyb and cinnabar network ingestion. Follow-ups: Boltz-2 affinity
  prediction, and running (not just ingesting) an FEP calculation.
- **Unified scoring.** A `molforge.scoring` layer exposing docking
  scorers (Vina, Gnina CNN) and learned scorers (ESM perplexity,
  ProteinMPNN confidence) as a common interface, so users can score *any*
  structure with *any* scorer — decoupled from the docking wrappers.
- **Enhanced sampling — later.** PLUMED metadynamics, replica exchange,
  MELD. Heavy, but what serious MD users do.
- **Pocket detection.** fpocket is shipped; P2Rank (the ML-based modern
  counterpart) is the natural next one when its install path is cleaner —
  it drops into the same Pocket-dataclass + detector + provenance shape.

## 5. Engine matrix — *frozen*

The engine ABCs each have ≥2 real implementations. **Resist adding more
until the verification story (§1) covers what's already here** — a new
wrapper that's green-on-mocks widens the trust gap rather than closing it.
Kept here as a demand-driven wishlist, not a plan:

- **Folding:** ESM3, AlphaFold-3 (DeepMind), Protenix. Modifications,
  restraints, per-entity MSAs on the existing multi-component path.
- **Docking:** AutoDock-GPU, Uni-Dock (GPU-accelerated Vina variants).
- **MD:** NAMD, LAMMPS (non-bio workloads).
- **Generative:** LigandMPNN, Chroma, Protpardelle.
- **MSA / sequence search:** local `mmseqs2` / `hmmer` / `jackhmmer` — the
  one genuinely-wanted gap, since folding wrappers currently lean on
  ColabFold's MSA server. Serious users need a local path; this one may
  jump to *Next* on demand.

## 6. Opportunistic — ML, performance, I/O

Pulled forward by real demand, not pushed.

- **ML / data.** A single `molforge.ml.embed(protein, model=...)` for
  ESM-2/3, ProtT5, Ankh (ESM-2 embeddings already ship); structure
  tokenizers (FoldSeek 3Di, ESM-3 structure tokens); sequence-identity-
  respecting dataset splits (people get these wrong constantly); a shared
  `GenerativeBackbone` interface over the diffusion engines.
- **Performance.** Numba/Rust hotpaths for the benchmark-identified
  candidates (DSSP, pairwise RMSD, contact maps); optional torch-backed
  paths for embarrassingly-parallel ensemble ops; async engine calls.
- **Error taxonomy.** Replace bare `RuntimeError`-with-stderr in the
  wrappers with `EngineConfigError` / `ResourceError` / `ConvergenceError`
  / `OutOfMemoryError` so users can catch what they want to retry on.

## 7. Documentation

- **Cookbook + engine comparison tables — shipped.** Task-oriented
  recipes and decision tables live under `docs/cookbook/`.
- **Performance benchmarks page.** Publish the benchmark-suite numbers so
  users know what to expect.
- **Migration guides.** "Coming from BioPython / Biotite / MDAnalysis" —
  lower the switching cost.

---

## Open questions

- **`Pipeline` class — build our own or compose with Prefect/Hydra?**
  Default: compose. Revisit only if the friction is real.
- **Plugin ecosystem strategy.** Seed it ourselves, or wait for organic
  contributors? Seeding is cheaper than it looks once one good example
  exists.
- **Reproducibility format.** YAML? JSON? Something runnable? The
  ecosystem hasn't converged — watch what Chai / Boltz / AlphaFold settle
  on before committing.
