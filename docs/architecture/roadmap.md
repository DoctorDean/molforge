# Roadmap

The goal: molforge as the layer that lets a researcher, a startup, or a
big-enterprise pipeline glue together folding, docking, MD, and generative
design across many engines without rewriting the boilerplate each time.

## Where we are

molforge has **breadth**. Nineteen engine wrappers across seven modalities
(folding, docking, MD, generative design, binding free energy, pocket
detection, homology search); a canonical data model with `Provenance` and a
content-addressed cache; a from-scratch analysis stack (RMSD, SASA, DSSP,
contacts, dihedrals, superposition, plus TM-score / GDT / lDDT / DockQ);
structure-quality validation (clashes, Ramachandran, chirality,
bond-length); remote data ingestion (RCSB / AlphaFold DB / ChEMBL / the
PDB chemical-component dictionary), cached; reproducibility that emits,
replays, and aggregates across many outputs; and ML featurization. It
ships on PyPI with docs.

So the next chapters are **not more engines.** They are:

1. **Trust** — verify the breadth that already exists. The 0.6.x
   correctness pass found that several from-scratch algorithms *and* the
   green-on-mocks engine tests were hiding real bugs, and that CI wasn't
   even running on the default branch. Verification is the
   highest-leverage work in the project.
2. **Identity** — the glue nobody else owns: engine-agnostic cross-engine
   ensembles, a real design loop, and reproducible pipeline emission.
   molforge already had every underlying piece; this was integration, not
   new science. **The Identity chapter is now shipped** (see §2).

Everything else — more engines, ML layers, performance hotpaths — is
**opportunistic**: pulled in when a real user needs it, not pushed to fill
a matrix.

## Now / Next / Later

The **Trust** (§1) and **Identity** (§2) chapters are both complete, and the
depth items that had concrete consumers — `molforge.scoring`,
`validate.report()`, P2Rank, Boltz-2 affinity, and the `pipeline.yaml`
*replay* layer — have all shipped.

- **Now — the FEP endeavour.** *Running* (not just ingesting) an FEP/TI
  calculation is the one large, high-value piece left in §4 — the
  orchestration that would make molforge genuinely all-encompassing. A real
  undertaking, and the only thing of that size still outstanding.
- **Next — nothing is queued.** Both items that sat here are done: the
  replay surface (every operation a wrapper records is now recorded *and*
  replayable, §3) and the local MSA path (§5), which shipped in 0.10.0 as
  `molforge.sequence.MultipleSequenceAlignment`, the `MMseqs2` wrapper, and
  `Entity.msa` honoured by both AF3-class engines.

  The obvious continuations, none of them started: `hmmer` / `jackhmmer`
  alongside MMseqs2 (needs a Stockholm reader), paired multi-chain MSAs
  built from the `pairing_key` the type already carries, and `DesignLoop`'s
  round-0 `generator` slot (§2). Each is demand-driven rather than
  scheduled.
- **Later — opportunistic breadth.** More engines, ML/data layers, enhanced
  sampling, performance work, the error taxonomy (§6) — pulled forward by
  demand, not by the matrix.

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

## 1. Trust & verification  — *Complete*

The dominant risk is not missing features; it's unverified ones. A
structural-bioinformatics library lives or dies on whether its numbers
match the literature and its wrappers still drive the real tools. This
chapter is **done**.

- **CI actually runs — done.** The pipeline (lint, strict mypy, the
  3-OS × 3-Python test matrix, build, notebooks) triggers on `master`; for
  a long stretch it silently ran on a non-existent `main` branch and gated
  nothing. Accumulated lint/format debt was cleared and the ruff version
  pinned to match pre-commit.
- **Reference-value guards — done.** Golden values computed offline against
  independent oracles and committed as fast regression tests, now covering
  **TM-score, GDT, DSSP** (via `tmtools` / `mdtraj`), **DockQ** (vs the
  reference DockQ package), **lDDT** (vs an independent from-scratch impl +
  a hand-computed case), and the **Needleman-Wunsch / Smith-Waterman
  aligners** (vs Biopython).
- **Nightly real-engine smoke tests — done.** An opt-in nightly runs the
  CPU-installable engines against their *real* implementations, not mocks —
  docking-prep (RDKit + meeko + gemmi), **Vina**, and chem descriptors. It
  earned its keep by exposing a missing runtime dependency (scipy for
  meeko).
- **Correctness audit — done.** Fixed across 0.6.x–0.7.0:
  Smith-Waterman affine-gap traceback, PQR charge/radius parsing, DockQ
  Fnat (residue-level) and iRMS, DSSP amide-hydrogen geometry (+ B-bridge),
  TM-score/GDT (maximized over superpositions), the ML featurizer residue-
  set misalignment, the `chem` descriptor set, the **lDDT all-atom
  variant**, and a **dihedral sign-convention bug** (φ/ψ were negated,
  breaking Ramachandran classification — caught vs Biopython).
- **Engine version guards — done.** Wrappers record the installed engine
  version in provenance and warn non-fatally on drift outside the tested
  range, so a new engine release that changes an output format is
  diagnosable. 0.9.0 added `molforge.engine_versions()`, a registry over
  every backend molforge can drive — including the native binaries that
  have no version to record at call time, which used to leave a manifest
  blank exactly where a reader wants a number.

## 2. Distinguishing identity  — *Shipped*

The long-horizon items that give molforge an identity nobody else owns,
built on the now-trustworthy base. **All four shipped:**

- **Engine-agnostic cross-engine ensembles — shipped.**
  `molforge.ensembles.cross_engine_fold`: fold one sequence with several
  engines (ESMFold / AlphaFold / Boltz / RoseTTAFold), superpose them, and
  get back a `CrossEngineEnsemble` — pairwise TM / RMSD matrices, a medoid
  consensus, and the per-residue cross-engine disagreement (the
  model-agnostic confidence signal). Everyone else does "pick one and run."
- **`DesignLoop` tooling — shipped.** `molforge.design.DesignLoop`: the
  generate → fold → (dock) → score → iterate loop, with a single-engine or
  cross-engine folder, built-in objectives (self-consistency scTM / plddt /
  affinity) plus custom callables, genuine round-to-round refinement, and a
  ranked `DesignTable`. Round-0 backbone generation (the `generator` slot,
  e.g. RFdiffusion) is designed-in but deferred.
- **Reproducibility / `pipeline.yaml` — shipped (emit *and* replay).**
  `emit_pipeline` linearizes an output's provenance chain into a citable
  `pipeline.yaml` with a consolidated environment block; `replay`
  re-executes it, resolving engines from a self-registering registry and
  reconstructing each step via per-*operation* handlers. Backed by a
  `Provenance.operation` field. Every operation a wrapper performs is now
  both recorded and replayable — `predict` (single-chain and complex),
  `predict_samples`, `predict_affinity`, `dock`, `generate`,
  `detect_pockets`, `fetch` — with a test asserting each recorded operation
  keeps a handler.
- **Plugin ecosystem — shipped.** `molforge.plugins` is documented, and
  `plugins/example_plugin/` is a complete, CI-verified reference plugin
  (registers a real `FoldingEngine`) that doubles as the copy-paste
  template. Compare napari: its plugin system is most of its value.

## 3. Workflow primitives

molforge has good *components*; these make chaining them ergonomic.

- **Parallelism primitives — shipped.** `molforge.parallel`:
  `map_parallel` plus `fold_many` / `dock_many` / `run_many` taking a list
  of inputs and a backend, with each engine declaring a `parallelism` hint
  (`"serial"` for GPU, `"process"` for CPU / subprocess). Replaced the
  `multiprocessing.Pool` loop every user was writing, and underpins
  `cross_engine_fold` and `DesignLoop`.
- **`pipeline.yaml` replay — shipped.** Re-executing an emitted manifest,
  via a `Provenance.operation` field, a self-registering engine registry
  (`molforge.plugins`), and per-operation replay handlers. See §2. The
  last two holes closed after 0.9.1, and both were about what got
  *recorded* rather than what got handled: the generative and pocket
  wrappers attached provenance with no `operation` at all, and
  `predict_affinity` recorded `operation="predict"`, so a replay returned a
  fold with no affinity on it. Worth remembering that the
  handler-coverage test passed throughout — it proves handlers exist for
  recorded operations, not that a step is replayable.
- **Provenance — shipped.** `molforge.core.Provenance`: a frozen
  dataclass (engine / operation / version / parameters / inputs / recursive
  parent), JSON-round-trippable, on `metadata[PROVENANCE]`, adopted across
  every wrapper and the prep pipelines. Engine-version introspection
  landed in 0.9.0 as `molforge.engine_versions()` (§1); sidecar
  persistence is still deferred. `Provenance.content_id()` gives the
  canonical "same computation" identity the cache and aggregate manifests
  now share.
- **Caching — shipped.** Content-addressed result cache keyed on
  `(engine, parameters, inputs, parent_chain)` with cascading
  invalidation. Folding, sequence design, docking, free-energy results,
  Chai-1 sample ensembles and `io.fetch` downloads all participate — six
  entry types. MD trajectories stay deliberately uncached (multi-GB; use
  the upstream framework's checkpointing), which is now the only gap
  rather than a follow-up.
- **A `Pipeline` / DAG builder — deferred, probably don't.** Open
  question below; the default answer is to compose with
  Prefect/Hydra/Snakemake rather than build another runtime.

## 4. Depth where it counts

Deeper, not wider — on the things users actually gate on.

- **Structure-quality validation — shipped, incl. the rollup.** Clash
  detection, Ramachandran, Cα chirality, and backbone bond-length checks,
  plus a MolProbity-style `validate.report(protein)` that combines them
  into one `QualityReport` (per-check pass/fail + an all-pass gate + a
  score) so folding/docking output can be gated on geometry in one call.
- **Binding free energy — shipped, incl. Boltz-2 affinity.** MM-PB(GB)SA
  via AmberTools and gmx_MMPBSA (with per-residue decomposition), FEP/TI
  ingestion via alchemlyb + cinnabar, and **Boltz-2 binding-affinity
  prediction** (`Boltz.predict_affinity`). **Remaining:** running (not just
  ingesting) an FEP calculation — the big §4 endeavour.
- **Unified scoring — shipped.** `molforge.scoring`: a self-describing
  `Score` with an explicit `Direction`, so docking affinity, folding
  confidence, and learned scores rank/compare uniformly, plus
  `ConfidenceScorer` / `DockingScorer` / `FunctionScorer` and `rank` /
  `best`. Plugs into `DesignLoop` as an objective — the consumer that
  motivated it.
- **Enhanced sampling — later.** PLUMED metadynamics, replica exchange,
  MELD. Heavy, but what serious MD users do.
- **Pocket detection — shipped.** fpocket (geometric) and **P2Rank**
  (ML-based, `detect_pockets_p2rank`) both return the same `Pocket` shape,
  drop-in alternatives feeding the docking workflow.

## 5. Engine matrix — *frozen*

The engine ABCs each have ≥2 real implementations. **Resist adding more
until the verification story (§1) covers what's already here** — a new
wrapper that's green-on-mocks widens the trust gap rather than closing it.
Kept here as a demand-driven wishlist, not a plan:

- **Folding:** ESM3, AlphaFold-3 (DeepMind), Protenix. On the existing
  multi-component path: modifications, restraints, and per-entity MSAs.
  (MSA *depth* and template policy landed in 0.9.0 as `msa_depth` and
  `TemplatePolicy`; these are the per-entity refinements, not the
  whole-prediction controls.)
- **Docking:** AutoDock-GPU, Uni-Dock (GPU-accelerated Vina variants).
- **MD:** NAMD, LAMMPS (non-bio workloads).
- **Generative:** LigandMPNN, Chroma, Protpardelle.
- **MSA / sequence search — `mmseqs2` shipped in 0.10.0.**
  `molforge.wrappers.msa.MMseqs2` searches a local database and returns a
  `MultipleSequenceAlignment`, which `Entity.msa` feeds to Boltz (as an
  a3m path) or Chai-1 (as an `.aligned.pqt` directory). `hmmer` /
  `jackhmmer` remain unwrapped; they emit Stockholm rather than a3m, so
  they need a reader first. molforge deliberately does not build or
  download the search databases.

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
- **Performance benchmarks page — shipped.** Real numbers from the
  benchmark suite (RMSD / DSSP / lDDT / alignment on a 200-residue input),
  with a reproduce-it command, under Architecture.
- **Migration guides — shipped.** "Coming from BioPython / Biotite /
  MDAnalysis" with concept-mapping tables, under Getting started.

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
