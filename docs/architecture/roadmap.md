# Roadmap

The goal: molforge as the layer that lets a researcher, a startup, or a
big-enterprise pipeline glue together folding, docking, MD, and
generative design across many engines without rewriting the boilerplate
each time.

## Audience and assumptions

- **Heavy deps are acceptable as opt-in extras.** Users who want
  GROMACS, AMBER, RDKit, torch, or PyTorch-Geometric pull them in
  through the appropriate extra; molforge itself stays light.
- **GPUs are assumed for "serious" work** but every feature that
  requires one carries a clear runtime warning and documentation note,
  and every CPU-feasible path has a CPU fallback.
- **Interop over reimplementation.** Where Biotite, ProDy, BioPython,
  MDAnalysis, or RDKit already solve a problem well, molforge reads
  and writes their formats rather than competing.

---

## A. Fill in the obvious gaps

### **COMPLETED** 

Small-to-medium items where the import path already exists and a real
implementation closes a visible hole.

- **Format I/O completion.** **Done** 
  - `read_sdf`, `read_mol2`, `read_pdbqt`,
    `read_pqr` are all real.
- **mmCIF write support.** **Audited and hardened.** 
  - The existing `write_cif` shipped in v0.3 was audited against every PDB fixture in
    the repo; five concrete fidelity bugs were fixed (model_id, partial
    charges, classification / deposition_date, _entry.id / block-name
    divergence, serial). 38 new round-trip regression tests; the
    parametrized `TestFixtureSweep` guards future regressions across the
    whole corpus.
- **Automatic system preparation.** **Done**
  - `remove_heterogens`, `fix_missing_atoms`, `add_caps`,
    `add_hydrogens`, and a `prepare_for_md` convenience pipeline let a
    user go from an AlphaFold-or-RCSB PDB to an MD-ready structure in
    one call. Ion neutralization remains future work — it belongs with
    explicit solvation, which is its own item.
- **Trajectory I/O.** **Done** 
  - `molforge.io.read_trajectory` / `iter_trajectory` / `write_trajectory`. 
    Wraps mdtraj for `.xtc`, `.trr`, `.dcd`, `.nc`, `.h5`, multi-MODEL PDB; 
    chunked iteration via `iter_trajectory` bounds memory for large files.

## B. Round out the engine matrix

The engine ABCs all have ≥2 real implementations now. A few more in
each modality would solidify molforge as *the* swap-engines abstraction
layer.

- **Folding.** Chai-1 (actively used, clean CLI), ESM3, Boltz-2 once
  released. AlphaFold 3 when its license permits.
- **Docking.** Gnina is **shipped**
  - (post-0.4.0; CNN-rescored Vina via
    the `gnina` binary). Smina remains unwrapped — Gnina with
    `cnn_scoring="none"` is effectively smina, so a dedicated wrapper
    is low priority. AutoDock-GPU and Uni-Dock (GPU-accelerated Vina
    variants) remain on the wishlist.
- **MD.** AMBER is **shipped**
  - (post-0.4.0; wraps `tleap` + `sander` + optional `pmemd`); 
    NAMD and LAMMPS for non-bio workloads remain on the wishlist.
- **Generative.** ESM-IF1 is **shipped**
  - (post-0.4.0; pip-installable
    inverse folding via fair-esm, companion to ProteinMPNN for
    cross-engine validation). LigandMPNN (ProteinMPNN extension that
    handles ligand context), Chroma (diffusion-based backbone
    generation), and Protpardelle remain on the wishlist.
- **MSA / sequence search.** Wrap `mmseqs2`, `hmmer`, `jackhmmer`.
  Most folding wrappers currently dodge this via ColabFold's MSA
  server; a local MSA path is what serious users need.

## C. Workflow primitives — the "pipeline" part

Right now molforge has good *components* but users have to
stitch them together.

- **A `Pipeline` / DAG builder.** Chain `fold → dock → md → score`
  declaratively, with checkpointing so a partial run resumes. Open
  design question: build our own, or just provide `@cached`
  decorators and let users compose with Prefect/Hydra/Snakemake?
  Default answer probably the latter — fight that battle only if no
  existing tool fits.
- **Provenance tracking.** **Done.** 
  - `molforge.core.Provenance` is
    the canonical "what produced this output" record — frozen
    dataclass with engine / version / parameters / inputs / recursive
    parent, JSON-round-trippable, stored on `metadata[PROVENANCE]`.
    Adoption is complete across both passes: pass 1 covered the
    simple wrappers (ESMFold, AlphaFold, Boltz, RoseTTAFold, Vina,
    DiffDock, RFdiffusion, ProteinMPNN, `load_alphafold`); pass 2
    covered the MD multi-step pipelines (OpenMM, GROMACS each
    chaining `prepare → minimize → run`) and the prep functions
    (`remove_heterogens`, `fix_missing_atoms`, `add_caps`,
    `add_hydrogens`, `prepare_for_md`). The headline-feature
    scenario from the original roadmap entry — "20 designs from
    ProteinMPNN, docked with Vina, refined with OpenMM" — is now
    fully traceable end-to-end via `result.metadata[PROVENANCE].chain()`.
    Optional polish (sidecar persistence, hash-keyed caching, deeper
    engine-version introspection) is deferred.
- **Parallelism primitives.** `dock_many`, `fold_many`, `run_many`
  taking a list of inputs and a parallelism level. Every user ends
  up writing the same `multiprocessing.Pool` loop. Tie this to the
  wrappers so each engine declares whether it parallelizes across
  processes (CPU engines) or within one process (GPU engines).
- **Caching layer.** The single biggest "real-user pain" item. Folding
  200 proteins overnight is common; if 30 crash, redoing the other
  170 is wasted compute. A content-addressed cache keyed on (engine,
  args, input hash), even just file-based, would change the user
  experience dramatically.

## D. Quality and correctness depth

Where existing functionality could be deeper, not wider.

- **Structure validation.** `molforge.validation` exists but is
  design-validator-focused. Real biology validation — Ramachandran
  checks, clash detection, chirality, bond-length sanity — would let
  users gate folding output on quality. A `validate.molprobity(protein)`
  wrapper would be high-value.
- **Unified scoring.** A `molforge.scoring` layer exposing Vina,
  Smina, AutoDock-GPU, plus learned scorers (ESM perplexity,
  ProteinMPNN confidence, Gnina's CNN affinity). Pulling scoring out
  of the docking wrappers lets users score docking results with
  multiple scorers, or score AlphaFold-folded structures with a
  scorer.
- **Active-site / pocket detection.** 
  - `fpocket` is shipped (post-0.4.0); 
  - P2Rank and SiteHound remain unwrapped. P2Rank in
    particular is the ML-based modern counterpart to fpocket and
    the natural next pocket detector to add when its install path
    gets cleaner. fpocket's adoption validated the pattern (Pocket
    dataclass + free-function detector + Provenance chaining), so
    follow-ons drop into the same shape.
- **Free energy / binding affinity.** A wrapper for `gmx_MMPBSA` (the
  GROMACS-based MM-PBSA / MM-GBSA workflow) or AmberTools'
  `MMPBSA.py` would put molforge somewhere only a handful of unified
  packages live.
- **Enhanced sampling.** PLUMED metadynamics, Replica Exchange,
  MELD-style sampling. Heavy items but what serious MD users do.

## E. ML and data layers

- **Pre-trained embedding access.** A single
  `molforge.ml.embed(protein, model="esm2-3b")` API for ESM-2 / ESM-3
  / ProtT5 / Ankh embeddings. Lazy-load weights, cache them. Lots of
  people write the same 30 lines to get ESM embeddings.
- **Inverse-folding inference.** ESM-IF1 (Meta), LigandMPNN —
  alongside ProteinMPNN which is already wrapped.
- **Structure tokenizers / discrete-structure models.** FoldSeek's 3Di
  alphabet, ESM-3's structure tokens. These let you do sequence-style
  ML on structures and are the basis of every recent paper.
- **Diffusion / flow-matching infrastructure.** RFdiffusion is wrapped.
  Chroma, FrameDiff, FoldFlow do similar things differently. A shared
  `GenerativeBackbone` interface above `GenerativeEngine` would let a
  user say "give me 20 backbones via X" without caring which X.
- **Dataset utilities.** PDB / AlphaFold-DB / CATH / SCOP loaders,
  train/val/test splits that respect sequence-identity clusters (this
  is non-trivial and people get it wrong constantly), MMseqs2-clustered
  splits, on-the-fly augmentation.

## F. Performance and infrastructure

- **Numba / Cython / Rust hotpaths.** The benchmark suite already
  identifies the candidates; rewriting the top 3-5 (DSSP, pairwise
  RMSD, contact map) in Numba would be a measurable speedup at modest
  cost.
- **GPU acceleration where it fits.** Distance maps, RMSD, contact
  maps over an ensemble are embarrassingly parallel and PyTorch-
  friendly. Optional `torch`-backed paths that activate when `torch`
  is importable and the input is large.
- **Async engine calls.** `async` versions of long-running engine
  calls let users pipeline `fold(seq1) | dock(...)` without explicit
  threading.
- **Better error taxonomy.** Each wrapper currently raises
  `RuntimeError` with subprocess stderr. A taxonomy —
  `EngineConfigError`, `ResourceError`, `ConvergenceError`,
  `OutOfMemoryError` — lets users catch the failures they actually
  want to retry on.

## G. Documentation and onboarding

- **A cookbook.** **Done**
  - Six task-oriented recipes (folding, fold-then-dock, prep-for-MD, MD +
   RMSD, design-then-refold, inspect-provenance) and three
   decision-oriented comparison tables (folding, docking,
   generative engines) live under `docs/cookbook/`. The landing
   page points to the cookbook as the answer to "trying to do
   something specific?"
- **Performance benchmarks page.** Publish the numbers from the
  benchmark suite — "molforge X takes Y ms on Z input" — so users
  know what to expect.
- **Engine comparison tables.** **Done**
  - "Which folding engine should I use
    for this case" with rows = engines, columns = (accuracy, speed,
    license, install difficulty, memory). High value, low effort, often
    missing in similar packages.
- **Migration guides.** "Coming from BioPython", "Coming from
  Biotite", "Coming from MDAnalysis". Lower the cost of switching.

## H. Distinguishing identity

These are the long-horizon items.

- **Engine-agnostic comparison.** "Fold this with ESMFold, AlphaFold,
  Boltz, and RoseTTAFold; show me the ensemble." Nobody does this
  well — it's always "pick one and run." If molforge makes cross-engine
  ensembles trivial, it has an identity nobody else owns.
- **Design loop tooling.** The protein-engineering loop (generate →
  fold → dock → score → iterate) is what real labs do, and nothing
  glues the parts cleanly. molforge has every individual piece. A
  `DesignLoop` class with sane defaults, logging, and a ranked design
  table would be unique.
- **Reproducibility.** Most papers in this space don't ship
  reproducible code. If molforge can run a workflow and emit a
  `pipeline.yaml` that fully describes what it did, including engine
  versions and weight hashes, that becomes the citable thing.
- **Plugin ecosystem.** `molforge.plugins` already exists. Documenting
  it well, providing a template repo, and writing one or two example
  third-party plugins would seed an ecosystem. Compare with `napari`:
  its plugin system is most of its value.

---

## Working order toward `1.0`

A rough sequencing:

1. ~~**Finish the format I/O stubs.** SDF, MOL2, PDBQT,
   PQR all real; DiffDock parses through `molforge.io.sdf` and Vina
   through `molforge.io.pdbqt`.~~
2. ~~**Automatic system preparation.** `molforge.prep` —
   `remove_heterogens` / `fix_missing_atoms` / `add_caps` /
   `add_hydrogens` / `prepare_for_md`.~~
3. ~~**Trajectory I/O.** `molforge.io.read_trajectory` /
   `iter_trajectory` / `write_trajectory` via mdtraj. XTC, TRR, DCD,
   NetCDF, HDF5, multi-MODEL PDB. Eager + streaming.~~
5. **Caching layer + provenance tracking.** Compounds everywhere
   downstream.
6. ~~**Cookbook + engine comparison tables.** Turn the existing surface
   into something people can find.~~
7. **Horizontal engine expansion.** Gnina, ESM-IF1, AMBER, pocket
   detection, MMPBSA. One session per engine.

Post-1.0, the identity items (cross-engine ensembles, design loop,
plugin ecosystem) become the 1.x roadmap and the long-term
differentiation work.

## Open questions

- **`Pipeline` class — build our own or compose with Prefect/Hydra?**
  Default answer: compose. Revisit only if the friction is real.
- **Plugin ecosystem strategy.** Seed it ourselves, or wait for organic
  contributors before investing? Seeding is cheaper than it looks if
  one good example exists.
- **Reproducibility format.** YAML? JSON? Something runnable? The
  ecosystem hasn't converged. Watch what Chai/Boltz/AlphaFold settle
  on.
