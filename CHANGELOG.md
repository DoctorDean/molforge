# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] 2026-06-30

### Added
- **Result caching across folding + generative engines.** The single
  largest "real-user pain" item from the roadmap, now shipped. Engines
  silently cache successful results keyed on the same `(engine,
  parameters, inputs, parent_chain)` shape that `Provenance` already
  captures. Re-running an identical computation returns the cached
  result in milliseconds; rerunning a downstream step after an
  upstream change invalidates correctly thanks to parent-chain
  participation in the cache key.

  Folding (ESMFold, Boltz, Chai-1) and sequence design (ProteinMPNN,
  ESM-IF1) all participate. A user pipeline that previously had to
  rerun a 10-minute Boltz call every time a downstream analysis
  tweak iterated is now a millisecond cache hit.

  New `molforge.cache` module:

  - `Cache` class — file-system-backed, one directory per entry
    named by SHA-256 of the cache key. Atomic writes via
    tmp-directory + rename so partial writes never leave broken
    entries. Corrupted entries (missing files, parse errors) are
    silently treated as cache misses; never crash the caller.
  - `cache_key(provenance)` — canonical hashing. Strips timestamps
    (different runs of the same computation share a slot) and
    mixes in molforge's major.minor version (upgrades invalidate
    transparently; patch versions don't).
  - `default_cache_dir()` — XDG-respecting default path resolution:
    `MOLFORGE_CACHE_DIR` → `$XDG_CACHE_HOME/molforge` →
    `~/.cache/molforge`.
  - `get_default_cache()` — process-wide singleton engines
    consult on every call.
  - `register_serializer(type_tag, serializer, deserializer)` —
    extension hook for new cacheable result types.
  - Built-in serializers for `Protein` (mmCIF + metadata JSON +
    numpy archive) and `list[DesignedSequence]` (JSON + numpy
    archive). ComplexSpec values in metadata round-trip via marker
    dicts; Provenance values round-trip via its existing
    `to_dict`/`from_dict`.

  Env-var control:

  - `MOLFORGE_CACHE=disabled` (or `0` / `false` / `off` / `no`)
    disables the cache globally. Useful for benchmarking, where
    cached returns would skew timing, or for diagnosing
    nondeterminism, where the cache would hide real differences.
  - `MOLFORGE_CACHE_DIR=/path` overrides the default location.
    Useful on shared clusters where `~/` has tight quotas.

  Engine integration (3-5 lines per engine):

  - Each wrapper gained a `_build_provenance(...)` helper —
    a pure function of inputs + constructor parameters,
    factored out of the result-attachment code path. Used both
    as the cache key (built upfront before any computation) and
    as the final result's Provenance.
  - Entry-point methods (`predict`, `predict_complex`, `generate`)
    check `cache.get(provenance, type_tag)` before invoking the
    expensive compute. On hit, return immediately. On miss, run
    the compute, attach the prebuilt Provenance to the result,
    `cache.put` it, and return.
  - `_parse_outputs` methods accept the prebuilt Provenance via
    a new keyword arg, with a fallback path that rebuilds it
    locally for direct callers (tests that exercise the parser
    in isolation).

  Tests added:

  - `tests/unit/test_cache.py` (40 tests) covers: cache-key
    determinism (timestamps stripped, molforge version
    participates), keys change on engine/params/inputs/parent
    changes, default-dir resolution with all three env-var
    interactions, Protein round-trip with numpy arrays +
    Provenance + ComplexSpec metadata, DesignedSequence list
    round-trip with per-design arrays, corruption recovery
    (missing files, wrong type tag, garbage JSON all → miss not
    crash), disabled mode (explicit constructor + all six
    documented env-var values), `clear()` only removing
    hex-named entries (never touches user files in the cache
    dir), atomicity (failed serializer leaves no partial
    entry), default-cache singleton.
  - `tests/conftest.py` gained an autouse `_isolate_cache`
    fixture: every test in the suite gets a per-test temp dir
    for the default cache, with proper singleton reset and
    env-var save/restore. This prevents test runs polluting (or
    reading from) the real user cache, and prevents tests
    leaking entries to each other within a single pytest run.

  Verification: **1,422 passed + 33 skipped** (was 1,382 + 33;
  +40 dedicated cache tests + zero regressions). mypy --strict
  clean across 93 source files (+1 for cache.py). ruff + mkdocs
  --strict clean. The existing 75 Boltz + Chai-1 + ESMFold +
  ProteinMPNN + ESM-IF1 tests pass unchanged through the cache
  integration — backward compatibility preserved.

  Cookbook:

  - New `docs/cookbook/caching-results.md` walks through the
    default behaviour, cascading invalidation, what's in the
    cache directory (with the layout sketched), three ways to
    disable, how to clear, and notes on what's deliberately not
    cached and when you might want to disable.

  Scope deliberately deferred:

  - **Docking engines** (Vina, Gnina, DiffDock). Same pattern
    applies; punted to a follow-up commit to keep this one
    focused on folding + generative.
  - **MD trajectories.** Deliberately uncached — multi-GB per
    simulation, users should rely on the upstream MD framework's
    checkpointing.
  - **LRU eviction / size caps.** v1 has no eviction; entries
    accumulate until manual cleanup. Easy to add when first user
    runs into the issue.
  - **Concurrent-write safety.** Multiple processes writing the
    same key are safe in that the final state is correct
    (last-write-wins, content is determined by the key), but no
    file locking. Sufficient for the typical single-user case.

- **Multi-component cofolding via `ComplexSpec` + `predict_complex`.**
  The headline AlphaFold-3 capability — predict structures of
  *complexes* of multiple protein chains, DNA/RNA strands, and
  small-molecule ligands in a single forward pass — is now exposed
  through a unified engine-agnostic interface. Both Boltz and Chai-1
  participate. This closes the gap between molforge's previous
  single-protein folding interface (`predict(sequence)`) and what
  these engines can actually do underneath, which was the headline
  drug-discovery and structural-biology use case for both.

  New top-level module `molforge.folding`:

  - `Entity` (frozen dataclass): one component of a complex.
    `kind` is one of `"protein"` / `"dna"` / `"rna"` / `"ligand"`.
    Polymers carry a one-letter `sequence`; ligands carry a `smiles`
    string XOR a `ccd` code (3-letter CCD codes like `"ATP"`,
    `"NAD"`, `"ZN"`). Optional `chain_id` (auto-assigned A, B, C,
    ... when omitted), `copies` for homo-oligomers, and `name` for
    human-readable identification. Upfront validation rejects bad
    input combinations (ligand with sequence, polymer without
    sequence, copies <= 0, invalid alphabet characters per kind,
    overlong CCD codes, overlong chain IDs) so user mistakes
    surface immediately rather than as opaque engine errors.

  - `ComplexSpec` (frozen dataclass): an ordered tuple of entities
    defining the system to fold. Validators reject empty entity
    tuples and duplicate explicit chain_ids. Convenience
    constructors `from_protein(sequence)` and `protein_ligand(...)`
    for the two most common shapes. `assigned_chain_ids()` returns
    the per-entity chain-ID layout that the serializer will
    produce, exposing the auto-assignment logic for users who need
    to know what chains the output structure will have.

  - `_index_to_chain_id(i)` (module-private): base-26 chain naming
    A, B, ..., Z, AA, AB, ..., handles up to 676 chains (well
    beyond any practical cofolding use case).

  New public method on `Boltz` and `Chai1`:

  - `predict_complex(spec: ComplexSpec) -> Protein` — predicts the
    full multi-component complex. Returns a `Protein` whose
    `atom_array` has one chain per entity (or per copy, for
    homo-oligomers). Ligand atoms appear as hetero-atoms with
    their assigned chain IDs.

  ```python
  from molforge.folding import ComplexSpec, Entity
  from molforge.wrappers.folding import Boltz, Chai1

  # Protein + small-molecule drug (the headline drug discovery shape).
  spec = ComplexSpec.protein_ligand(
      protein_sequence="MVTPEG...",
      ligand_smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
  )
  pred = Boltz(use_msa_server=True).predict_complex(spec)
  print(pred.metadata["iptm"])  # interface pTM (was always 0 before)

  # Antibody-antigen 3-chain complex.
  spec = ComplexSpec(entities=(
      Entity(kind="protein", sequence=heavy, chain_id="H", name="heavy"),
      Entity(kind="protein", sequence=light, chain_id="L", name="light"),
      Entity(kind="protein", sequence=antigen, chain_id="A", name="antigen"),
  ))
  abag = Chai1(use_msa_server=True).predict_complex(spec)

  # Homodimer + ATP cofactor.
  spec = ComplexSpec(entities=(
      Entity(kind="protein", sequence=enzyme, copies=2),
      Entity(kind="ligand", ccd="ATP"),
  ))

  # Protein on DNA (transcription factor).
  spec = ComplexSpec(entities=(
      Entity(kind="protein", sequence=tf),
      Entity(kind="dna", sequence="ATCGTAATCG"),
      Entity(kind="dna", sequence="CGATTACGAT"),
  ))
  ```

  The same `ComplexSpec` accepts either engine — Boltz and Chai-1
  are independent AF3 reimplementations from different teams, so
  running both and comparing predictions is a meaningful cross-
  check.

  Returned metadata (in addition to the single-sequence metadata):

  - `metadata["complex_spec"]`: the original `ComplexSpec` passed
    in, for traceability.
  - `metadata["per_chain_ptm"]`: per-chain pTM (when the engine
    produces it; Boltz exposes it as `chains_ptm` in its
    confidence JSON, Chai exposes it under `per_chain_ptm` in the
    scores NPZ).
  - `metadata["pair_chains_iptm"]` (Boltz) / `["per_chain_pair_iptm"]`
    (Chai): pairwise interface pTM when present.
  - `metadata["iptm"]`: now meaningful (was always 0 for single-
    chain inputs). For complexes, iPTM is the headline confidence
    signal — pLDDT alone can be high while interfaces are
    badly-modelled.
  - `metadata["provenance"]`: as before, with `provenance.inputs`
    carrying a JSON-safe serialization of the spec (a list of
    typed dicts with chain_ids resolved) so it round-trips through
    `Provenance.to_json()`.

  Backward compatibility: all five folding engines still expose
  `predict(sequence) -> Protein` with unchanged metadata contracts.
  For Boltz and Chai-1, `predict(sequence)` now internally
  delegates to the spec-based path with a single-entity spec —
  one canonical implementation, two entry points. The legacy
  single-sequence `metadata["source_sequence"]` key continues to
  be populated when called via `predict(sequence)`; for
  `predict_complex(spec)` calls, that key is omitted and
  `metadata["complex_spec"]` is populated instead.

  Scope deliberately deferred:

  - **Modified residues** (Boltz's `modifications` list, Chai's
    per-residue overrides). The base `Entity` sequence is treated
    as unmodified; modified-residue support will land as engine-
    specific kwargs in a follow-up commit.
  - **Restraints** (Boltz pocket constraints, Chai covalent bonds
    or restraint files). Drop down to the engine's raw API for
    now.
  - **Custom MSAs per entity** — use `use_msa_server=True` for v1.
  - **Templates** — use the engine's underlying API for v1.
  - **Boltz-2 affinity prediction** (the `properties: affinity:`
    YAML shape). Predicting binding affinity in addition to
    structure deserves its own input field and output keys.
  - **Multi-component prediction in RoseTTAFold-AA**. Has a
    different input shape; will follow with its own commit when
    needed.

  Engine-specific serializer differences:

  - **Boltz YAML** uses the `id: [A, B]` list shape for multi-copy
    entities — Boltz's documented convention for declaring
    identical chains share a single input sequence.
  - **Chai FASTA** uses separate records per copy (Chai's typed-
    FASTA format has no id-list shape). A homodimer becomes two
    `>protein|name=...` records with the same sequence and
    distinct `name=` suffixes embedding the chain_id.

  Both serializers are module-level helpers
  (`_boltz_yaml_entity(entity, chain_ids)`,
  `_chai_fasta_entity(entity, chain_ids)`) testable in isolation
  without a full engine instance.

  Tests added:

  - `tests/unit/test_folding.py` (51 tests). Entity validation
    per kind (5 protein, 4 DNA, 2 RNA, 8 ligand, 4 copies, 4
    chain_id, 1 kind), ComplexSpec validation (3), convenience
    constructors (2 from_protein, 4 protein_ligand), chain-ID
    assignment (5 spec patterns + 3 allocator), source-inspection
    invariants (2).
  - `tests/unit/wrappers/test_boltz.py` — `TestBoltzYamlFromSpec`
    (6 tests covering single-protein backcompat, smiles ligand,
    ccd ligand, DNA/RNA, homodimer id-list, mixed explicit+auto),
    `TestBoltzPredictComplex` (3 tests: end-to-end with `_invoke`
    mocked, legacy source_sequence key preserved, provenance
    records serialized spec), `TestBoltzSpecHelpers` (4 tests on
    `_boltz_yaml_entity` + `_serialize_spec_for_provenance`).
  - `tests/unit/wrappers/test_chai.py` — `TestChaiFastaFromSpec`
    (7 tests covering legacy single-protein, smiles + ccd ligand,
    DNA/RNA, homodimer-as-two-records, user name override,
    user name with multi-copy), `TestChaiPredictComplex` (4 tests
    parallel to Boltz), `TestChaiSpecHelpers` (4 tests).

  Total: 1,448 passed + 17 skipped (was 1,369 + 17 baseline;
  +79 new tests). mypy `--strict` clean across 92 source files
  (+1 for `folding.py`). ruff format + check + mkdocs `--strict`
  all clean.

  Cookbook:

  - New `cookbook/multi-component-folding.md` recipe with four
    worked examples (protein-ligand, antibody-antigen,
    protein-DNA, homo-oligomer), a Boltz-vs-Chai cross-checking
    snippet, a metadata-keys reference table, and explicit notes
    on what's deliberately not in v1.
  - `cookbook/choosing-folding.md` updated: "Predicting with
    cofactors, ligands, or nucleic acids" section now points at
    `predict_complex` and the new recipe; an example shows the
    one-line engine swap.
  - `cookbook/index.md` "If you want to..." table gains a
    multi-component cofolding row.
  - `mkdocs.yml` nav updated.

  Roadmap "Folding" entry updated: multi-component cofolding
  marked as shipped; modifications, restraints, per-entity MSAs,
  and Boltz-2 affinity prediction noted as remaining follow-ups.

- **Chai-1 folding wrapper.** Fifth folding engine, joining ESMFold,
  AlphaFold, Boltz, and RoseTTAFold. Chai-1 (Chai Discovery, October
  2024) is an open-weights re-implementation of AlphaFold-3-style
  biomolecular structure prediction — proteins, nucleic acids, and
  small-molecule ligands in a single forward pass, at AF3-class
  accuracy.

  This is the *second* AF3-style engine in molforge (the first being
  Boltz, shipped pre-v0.4). Boltz and Chai-1 are independent
  reimplementations from different teams (MIT Jameel Clinic and
  Chai Discovery) released within weeks of each other. Running both
  on a hard target and comparing predictions is a robust confidence
  signal — when two AF3-class models from independent codebases
  agree on a binding pose or interface geometry, that's much
  stronger evidence than either alone.

  Public surface:

  - `Chai1(device=, use_msa_server=, msa_server_url=, num_trunk_recycles=,
    num_diffn_timesteps=, seed=, cache_dir=)` — constructor. Argument
    validation upfront (num_trunk_recycles >= 1, num_diffn_timesteps
    >= 1). Lazy: construction touches no torch, no chai_lab, no
    weight downloads.
  - `.predict(sequence)` — same `predict(sequence) -> Protein`
    interface as the other folding engines. A one-line swap from
    `Boltz()` to `Chai1()` produces a Chai-1 prediction with the
    same downstream code.

  Mirrors :class:`Boltz`'s v1 scope deliberately: single protein
  chain only. Chai-1 natively supports multi-component complexes
  (its FASTA uses typed headers like `>protein|name=...` /
  `>ligand|name=...` / `>dna|name=...`); multi-component support
  will land for all AF3-class wrappers together in a future commit.

  Architectural difference from Boltz: Chai-1 has a clean Python
  entry point (`chai_lab.chai1.run_inference`), so the wrapper
  calls it directly — no subprocess plumbing, no YAML construction.
  Tests mock the single `_run_inference` seam to drive the full
  predict() pipeline without chai_lab installed or a GPU available.

  Output handling: Chai-1 always emits 5 diffusion samples per call
  (hard-coded upstream). The wrapper picks the highest-ranked one
  by `aggregate_score` as the canonical returned `Protein`; the
  other four samples' headline scores (`aggregate_score`, `ptm`,
  `iptm`) are preserved in `metadata["per_sample_scores"]` for
  users who want to inspect ranking spread or pick a non-best
  sample.

  Per-residue pLDDT is extracted from the chosen CIF's B-factor
  column (AlphaFold convention, which Chai-1 follows). The helper
  prefers CA-atom B-factors for canonical residues and falls back
  to per-residue mean B-factor when CAs are absent (e.g. all-ligand
  structures) rather than crashing.

  Metadata exposes the full Chai score set:

      protein.metadata["confidence_per_residue"]  # (L,) float32
      protein.metadata["mean_confidence"]         # scalar
      protein.metadata["aggregate_score"]         # best sample
      protein.metadata["ptm"]                     # best sample
      protein.metadata["iptm"]                    # best sample
      protein.metadata["best_sample_index"]       # 0..4
      protein.metadata["per_sample_scores"]       # list of 5 dicts

  Provenance: attached at the result level (`metadata[PROVENANCE]`)
  with all constructor kwargs captured for reproducibility.

  Install path: `pip install chai_lab` (much lighter than
  RoseTTAFold's clone-and-env-var path; similar to Boltz). Chai-1
  requires Linux, Python 3.10+, and a CUDA-capable GPU with
  bfloat16 support. The wrapper raises
  `FoldingEngineNotInstalledError` with both `chai_lab` install
  guidance and GPU-requirement context if the package or CUDA is
  missing.

  What this wrapper deliberately doesn't cover (v1 scope):

    - Multi-component complexes (deferred until multi-component
      lands for all AF3 wrappers together).
    - MSA file inputs (use `use_msa_server=True` for v1).
    - Restraints, templates, custom feature contexts (the lower-
      level `run_folding_on_context` API).

  Tests (`tests/unit/wrappers/test_chai.py`, 29 unit tests + 1
  real-model skipped):

  - `TestConstruction` (5) — defaults, custom options, validators
    (num_trunk_recycles, num_diffn_timesteps), construction is
    lazy.
  - `TestFastaConstruction` (2) — typed FASTA header
    (`>protein|name=...`).
  - `TestMissingDependency` (1, skipped when chai_lab installed) —
    friendly error mentions chai_lab AND the GPU requirement.
  - `TestLoadScoresNpz` (3) — flattens 0-d arrays to Python
    scalars, preserves multi-dim arrays, handles bool scalars
    correctly.
  - `TestPerResiduePlddtFromCif` (3) — extracts CA B-factors as
    per-residue pLDDT, falls back to residue-mean B-factor when
    no CAs (ligand-only structures), handles empty Protein.
  - `TestCollectSamples` (3) — gathers all 5 samples with CIF
    text + scores; missing CIF raises with the right filename;
    missing NPZ raises with the right filename.
  - `TestParseOutputs` (7) — picks best by aggregate_score, all
    metadata keys populated per the docstring contract,
    per-residue confidence from CA B-factors, per-sample scores
    preserved for all 5, Provenance captures engine config,
    empty samples list raises, missing aggregate_score samples
    sort last rather than crashing.
  - `TestPredictPipeline` (3) — full predict() with
    `_run_inference` mocked: end-to-end shape, sequence
    validation runs before model invocation, written FASTA uses
    the typed header.
  - `TestSourceInspection` (2) — regression net for the
    hard-coded 5-sample constant and the consistency of the
    "Chai-1" engine string across metadata, Provenance, and
    class name.
  - `TestRealChai1` (1, skipped without chai_lab) — full
    end-to-end with the real model. Marked `@pytest.mark.slow`
    since it requires a CUDA GPU and downloads ~3 GB of weights
    on first run.

  Cookbook updated:

  - `cookbook/choosing-folding.md` — header from "four engines"
    to "five engines"; comparison table gains a Chai-1 row;
    "Predicting with cofactors, ligands, or nucleic acids"
    section updated to position Chai-1 alongside Boltz and
    RoseTTAFold as multi-component options; new
    "Cross-checking with Chai-1 and Boltz" section shows the
    headline two-engine workflow (independent reimplementations
    of AF3, agreement is a strong signal); Installation
    footprint, Licenses, Confidence metrics, Cross-engine
    workflows sub-sections updated.
  - The "What molforge doesn't wrap (yet)" list updated to
    remove Chai-1, retaining AlphaFold-3 (DeepMind release) and
    Protenix on the wishlist.

  `pyproject.toml` mypy override added: `"chai_lab.*"` joins the
  list of optional dependencies whose missing stubs mypy should
  ignore.

- **ESM-IF1 inverse-folding wrapper.** Third generative engine,
  joining ProteinMPNN (inverse folding) and RFdiffusion (backbone
  generation). ESM-IF1 (Hsu et al. 2022) solves the same problem
  as ProteinMPNN — given a backbone, design sequences that adopt
  it — with a fundamentally different architecture (GVP-GNN +
  seq2seq transformer) and dramatically different training data
  (~12M AlphaFold2 predictions vs ProteinMPNN's ~20k PDB
  structures). Cross-checking with both engines is the headline
  workflow: orthogonal training data means residue-identity
  agreement between the two is a strong signal that's not just
  capturing training-distribution biases.

  Same `GenerativeEngine` interface as ProteinMPNN — swap from one
  to the other is a one-line change in user code, and both return
  the same `list[DesignedSequence]` shape:

      engine = ESMIF1(num_seqs=8, temperature=0.1, seed=42)
      designs = engine.generate(backbone)
      print(designs[0].sequence, designs[0].score, designs[0].recovery)

  Public surface:

  - `ESMIF1(model_name=, device=, num_seqs=, temperature=,
    score_sequences=, compute_recovery=, seed=)` — constructor.
    Argument validation upfront (num_seqs >= 1, temperature > 0).
    Lazy: construction does no model loading, no network calls,
    no torch import. Users can import ESMIF1 without fair-esm
    installed.
  - `.generate(backbone, *, chain_id="A")` — accepts a Protein,
    PDB path, or mmCIF path. Returns a list sorted best-first by
    negative log-likelihood (lower = better; matches the existing
    DesignedSequence convention from ProteinMPNN).

  Per-design metadata exposes engine config + sampling index for
  reproducibility:

      design.score                    # -ll_fullseq (lower better)
      design.recovery                 # vs native, in [0, 1]
      design.metadata["engine"]       # "ESM-IF1"
      design.metadata["model_name"]   # "esm_if1_gvp4_t16_142M_UR50"
      design.metadata["temperature"]  # passed-through sampling temp
      design.metadata["sample_index"] # 0..num_seqs-1

  All designs from one `.generate()` call share a single
  Provenance (frozen, by-reference) — same pattern as ProteinMPNN
  and Gnina. The chain extends through any upstream wrapper's
  provenance, so an RFdiffusion → ESM-IF1 pipeline produces a
  chain reading `["RFdiffusion", "ESM-IF1"]` oldest-first.

  Install path is dramatically lighter than ProteinMPNN's: ESM-IF1
  ships in the `fair-esm` PyPI package (no clone, no
  `ESMIF1_HOME` env var). `pip install "molforge[ml]"` pulls
  fair-esm; users separately install `torch-geometric` for the
  GVP-GNN layers (the upstream's environment-setup note).
  `GenerativeEngineNotInstalledError` with install-path guidance
  fires when fair-esm or its deps aren't available.

  What this wrapper deliberately doesn't cover (v1 scope):

  - **Multi-chain conditioning.** ESM-IF1 has
    `multichain_util.sample_sequence_in_complex` for designing one
    chain conditioned on others; that takes a dict input shape and
    deserves its own commit when concrete user needs surface.
  - **Partial sequence conditioning.** The model supports masking
    specific positions with `score_sequence` partial inputs;
    deferred until needed.
  - **Custom model checkpoints.** ESM-IF1 currently ships one
    production model. The `model_name` constructor argument exists
    for future-proofing.

  Multi-chain or partial-sequence use cases can fall back to the
  underlying model handle (lazily loaded; exposed via `ESMIF1.model`
  after the first `generate()` call) and call the `fair-esm`
  library directly.

  Internal seams: `_sample_one(coords)` and
  `_score_one(coords, sequence)` are deliberately small instance
  methods that bracket the calls into
  `esm.inverse_folding.util.sample_sequence` and `.score_sequence`.
  Tests patch.object these to drive the full `generate()` pipeline
  without fair-esm installed.

  Tests (`tests/unit/wrappers/test_esm_if1.py`, 25 unit tests +
  1 real-model skipped):

  - `TestConstruction` (5) — defaults, custom options, validators
    (num_seqs, temperature), construction is lazy (no torch
    needed).
  - `TestMissingDependency` (1, skipped when fair-esm installed)
    — friendly error mentions fair-esm and the molforge[ml]
    install path.
  - `TestComputeRecovery` (7) — pure-Python utility: perfect /
    partial / no match, empty strings either side, length
    mismatch uses overlap.
  - `TestSampleDesigns` (4) — drives `_sample_designs` with
    `_sample_one` and `_score_one` patched: basic sampling
    produces correctly-scored designs with right metadata,
    `score_sequences=False` skips scoring (zero score, never
    calls `_score_one`), `compute_recovery=False` yields
    `recovery=None`, empty native sequence yields `recovery=None`.
  - `TestGenerate` (4) — full `generate()` flow with
    `_ensure_loaded` / `_load_coords` / `_sample_one` /
    `_score_one` patched: designs returned sorted best-first by
    score, Provenance shared by-reference across designs,
    Provenance chains through an upstream RFdiffusion provenance,
    Provenance.parameters captures every constructor field
    relevant to reproducing the call.
  - `TestMaterialiseBackbone` (2) — Path/string passes through,
    Protein writes to a temp PDB in the supplied tmp dir.
  - `TestSourceInspection` (2) — regression net: the literal
    `ll_fullseq, _ = esm.inverse_folding.util.score_sequence`
    destructure is present (catches a future refactor that mixes
    up which of the two return values to use), the
    `score = -float(ll_fullseq)` negation is present (catches a
    refactor that flips the sign).
  - `TestRealESMIF1` (1, skipped when fair-esm missing) —
    end-to-end test that samples three sequences against the
    `ala_tripeptide_heavy.pdb` fixture with the real model.
    Runs when fair-esm is on PATH; downloads ~145 MB of weights
    on first execution.

  Cookbook updated:

  - `cookbook/choosing-generative.md` — header updated from "two
    engines" to "three engines"; comparison table gains an
    ESM-IF1 row; new "You're choosing between ProteinMPNN and
    ESM-IF1" subsection contrasts them across architecture,
    training data, sequence recovery, install footprint,
    multi-chain support, and sampling control; example
    cross-engine consensus workflow demonstrating the headline
    use case (residue-identity agreement as a robust filter);
    Installation footprint and Confidence-signals tables updated.
  - `cookbook/design-then-refold.md` — new "Cross-checking with
    ESM-IF1" section after "Pairing with RFdiffusion", showing
    the MPNN-then-ESM-IF1-validate workflow with refold-pLDDT
    triangulation.

  `pyproject.toml` mypy override added: `"esm.*"` joins the list
  of optional dependencies whose missing stubs mypy should ignore.

- **Gnina docking wrapper.** Third docking engine, joining Vina and
  DiffDock. Gnina is a fork of smina (itself a fork of AutoDock
  Vina) with integrated CNN scoring — same Monte-Carlo search as
  Vina, but each pose is rescored by a 3D convolutional neural
  network trained on PDBbind. The CNN output gives both a learned
  pose-quality score (`CNNscore`, 0–1) and a learned affinity
  estimate (`CNNaffinity`, pK units).

  Same `DockingEngine` interface as Vina and DiffDock —
  `engine.dock(receptor, ligand, *, center, box_size, ...)`
  returns a `DockingResult` with poses sorted best-first. The
  swap from Vina to Gnina is a one-line change in user code.

  Public surface:

  - `Gnina(gnina_executable=, cnn_scoring=, cnn=, sort_order=,
    scoring=, seed=, cpu=, timeout=, verbose=)` — constructor.
    `cnn_scoring` is one of `{"none", "rescore", "refinement",
    "all"}` (default `"rescore"`, matching gnina's own default);
    `sort_order` is one of `{"CNNscore", "CNNaffinity", "Energy"}`
    (default `"CNNscore"`). Argument validation is upfront —
    typos ValueError on construction rather than failing deep
    in a subprocess.
  - `.dock(receptor, ligand, *, center, box_size, exhaustiveness,
    n_poses, min_rmsd)` — same interface as `Vina.dock`. Returns
    a `DockingResult`.

  Pose-level metadata exposes all three scores per pose
  regardless of which one was used for ranking:

      pose.score                       # primary, by sort_order
      pose.metadata["vina_affinity"]   # gnina's minimizedAffinity
      pose.metadata["cnn_score"]       # CNNscore (0-1)
      pose.metadata["cnn_affinity"]    # CNNaffinity (pK)
      pose.metadata["cnn_variance"]    # ensemble variance, if present

  This lets users post-filter on a different metric than the one
  used for ranking — e.g. dock with `sort_order="CNNscore"` but
  filter the top results by `cnn_affinity > 6.0` afterward.

  Provenance: all poses from one `dock()` call share a single
  `Provenance` (frozen, by-reference) attached at the
  `DockingResult.metadata` level — same pattern as Vina and
  DiffDock. The chain extends back through any upstream wrapper's
  provenance, so an ESMFold → Gnina pipeline produces a chain
  reading `["ESMFold", "Gnina"]` oldest-first.

  The wrapper shells out to the `gnina` binary which isn't
  pip-installable; users install via `brew install gnina` on macOS,
  download a release from github.com/gnina/gnina/releases, or
  build from source. The wrapper raises
  `DockingEngineNotInstalledError` with install-path guidance when
  the binary's missing, and points at Vina as the no-CNN fallback
  so users who tried Gnina first don't think they need a CNN-
  capable install to do any docking at all.

  Tests (`tests/unit/wrappers/test_gnina.py`, 29 unit tests +
  1 binary-skipped end-to-end):

  - `TestConstruction` (7) — defaults, custom paths, all
    validators (cnn_scoring, sort_order, scoring, cpu, timeout),
    lazy resolution.
  - `TestMissingBinaryError` (1) — friendly error mentions
    install paths and the Vina fallback.
  - `TestCommandBuilder` (5) — assembled command-line has the
    right flags (--receptor, --ligand, --out, --center_x/y/z,
    --size_x/y/z, --exhaustiveness, --num_modes, --cnn_scoring,
    --pose_sort_order, --scoring), optional flags appear only
    when set (--cnn, --seed, --cpu).
  - `TestExtractRemarks` (4) — SDF tag-field parser handles the
    real gnina output format (verified against gnina/issues/294
    and the gninatorch docs), vina-only mode without CNN keys,
    empty SDF, malformed numeric values gracefully skipped.
  - `TestParseSdfOutput` (7) — end-to-end parser: two poses with
    default sort, sort_order='Energy' uses minimizedAffinity,
    sort_order='CNNaffinity', all three score types in per-pose
    metadata regardless of sort, Provenance attached at result
    level, Provenance chains through upstream, cnn_scoring='none'
    + sort_order='CNNscore' produces 0.0 scores without crashing.
  - `TestSubprocessSeam` (3) — `subprocess.run` mocked: dock()
    invokes gnina with correct binary, raises clearly on non-zero
    exit, raises clearly on missing output SDF.
  - `TestRealGnina` (1, skipped) — end-to-end against 1AKE +
    aspirin with the real binary. Skip-marked when `gnina` isn't
    on `$PATH`.

  Cookbook updated:

  - `cookbook/choosing-docking.md` — header updated from "two
    engines" to "three engines"; comparison table gains a Gnina
    row; new "You know the binding site but want better scoring
    than Vina" section explains the CNN-rescoring trade-offs and
    when to use `sort_order="CNNaffinity"` vs `"CNNscore"`;
    Installation footprint, Reproducibility, Receptor preparation
    sub-sections updated to cover all three engines.
  - `cookbook/folding-then-docking.md` — "When to pick a different
    docking engine" gains a Gnina bullet.

  Roadmap "Docking" entry updated: Gnina shipped; Smina deferred
  (effectively reachable via `Gnina(cnn_scoring="none")`);
  AutoDock-GPU and Uni-Dock remain on the wishlist.

- **AMBER MD wrapper.** Third MD engine, joining OpenMM and GROMACS.
  Wraps the AmberTools toolchain — `tleap` for topology / coordinate
  setup, `sander` for minimisation and (fallback) production
  dynamics, optional `pmemd` for production dynamics when available
  (the paid academic Amber package; the wrapper falls back to
  sander when only AmberTools is installed).

  Same `MDEngine` interface as OpenMM and GROMACS — `prepare →
  minimize → run` returning `Simulation` then `Trajectory`. The
  Provenance chain on the final Trajectory reads as
  `["AMBER.prepare", "AMBER.minimize", "AMBER.run"]` oldest-first,
  extending back through any upstream wrapper's provenance.

  Public surface:

  - `AMBER(tleap_executable=, sander_executable=, pmemd_executable=,
    water_model=, box_buffer_a=, verbose=)` — constructor.
    Resolution is lazy; construction never touches the filesystem.
  - `.prepare(protein, *, force_field="ff14SB", temperature=300.0,
    timestep=0.002)` — runs tleap to build `.prmtop` + `.inpcrd`,
    returns a `Simulation` whose `engine_handle` is the run
    directory. Supports vacuum (water_model="none"), TIP3P, TIP4P/Ew,
    OPC, SPC/E.
  - `.minimize(simulation, *, max_iterations=1000, tolerance=10.0)`
    — writes a sander min input, runs sander, reads the .rst7 back.
    ncyc (steepest-descent prelude) is chosen as half of
    max_iterations, capped at 500 — standard heuristic.
  - `.run(simulation, *, n_steps, save_every=1)` — writes a sander
    production input, runs pmemd if available else sander, parses
    the NetCDF (.nc) trajectory through molforge's existing
    `read_trajectory`.

  Curated force-field allowlist (`ff14SB`, `ff19SB`, `ff99SB`,
  `ff99SBildn`); unknown names raise ValueError upfront rather
  than failing deep in a tleap subprocess. Water-model allowlist
  similarly curated. The wrapper deliberately doesn't expose
  AMBER's full configuration surface — custom ff combinations,
  free-energy methods, REMD — those are out of v1 scope and
  extendable when concrete user needs surface.

  Tests (`tests/unit/wrappers/test_amber.py`, 20 unit tests +
  1 binary-skipped end-to-end):

  - `TestConstruction` (6) — constructor defaults, custom paths,
    vacuum mode, invalid water-model / box-buffer rejection,
    construction is lazy (no filesystem touching).
  - `TestForceFieldValidation` (3) — unknown force-field rejection,
    tleap-missing error path, error message points at OpenMM as a
    fallback.
  - `TestSubprocessSeam` (4) — drive prepare/minimize with
    `_run_subprocess` mocked, assert the generated tleap script
    contains the right directives (`source leaprc.protein.X`,
    `solvateBox`, ...), the sander min input has correct maxcyc
    and ncyc, the ncyc heuristic caps at 500 for long
    minimisations, vacuum runs skip the solvate directive.
  - `TestRunDirValidation` (4) — Simulation without an AMBER run
    directory raises clearly, pmemd/sander resolution prefers
    pmemd when available, falls back to sander when only
    AmberTools is installed, fails with a clear error when
    neither is on PATH.
  - `TestSourceInspection` (3) — regression net: the three
    Provenance step engine strings exist in the source, the
    `_parent_provenance(...)` helper is used (not raw
    `.get(...)`), `subprocess.run` is called exactly once (only
    inside `_run_subprocess`, the single seam) so future code
    that adds direct subprocess calls trips this test.
  - `TestRealAmber` (1, skipped) — full prepare→minimize→run
    pipeline against 1AKE with the real AmberTools installed.
    Skip-marked when `tleap` and `sander` aren't both on PATH;
    runs end-to-end when available, asserting the Provenance
    chain ends with the three AMBER steps.

  Pattern matches the GROMACS source-inspection regression net:
  the binary won't usually be in CI (AmberTools is conda-only,
  several GB, optional), so the wrapper's wiring is verified by
  source inspection plus mocked-subprocess pipeline tests, and
  the end-to-end check is gated by binary availability.

  Cookbook updated: the [MD and RMSD](https://doctordean.github.io/molforge/cookbook/md-and-rmsd/)
  recipe's "When to pick GROMACS instead" section is now "When to
  pick a different MD engine" and covers all three (OpenMM,
  GROMACS, AMBER) with a code snippet showing how to swap engines
  while keeping the rest of the pipeline identical.

  Roadmap MD entry updated: AMBER is now shipped; NAMD and LAMMPS
  remain on the wishlist for non-biological-MD workloads.

- **fpocket wrapper — pocket detection.** A new
  `molforge.wrappers.pockets.detect_pockets(protein)` returns a list
  of `molforge.docking.Pocket` ranked by fpocket's score (best
  first). Pockets carry a `center` (the `(3,)` geometric centre of
  the alpha-sphere cluster, ready to pass to a docking engine's
  `center=` argument), a list of lining residues, plus `volume`,
  `score`, and `druggability` headline descriptors. Each pocket's
  metadata carries the full fpocket descriptor dict for users who
  want the polar SASA / hydrophobicity / etc. extras, plus a
  `Provenance` shared across all pockets from one call (same
  immutability-by-reference pattern as ProteinMPNN's
  `DesignedSequence`).

  When the input `Protein` has its own provenance (e.g. from
  ESMFold), pocket detection chains through it, so an
  ESMFold → fpocket → Vina pipeline produces a final
  `DockingResult` whose `chain()` reads as the full
  sequence-to-pose history.

  The `Pocket` dataclass lives in `molforge.docking` alongside
  `Pose` / `DockingResult` since pockets are *for* docking — their
  canonical downstream use is feeding `center=` to a docking call.
  The detector function lives in a new
  `molforge.wrappers.pockets` subpackage, mirroring the
  `molforge.wrappers.docking` / `.folding` / `.md` / `.generative`
  layout. Free function rather than class because fpocket is
  stateless — no model to warm-load. Future ML-based detectors
  (P2Rank, PUResNet, etc.) may follow a class-based pattern for
  weight reuse; that's a per-detector decision.

  The wrapper shells out to the `fpocket` binary, which isn't
  pip-installable; users install via system package manager
  (`brew install fpocket` / `apt install fpocket`) or build from
  https://github.com/Discngine/fpocket. The wrapper raises
  `FpocketNotInstalledError` with install-path guidance when the
  binary's missing — friendlier than letting Python's raw
  `FileNotFoundError` bubble up.

  Tests (`tests/unit/wrappers/test_fpocket.py`, 19 new + 1
  binary-skipped):

  - Parser tests against real-shape fpocket output (the exact
    format documented in fpocket's GETTINGSTARTED.md):
    multi-pocket parsing, headline-descriptor extraction,
    forgiving on missing blank lines, ignoring banner / version
    lines outside pocket blocks.
  - `_maybe_float` coercion: valid floats, integers, `None`,
    non-numeric input returning `None`.
  - End-to-end parser on a synthetic output directory: pocket
    centre computed from `*_vert.pqr` alpha-sphere coords, lining
    residues extracted from `*_atm.pdb`, missing side-files
    yielding NaN centre rather than crashing, Provenance attached
    and shared across pockets from one call, parent-chaining
    through an upstream ESMFold provenance.
  - Error path: `FpocketNotInstalledError` with friendly install
    guidance when the binary's missing.
  - One real-binary smoke test (`TestRealFpocket`) skip-marked
    when fpocket isn't on `$PATH`; verifies the full pipeline
    against 1UBQ when run locally with fpocket installed.

  The pocket centre is computed from the `*_vert.pqr`
  alpha-sphere centres rather than the lining-residue centroid —
  the alpha-sphere mean is fpocket's own geometric definition of
  the pocket. The PQR I/O shipped in 0.4.0 is exactly what makes
  this composition clean (`read_pqr` does the heavy lifting; the
  wrapper just averages the coords).

  Cookbook updated: the [Fold then dock](https://doctordean.github.io/molforge/cookbook/folding-then-docking/)
  recipe now shows pocket detection as a real molforge feature
  (replacing the previous "no built-in tool yet" note), and the
  [Choosing docking engines](https://doctordean.github.io/molforge/cookbook/choosing-docking/)
  comparison adds a "detect pockets, then dock" pre-step
  workflow as an alternative to DiffDock for the
  unknown-binding-site case.

- **Cookbook + engine comparison tables.** A new
  [`docs/cookbook/`](https://doctordean.github.io/molforge/cookbook/)
  section answers the "I want to do X, what do I write?" question
  in a task-oriented way, distinct from the concept-oriented user
  guide and the learning-oriented walkthrough notebooks.

  Six recipes, each structurally complete (real imports, real
  signatures, real arguments — runnable as written if you have
  the engine's dependencies):

  - **Fold a sequence** — ESMFold workflow with confidence-based
    filtering. Covers when to switch to AlphaFold / Boltz /
    RoseTTAFold.
  - **Fold then dock** — cross-engine ESMFold → Vina pipeline,
    including how to pick a docking-site centre (active-site
    residue, co-crystallised ligand reference, geometric centre).
  - **Prepare for MD** — `prepare_for_md` end-to-end, with the
    individual `remove_heterogens` / `fix_missing_atoms` /
    `add_caps` / `add_hydrogens` steps and how to keep
    co-crystallised ligands.
  - **MD and RMSD** — OpenMM `prepare → minimize → run`, with
    `trajectory.frame(i)` and `molforge.structure.rmsd` for
    per-frame RMSD analysis. Covers trajectory I/O and
    production-scale scaling.
  - **Design then refold** — ProteinMPNN inverse folding + ESMFold
    refold validation; the standard "design score + refold RMSD +
    refold pLDDT" triple-signal check; how to pair with RFdiffusion
    for de novo design.
  - **Inspect provenance** — showcase for the
    `metadata[PROVENANCE]` feature shipped in 0.4.0. Walks the
    chain, filters by engine, saves to sidecar JSON, compares
    runs, shows the cache-key derivation pattern.

  Three decision-oriented comparison tables, separate from the
  recipes because they answer "which engine?" rather than "how do
  I run engine X":

  - **Choosing folding engines** — ESMFold vs AlphaFold vs Boltz
    vs RoseTTAFold across method, multimer support, MSA need,
    speed, install footprint, license. Includes a decision tree
    for monomer vs multimer vs cofactor predictions.
  - **Choosing docking engines** — Vina vs DiffDock, the
    force-field-search vs ML-pose-prediction trade-off, and a
    two-stage workflow that uses DiffDock for pocket discovery
    and Vina for precise scoring.
  - **Choosing generative engines** — RFdiffusion vs ProteinMPNN,
    framing them as complementary tools in the de novo design
    loop rather than competing options.

  Landing page (`docs/index.md`) gains a cookbook callout in
  "Where to go next" — first place users land. `mkdocs.yml` gets
  a Cookbook nav section between User guide and Walkthroughs.

  All 32 Python code blocks across the 10 cookbook files are
  syntax-validated (`ast.parse`); all 22 molforge imports
  resolve to real symbols (verified by import). The
  `mkdocs build --strict` pass catches any cross-link or nav
  reference that doesn't resolve.

## [0.4.0] 2026-06-25

### Added

- **Provenance adoption pass 2 — MD wrappers and prep functions.**
  Completes the wrapper-adoption work. Where
  pass 1 demonstrated cross-wrapper chaining (ESMFold → Vina),
  pass 2 exercises chaining **within** a single wrapper's
  multi-step pipeline and across the composable prep functions —
  the harder case and the one that proves the design composes.

  Adopted (2 wrappers + 5 functions):

  - **`molforge.wrappers.md.OpenMM`** — `prepare` / `minimize` /
    `run` each attach a `Provenance` to the output's metadata. Each
    step's parent is the previous step's `Provenance`, so a full
    pipeline leaves the final `Trajectory` with a 3-deep chain
    that reads as `["OpenMM.prepare", "OpenMM.minimize", "OpenMM.run"]`
    oldest-first. When the input `Protein` has its own `Provenance`
    (e.g. it came from ESMFold), the chain extends back through it
    — a sequence-to-trajectory workflow ends with a 4-deep chain
    that traces all the way to the sequence.

  - **`molforge.wrappers.md.GROMACS`** — same three-step pattern
    as OpenMM, with the engine strings `"GROMACS.prepare"`,
    `"GROMACS.minimize"`, `"GROMACS.run"`. The minimize step
    appends to `simulation.metadata` (since minimize returns the
    same Simulation in-place, not a new one), preserving the
    chain across the mutation.

  - **`molforge.prep.{remove_heterogens, fix_missing_atoms,
    add_caps, add_hydrogens, prepare_for_md}`** — each prep
    function chains a `Provenance` step onto the output's
    metadata. `prepare_for_md` (which composes the other four in
    sequence) leaves the result with a 4-deep chain naturally:
    `["molforge.prep.remove_heterogens",
      "molforge.prep.fix_missing_atoms",
      "molforge.prep.add_caps",
      "molforge.prep.add_hydrogens"]`. No special handling needed
    in the composite — the inner functions chain themselves.

  Two helpers, both private:

  - **`molforge.wrappers.md.{openmm,gromacs}._parent_provenance(meta)`**
    — extracts a `Provenance | None` from a free-form metadata
    dict, narrowing the type. Per-wrapper rather than shared so
    each MD wrapper stays self-contained.

  - **`molforge.prep._provenance.chain_prep_provenance(output, *,
    engine, parameters, input_protein)`** — the DRY-up for the
    five prep functions. One place to evolve the prep-side
    provenance shape later (e.g. when we want to also stamp the
    PDBFixer / OpenMM versions used).

  Backwards compatibility: every existing ad-hoc metadata key
  (`metadata["engine"]`, `metadata["run_dir"]`, `metadata["emtol"]`,
  etc.) is preserved unchanged. The new `metadata[PROVENANCE]` is
  additive.

  Tests (`tests/unit/wrappers/test_provenance_adoption.py`, 8 new
  in addition to the 11 from pass 1):

  - `TestOpenMMProvenanceChain` (2 tests) drives a full prepare →
    minimize → run pipeline against a real OpenMM install
    (skipped if openmm missing). The headline scenario
    `["ESMFold", "OpenMM.prepare", "OpenMM.run"]` is exercised
    end-to-end.

  - `TestGROMACSProvenanceWiring` (2 tests) — GROMACS needs the
    `gmx` binary which CI usually lacks. The tests inspect the
    module source to assert the three step engine strings appear
    and that `_parent_provenance(...)` is used — a regression net
    that catches future code that bypasses the helper without
    needing the real engine to run.

  - `TestPrepProvenanceChain` (4 tests) — exercises each prep
    function individually, then a two-function chain, then the
    full `prepare_for_md` 4-step chain, then the "ESMFold + 4
    prep steps" 5-deep chain (the most realistic scenario).
    Skipped if openmm + pdbfixer aren't both installed.

  With pass 2 complete, the headline scenario from the roadmap
  ("20 designs from ProteinMPNN, docked with Vina, refined with
  OpenMM") is now fully traceable: every output object's
  `metadata[PROVENANCE].chain()` reads as the producer pipeline,
  with sufficient detail in each step's `parameters` to
  reconstruct the exact call. The remaining provenance work is
  optional polish: persistence to a sidecar format, hash-keyed
  caching (the next roadmap item), and richer engine-version
  introspection.

- **Provenance adoption across folding, docking, and generative
  wrappers (pass 1).** Builds on the `Provenance` surface added in
  `9fafbba`. Every wrapper in scope now attaches a `Provenance` to
  the output's `metadata[PROVENANCE]` alongside its existing ad-hoc
  keys, so the "20 designs from ProteinMPNN, docked with Vina"
  scenario is now traceable end-to-end.

  Wrappers adopted:
    - **Folding**: ESMFold, AlphaFold, Boltz, RoseTTAFold. Each
      records its `__init__` config (model name, device, msa mode,
      recycles, etc.) in `parameters` and the input sequence in
      `inputs`. Folding has no upstream wrapper, so `parent` is
      `None`.
    - **Docking**: Vina, DiffDock. Each `DockingResult.metadata`
      gets a `Provenance` covering the run (box, exhaustiveness,
      seed, etc. in `parameters`; receptor + ligand refs in
      `inputs`); when the receptor was a `Protein` with its own
      `Provenance`, that becomes the `parent` so a Vina pose
      docked against an ESMFold prediction chains back to the
      sequence. Per-pose `Pose.metadata` keeps existing per-pose
      keys (`confidence`, `source_file`) — poses aren't
      independently produced, so they share the result-level
      provenance rather than each carrying their own.
    - **Generative**: RFdiffusion, ProteinMPNN. Each returned
      design (a `Protein` for RFdiffusion, a `DesignedSequence`
      for ProteinMPNN) gets its own `Provenance` — all designs
      from one call share the same Provenance object (frozen +
      immutable, so by-reference sharing is safe). `design_index`
      stays as a separate metadata key, not part of `parameters`,
      since it identifies *which* design rather than the engine
      config.
    - **`molforge.io.load_alphafold`**: the loader helper. The
      engine name reflects that this is the loader, not the
      AlphaFold run itself (`engine="load_alphafold"`); the file
      path goes in `inputs["path"]`.

  Existing ad-hoc metadata keys are preserved across every change
  — `metadata["engine"]`, `metadata["source_sequence"]`,
  `metadata["source_args"]`, etc. continue to work for 1.x
  backwards compatibility. The new `metadata[PROVENANCE]` is
  *additive*, not a replacement, until 2.x.

  Two wrapper-side surface changes worth noting:
    - `Vina._parse_poses_pdbqt` and `DiffDock._parse_outputs`
      gained optional `provenance_parameters` /
      `provenance_inputs` / `provenance_parent` kwargs (and
      DiffDock additionally `receptor_ref` / `ligand_ref`). The
      kwargs are optional so legacy tests calling the parsers
      directly without those refs still work — they just don't
      get a Provenance attached. The main `dock()` entry points
      always pass them.
    - The `Vina` module gained a `_provenance_ref` helper for
      converting a `Protein | str | PathLike` to a JSON-native
      string identifier.

  Tests (`tests/unit/wrappers/test_provenance_adoption.py`, 11
  new): one assertion per adopted wrapper plus a parent-chaining
  integration test that exercises the headline scenario
  (`ESMFold -> Vina` chain). Each adoption test holds the wrapper
  to a uniform contract: `engine` matches the documented name,
  `parameters` contains every promised key, `inputs` contains the
  expected input identifier(s). The chaining test confirms
  `result.metadata[PROVENANCE].chain()` reads as
  `["ESMFold", "Vina"]` oldest-first.

- **First-class provenance tracking: `molforge.core.Provenance`.** A
  raw PDB and a folded AlphaFold prediction look identical at the
  AtomArray level; the only difference is *how the structure was
  produced*. Pre-existing engine wrappers already wrote some of this
  information into `metadata` ad-hoc — ESMFold sets
  `metadata["engine"] = "ESMFold"`, RFdiffusion sets
  `metadata["source_args"]` — but the keys are scattered, the shapes
  disagree across engines, and there's no concept of a *parent*
  output, so a chain of operations (fold -> dock -> MD) is not
  traceable. This commit canonicalises the shape.

  The new `Provenance` dataclass (frozen, JSON-round-trippable) has:
    - **`engine`** — producer name; an engine ("ESMFold", "Vina")
      or a molforge function path ("molforge.prep.prepare_for_md").
    - **`engine_version`** — engine's own version string.
    - **`molforge_version`** — auto-filled from `molforge.__version__`.
    - **`timestamp`** — ISO-8601 UTC, auto-filled.
    - **`parameters`** — engine-specific arguments (must be
      JSON-native; validated eagerly at construction so a wrapper
      can't smuggle in a `Path` or NumPy array that crashes much
      later at serialisation).
    - **`inputs`** — identifiers of the input data (e.g.
      `{"sequence": "MKTVRQ..."}`, `{"receptor": "/path/to.pdb"}`).
    - **`parent`** — the provenance of the input this step
      *consumed*. Recursively a `Provenance` or `None`. This is what
      makes the system compositional: walking the parent chain
      reconstructs the whole history.

  Construction goes through `Provenance.from_engine(engine=...,
  parameters=..., inputs=..., parent=...)` which auto-fills the two
  version fields and the timestamp so wrappers don't have to think
  about them. The dataclass is frozen — mutating an attached
  provenance would corrupt the audit trail; use `.replace(**changes)`
  to derive an amended copy.

  Traversal helpers: `walk()` yields self then ancestors newest-first;
  `chain()` returns the same list oldest-first (suitable for printing
  as a left-to-right pipeline); `depth` is the step count.

  Serialisation: `to_dict()` / `from_dict()` give a stable plain-dict
  shape with parents nested recursively; `to_json()` / `from_json()`
  are JSON convenience wrappers. The on-disk shape is part of the
  stability commitment.

  The new `metadata_keys.PROVENANCE = "provenance"` constant is the
  documented key; `ProteinMetadata` TypedDict declares it. The
  intended use is `protein.metadata[mk.PROVENANCE] = prov`.
  **Wrappers are NOT updated in this commit** — that's deliberately
  separate work. The existing ad-hoc `metadata["engine"]` keys
  continue to work for the 1.x series; engines opt into the
  `Provenance` system gradually.

  *NOT persisted through PDB / mmCIF writers* — those preserve only
  the six documented IO header keys (per the mmCIF audit in
  `c3a012e`). Provenance is an in-memory concept; users wanting
  persistence serialise via `to_json` to a sidecar file. This is a
  documented limitation, not a bug. A future "molforge bundle"
  format could carry provenance alongside structure data; out of
  scope here.

  33 new tests in `tests/unit/core/test_provenance.py` covering
  construction (minimal, autofill, defensive copies, parent),
  strict JSON-input validation, immutability (FrozenInstanceError +
  `.replace`), traversal (walk / chain / depth), serialisation
  (dict shape, JSON round-trip, missing-engine error, forward-
  compatible deserialisation of older shapes), equality, and the
  metadata-key integration. Module coverage 96.8%; the residual
  two lines are a defensive `_molforge_version` exception fallback.

- **mmCIF writer round-trip audit and fidelity fixes.** A systematic
  audit of `write_cif_string` against every PDB fixture in the repo
  surfaced five concrete fidelity bugs in the pre-audit writer; all
  five are now fixed.
    1. **`model_id == 0` was clobbered to 1.** The old writer used
       `int(model_id) or 1`, which turned every single-model PDB's
       `model_id=0` (the `read_pdb` convention for files without
       MODEL records) into 1 on write. The reader's matching
       `or 1` default reinforced the change — every PDB → CIF
       round-trip silently lost the convention. Writer now emits
       the value verbatim; reader's default flipped from 1 to 0.
       *Affected every PDB fixture (19/19) before the fix.*
    2. **Partial / non-integer charges were truncated to int.** The
       old writer emitted `f"{int(charge):d}"`, turning typical
       PDBQT / PQR partial charges like `-0.297` into `0` and
       `-1.5` into `-1`. Writer now emits `f"{charge:.4f}"` (4
       decimal places preserves enough precision for typical
       force-field partial charges); zero still emits the `?`
       sentinel so "no charge information" round-trips cleanly.
    3. **`metadata['classification']` and `metadata['deposition_date']`
       not emitted at all.** Both PDB HEADER fields were captured
       by `read_pdb` and dropped by `write_cif`. Writer now emits
       `_struct_keywords.text` for classification and
       `_pdbx_database_status.recvd_initial_deposition_date` for
       the deposition date; reader picks them up symmetrically.
    4. **`_entry.id` and `data_<id>` block name could disagree.**
       The old writer used `protein.name` for the block name but
       `metadata[pdb_id]` for `_entry.id`. When those differed, the
       reader's `_entry.id` won and silently rewrote both fields on
       round-trip (e.g. dipeptide.pdb's `name='dipeptide'` became
       `'TEST'` because the HEADER's PDB id was `TEST`). The fix
       uses one chosen identifier — preferring `metadata[pdb_id]`,
       falling back to `protein.name`, then `"molforge"` — for both.
       Identifiers with embedded whitespace (which `read_pdb`
       tolerates from malformed HEADER lines) are now quoted in
       `_entry.id` even though the block name has to substitute
       underscores, so `pdb_id` whitespace survives round-trip.
       When no `pdb_id` exists at all, the writer emits the
       `_entry.id .` mmCIF sentinel and the reader knows to leave
       `metadata[pdb_id]` absent rather than manufacturing one from
       the block name.
    5. **`serial == 0` was clobbered to `i+1`.** Latent twin of #1;
       no fixture triggered it but the bug was there. Same fix
       pattern: only synthesize a default when `serial <= 0`.
  38 new tests in `tests/unit/io/test_mmcif.py` codify each fix as
  a regression guard. The `TestFixtureSweep` parametrized test
  iterates every PDB fixture in the repo and asserts that
  coordinates, residue/chain/atom-name fields, residue_id, model_id,
  serial, insertion_code, altloc, record_type, entity_type, and the
  six tracked metadata keys all survive a PDB → CIF → in-memory
  round-trip. Three classes — `TestEntryIdAndBlockNameConsistency`,
  `TestAltlocRoundTrip`, and `TestFixtureSweep` — explicitly
  document the two structural mmCIF limitations: (a) `Protein.name`
  is recovered from `metadata[pdb_id]` on round-trip because mmCIF
  carries only one identifier slot, and (b) altloc round-trip
  requires the caller to pass `altloc="all"` (the default strategy
  collapses to highest-occupancy and drops the label). Module
  coverage at 82.9%; the residual misses are defensive error
  branches in the tokenizer and reader.
- **Trajectory I/O: `read_trajectory`, `iter_trajectory`,
  `write_trajectory`.** `Trajectory` was previously in-memory only —
  any real MD trajectory bigger than RAM had no way into molforge.
  This commit adds binary-trajectory I/O wrapping mdtraj, exposed
  from `molforge.io`. Supported formats are everything mdtraj
  handles: `.xtc` (GROMACS lossy, the common case), `.trr` (GROMACS
  lossless), `.dcd` (CHARMM / NAMD / OpenMM), `.nc` / `.netcdf`
  (AMBER), `.h5` / `.h5md` (HDF5-based), and multi-MODEL PDB. Three
  functions: `read_trajectory(path, topology=..., stride=1,
  atom_indices=None)` loads a whole file into a
  `molforge.md.Trajectory` (use when it fits in memory);
  `iter_trajectory(path, topology=..., chunk_size=100, stride=1,
  atom_indices=None)` yields chunks of frames as `Trajectory` objects
  with bounded memory; `write_trajectory(trajectory, path)` writes
  out, format inferred from the extension. Coordinates are converted
  nm ↔ Å automatically (molforge convention is Å throughout; mdtraj's
  is nm), times in picoseconds either side. The `topology` argument
  is required for binary formats that don't embed it (XTC, TRR, DCD,
  NetCDF); it accepts either a `molforge.core.Protein` (reused
  directly — no PDB round-trip when the caller passes a Protein in)
  or a path to a PDB. PDB-format trajectories can pass
  `topology=None`. The `atom_indices` parameter slices both the
  coordinate array AND the returned topology, so callers analyzing
  a subset (e.g. backbone atoms only) get a self-consistent
  Trajectory rather than coords-and-topology mismatched. Lazy mdtraj
  import: importing `molforge.io` does not import mdtraj; only the
  trajectory functions do, and they raise
  `MDEngineNotInstalledError` with install instructions when mdtraj
  is absent. NOT wired into `load()` / `save()`: trajectories return
  a different type than the dispatcher's `Protein` / `list[Protein]`
  contract, and the topology argument is something the dispatcher
  has no way to supply. Kept as dedicated entry points. 21 new tests
  in `tests/unit/io/test_trajectory.py` covering reading from PDB,
  the Å unit conversion, Protein-topology reuse identity, path-string
  topology, no-topology PDB, atom-indices subset (including the
  metadata-preserving Protein-slice path), stride, time-array
  carry-through, the "source: mdtraj" metadata marker, the streaming
  chunk count (3+3+3+1 from 10 frames), chunk validation, the
  stride-plus-chunking compose case, the DCD / XTC / PDB write paths
  (XTC round-trip with the documented 0.001-nm precision), and the
  missing-mdtraj error path; mdtraj-using tests are
  `pytest.importorskip` gated. The new module is at high coverage
  (the few uncovered branches are unreachable error-handling
  fallbacks). `pdbfixer.*` was added to mypy's missing-stubs override
  list when `molforge.prep` shipped; no new mypy override is needed
  here since mdtraj is already there.
- **New subpackage: `molforge.prep` for MD-system preparation.** A
  raw PDB from AlphaFold, RoseTTAFold, the RCSB, or a docking engine
  almost always needs the same clean-up before MD: drop
  crystallographic clutter (waters, buffer salts, sometimes the
  ligand), rebuild missing heavy atoms, cap free termini with ACE /
  NME, and add explicit hydrogens at the right pH. The new
  `molforge.prep` subpackage exposes one composable function per
  step plus a convenience pipeline:
    - `remove_heterogens(protein)` — pure-Python residue-name filter.
      By default drops waters, ions, ligands, and everything outside
      the 20 canonical amino acids + standard nucleotides. `keep_water`
      / `keep_ions` / `keep_ligands` toggles plus an explicit `keep`
      allow-list for cofactors. Recognises multiple water aliases
      (HOH, WAT, H2O, SOL, TIP*) and the common monatomic ions.
    - `fix_missing_atoms(protein)` — wraps PDBFixer's rotamer-library
      rebuild for incomplete side chains. `fix_missing_residues=False`
      by default (de-novo loop modelling is risky);
      `replace_nonstandard=True` by default (MSE → MET, etc.).
    - `add_caps(protein)` — wraps PDBFixer to add ACE / NME caps at
      free termini of every protein chain. Multi-chain aware;
      non-protein chains (ligands, DNA) skipped. Either cap can be
      disabled with an empty string.
    - `add_hydrogens(protein, pH=7.4)` — wraps OpenMM
      `Modeller.addHydrogens` for pH-aware protonation. Idempotent on
      already-protonated input. The `force_field` kwarg accepts both
      registered aliases (`"amber14"`, `"charmm36"`) and bare XML
      filenames.
    - `prepare_for_md(protein)` — convenience entry point that chains
      the four steps with sensible defaults for an
      AlphaFold-PDB-to-OpenMM workflow. Each step's options are
      forwarded; individual steps can be turned off
      (`add_caps_to_termini=False`, `add_explicit_hydrogens=False`).

## [v0.3.0] 2026-06-22

### Added
- **`molforge.io.read_pqr` / `write_pqr` are implemented.** PQR
  (PDB2PQR / APBS) was a committed import path but a
  `NotImplementedError` stub; it now parses and writes PQR files,
  completing the format-I/O backlog (SDF, MOL2, PDBQT, PQR all real).
  PQR is a PDB-like format that appends per-atom partial charge and
  atomic radius as whitespace-separated trailing fields. Unlike PDB
  or PDBQT, PQR is **not** strictly fixed-column past the
  coordinates — different generators (PDB2PQR, AMBER, CHARMM, APBS)
  emit different widths. The reader handles all of them by parsing
  columns 1-54 as fixed (the atom record through coordinates,
  PDB-compatible) and whitespace-splitting the remainder for charge
  and radius. Charges land on `AtomArray.charge`; radii land on
  `protein.metadata["radii"]` as a per-atom list (no native
  `radius` field on `AtomArray` — electrostatics is a small enough
  slice of the surface that this didn't warrant a core-type change).
  The writer is the symmetric operation: it calls `write_pdb_string`,
  truncates each atom line at column 54, then appends
  `charge radius`. When no radii are recorded in metadata, a default
  1.5 Å is used (a reasonable middle-of-the-road heavy-atom radius
  that lets a charge-only Protein still be written as PQR). 19 new
  tests in `tests/unit/io/test_pqr.py` covering the charge/radius
  extractor (clean tokens, trailing garbage, defaults), reading from
  string and from disk, the dispatcher routes, two variant-width
  tails seen in real-world PDB2PQR / AMBER output, the full
  round-trip (coordinates, charges, radii), and the default-radius
  writer path; `pqr.py` is at 92.0% coverage.
  `test_dispatch.py`: the two stub-format tests (load and save) now
  monkeypatch a synthetic planned format rather than depending on
  any real format being unimplemented — the dispatcher's
  planned-readers fallback machinery still exists for future use, so
  the tests are still meaningful. The `_PLANNED_READERS` dict in
  `dispatch.py` is now empty.
- **`molforge.io.read_pdbqt` / `write_pdbqt` are implemented.** PDBQT
  (AutoDock / Vina) was a committed import path but a
  `NotImplementedError` stub; it now parses and writes PDBQT files.
  PDBQT is a thin extension of PDB — columns 1-66 are PDB-compatible,
  columns 71-76 hold the per-atom partial charge, and columns 78-79
  hold the AutoDock atom type (`C`, `OA`, `HD`, `NA`, ...). The
  reader reuses `molforge.io.read_pdb_string` for the heavy lifting
  (atom-array construction, altloc handling, entity classification,
  multi-MODEL parsing) and post-processes each atom line to pick up
  the extra columns: charges are written to `AtomArray.charge`,
  AutoDock types land on `protein.metadata["autodock_types"]` as a
  per-atom list. `ROOT` / `BRANCH` / `TORSDOF` rotatable-bond markers
  are read-tolerated (recognised and skipped — `AtomArray` doesn't
  carry bond topology). The writer is the symmetric operation: it
  calls `write_pdb_string`, then rewrites each `ATOM` / `HETATM` line
  to append the charge and AutoDock-type columns; when no AutoDock
  type is recorded in metadata, the element is used as a documented
  best-effort fallback. Round-tripping preserves coordinates,
  charges, and types. The Vina wrapper's pose parser is refactored to
  go through `read_pdbqt_string` rather than its previous "truncate
  every atom line to 66 columns and feed to the PDB reader" hack — so
  per-atom charges now propagate to `Pose.ligand` instead of being
  silently discarded. 23 new tests in `tests/unit/io/test_pdbqt.py`
  covering the column extractors (charge and AutoDock type, including
  the whitespace-split fallback), reading from string and from disk,
  the dispatcher routes, the full round-trip (coordinates, charges,
  types), the element-fallback writer path, multi-MODEL handling
  (Vina pose output), and `ROOT` / `BRANCH` / `TORSDOF` tolerance;
  `pdbqt.py` is at 93.8% coverage. The `test_dispatch.py` stub-format
  tests are updated to use `.pqr` (the only remaining stub format).
- **`molforge.io.read_mol2` / `write_mol2` are implemented.** MOL2
  (Tripos) was a committed import path but a `NotImplementedError`
  stub; it now parses and writes Tripos MOL2 files. Like the SDF
  reader, `read_mol2` is multi-molecule by default and returns
  `list[Protein]` (the format supports multi-molecule files via
  repeated `@<TRIPOS>MOLECULE` markers, common in docking output
  libraries). The reader populates coordinates, elements (extracted
  from the prefix of the Tripos atom type — `C.ar` → `C`, `N.am` →
  `N`, two-letter `Cl`/`Br` preserved), atom names, per-atom partial
  charges from the atom line's last column, and substructure info
  (residue id / name from the MOL2 `subst_id` / `subst_name`
  columns). Short atom lines (optional trailing columns omitted) and
  non-conforming writers that emit `***` for the subst_id or a
  non-numeric charge are tolerated with silent fallbacks rather
  than crashing the whole molecule. Bond orders, ring information,
  stereochemistry, and the `@<TRIPOS>SUBSTRUCTURE` /
  `@<TRIPOS>CRYSIN` / `@<TRIPOS>UNITY` sections are intentionally
  dropped — those need a chemistry toolkit. The writer emits a
  minimal, spec-conformant MOL2 with an empty `@<TRIPOS>BOND` section
  (some downstream tools error without the tag). The MOLECULE header
  declares the atom count; a mismatch between that and the ATOM
  section raises a clear error. `read_mol2` is wired into
  `molforge.io.load`. 34 new tests in `tests/unit/io/test_mol2.py`
  covering single/multi-molecule reading, Tripos atom-type element
  extraction, two-letter elements, partial charges, optional-column
  fallbacks, blank-line tolerance, every error path, dispatcher
  integration, and the full round-trip; mol2.py is at 94.4% coverage
  (the residual misses are unreachable defensive branches).
- **`molforge.io.read_sdf` / `write_sdf` are implemented.** SDF was a
  committed import path but a `NotImplementedError` stub; it now
  parses and writes V2000 SDF / MOL files. The reader is multi-
  molecule by default, returning a `list[Protein]` (a single-molecule
  `.mol` file still returns a one-element list, keeping the return
  type uniform across callers). The atom block, title line, and the
  ``> <Name>`` / value property block all round-trip; properties land
  on `Protein.metadata["properties"]`. The implementation uses no
  chemistry toolkit — the V2000 atom block has a fixed positional
  layout that's enough for everything molforge does downstream
  (coordinates, pose ranking, distance calculations). Bond orders,
  aromaticity, and stereochemistry are intentionally dropped; users
  who need them should call RDKit directly. V3000 files are detected
  and raise a clear error pointing at conversion paths. `read_sdf` is
  wired into `molforge.io.load`, so `load("foo.sdf")` works. The
  DiffDock wrapper, which previously parsed SDF inline, now goes
  through `molforge.io.sdf.read_sdf_string` — the inline
  `_ligand_from_sdf` helper is removed (-1 duplicated parser).
  `api-stability.md` is updated; MOL2, PDBQT, and PQR remain
  tentative. 26 new tests in `tests/unit/io/test_sdf.py` covering
  single/multi-molecule reading, the title and property block,
  round-trip writing, dispatcher integration, and every error path;
  the eight DiffDock tests that exercised the inline parser are
  removed (now subsumed by the SDF tests).
- **`GROMACS` is now a real MD engine.** `GROMACS`
  (`molforge.wrappers.md`) was a coherent stub whose `prepare` /
  `minimize` / `run` all raised `NotImplementedError`; it is now
  fully implemented. [GROMACS](https://www.gromacs.org/) is a
  command-line program (`gmx`), not a Python library, so the wrapper
  drives it as a subprocess. One `prepare` / `minimize` / `run` cycle
  maps onto the standard GROMACS workflow: `prepare` runs
  `pdb2gmx` → `editconf` → (optionally) `solvate`; `minimize` writes
  a steepest-descent `.mdp`, then `grompp` → `mdrun`; `run` writes a
  production `.mdp`, runs `grompp` → `mdrun`, then reads the frames
  back with `trjconv` and per-frame energies with `gmx energy`. All
  state for a simulation lives in one run directory whose path is
  carried on `Simulation.engine_handle` (and mirrored in
  `metadata["run_dir"]`), so `minimize` and `run` continue from
  whatever `prepare` produced. Trajectory frames are read back by
  asking GROMACS itself (`gmx trjconv`) to convert its binary `.xtc`
  to a multi-model PDB, which molforge's own PDB reader then parses —
  the wrapper deliberately takes no dependency on a third-party
  binary-trajectory library. Three small fixed-layout parsers
  (`.gro` coordinates, multi-model PDB, `.xvg` columns) handle the
  GROMACS outputs directly. `gmx` is resolved lazily via
  `shutil.which`, so construction never touches the filesystem; a
  clear `MDEngineNotInstalledError` (pointing at OpenMM) is raised
  when it is absent. A constructor flag covers the water model, box
  margin/type, and a `verbose` pass-through. 36 tests (a new
  `test_gromacs.py`), covering construction and validation, `gmx`
  resolution, the three parsers and their error paths, and the full
  `prepare` / `minimize` / `run` pipeline driven by a mocked
  `subprocess.run` that writes the files each `gmx` step would
  produce; the wrapper module is at 94% coverage (the residual
  misses are defensive `except` branches for corrupt output GROMACS
  would never actually emit). The 6 obsolete `TestGROMACSStub` tests
  are removed.
- **`DiffDock` is now a real docking engine.** `DiffDock`
  (`molforge.wrappers.docking`) was a coherent stub whose `dock()`
  raised `NotImplementedError`; it is now fully implemented.
  [DiffDock](https://github.com/gcorso/DiffDock) is a
  diffusion-generative model for *blind* protein-ligand docking — it
  needs no search box, sampling poses over the whole receptor and
  ranking them with a learned confidence model. Like the
  `RoseTTAFold` wrapper, DiffDock ships as a research repository
  rather than a pip package, so the wrapper drives it as a
  subprocess: it locates the cloned repo (`$DIFFDOCK_HOME` or an
  explicit `repo_dir`), materializes the receptor to PDB, accepts the
  ligand as a SMILES string or a path to an SDF/MOL2 file, runs
  `python -m inference`, and parses the ranked
  `rank{N}_confidence{C}.sdf` output into a `DockingResult`. DiffDock
  reports a *confidence* (higher = better), the opposite of Vina's
  affinity convention; the wrapper stores the raw value in
  `Pose.metadata["confidence"]` and sets `Pose.score` to its
  negation, so `score` ascending is best-first for every engine. SDF
  poses are parsed by reading the V2000 atom block directly (molforge's
  RDKit-backed SDF reader is still a stub, and the atom block —
  3D coordinates plus element symbols — needs no chemistry toolkit).
  A constructor flag covers `samples_per_complex`, `inference_steps`,
  and `batch_size`. 30 tests (a new `test_diffdock.py`), covering
  construction and validation, install resolution, SDF and
  confidence-from-filename parsing, and the `_run_cli` subprocess seam
  via a mocked `subprocess.run`; the wrapper module is at 100%
  coverage. The 4 obsolete `TestDiffDockStub` tests are removed.
- **OpenMM wrapper test coverage raised from 24% to 95%.** The
  OpenMM tests previously gated every real path behind
  `skipif(openmm installed)` — so `prepare` / `minimize` / `run`
  were exercised by *nothing*: when openmm was absent they couldn't
  run, and when present the negative-path tests skipped. The file is
  restructured into a dependency-free half (construction, the
  force-field registry, the missing-dependency errors) and a new
  `TestRealOpenMM` class that runs `prepare` / `minimize` / `run`
  end to end against a real OpenMM install — system building,
  hydrogen addition, the minimizer, the integration loop, trajectory
  assembly, and argument validation. A new chemically complete
  heavy-atom fixture, `tests/fixtures/pdb/ala_tripeptide_heavy.pdb`
  (ALA-ALA-ALA with all standard heavy atoms plus the C-terminal
  OXT), gives the force field something it can template. The
  real-engine tests are deliberately *not* marked `slow` — the
  tripeptide is tiny and a 20-step run is sub-second — so they run
  in the normal suite wherever openmm is installed, and skip cleanly
  (9 skips) where it isn't. A new CI job, `md-openmm`, installs the
  `[md]` extra and runs the MD wrapper tests on every push so
  `TestRealOpenMM` is actually exercised.
- **RFdiffusion wrapper test coverage raised from 84% to 99%.** The
  `_run_cli` subprocess seam of `wrappers.generative.rfdiffusion` —
  previously untested — now has direct tests via a mocked
  `subprocess.run` (no RFdiffusion or torch needed): command and
  Hydra-arg assembly, the `design_*.pdb` output-parsing path, the
  no-output `RuntimeError`, `CalledProcessError` → `RuntimeError`
  translation, the public `generate()` entry point, and
  `contigs` / `symmetry` pass-through. 5 new tests (16 → 21 in the
  file). Mirrors the ProteinMPNN coverage work from the previous
  cycle.

### Removed
- **BREAKING `molforge.wrappers.folding.Rosetta` removed.** The `Rosetta`
  name was a placeholder from the `0.0.x` series whose meaning was
  ambiguous — it could read as PyRosetta (the classical
  sequence-design library) or RoseTTAFold (the deep-learning
  model). The real wrapper now lives at `RoseTTAFold`. `Rosetta`
  had been kept this cycle as a `DeprecationWarning`-emitting alias,
  but since it never appeared in a tagged release, carrying it —
  and the day-one deprecation it implies — into the 1.0 stable
  surface added nothing. It is removed outright: import
  `RoseTTAFold` instead. A PyRosetta wrapper, if ever added, would
  be a separate class (`PyRosetta`) in its own module, since
  PyRosetta's surface is far wider than the `FoldingEngine`
  contract. The 3 alias tests are removed (folding-engine count
  unchanged: ESMFold, AlphaFold, Boltz, RoseTTAFold).

 ### Fixed
+- **`OpenMM.prepare()` now adds missing hydrogens, so heavy-atom
  structures are usable.** A force field needs explicit hydrogens,
  but `prepare()` called `ForceField.createSystem()` directly on
  whatever atoms the input had. Heavy-atom structures — the normal
  output of every folding and docking engine, and what most PDB
  files on disk contain — therefore failed with a cryptic
  OpenMM "no template found for residue" error, making the wrapper
  effectively unusable on exactly the structures molforge produces.
  `prepare()` now runs `Modeller.addHydrogens()` before building the
  system; the step is idempotent, so an already-protonated structure
  is unaffected. Because adding hydrogens changes the atom count, the
  molforge `Protein` attached to the returned `Simulation` is rebuilt
  from the protonated structure, so its topology and the coordinate
  array agree (previously a heavy-atom topology could be paired with
  a protonated coordinate array). A new `add_hydrogens` constructor
  flag (default `True`) lets callers who have pre-protonated their
  structure opt out.

## [v0.2.0] 2026-05-26 

### Added
- **ProteinMPNN wrapper test coverage raised from 69% to 96%.** The
  two previously-untested seams of `wrappers.generative.proteinmpnn`
  now have direct tests: `_parse_outputs` (FASTA-file discovery —
  single file, multi-PDB stem matching, the `.fasta` extension, and
  the no-output error path) and `_run_cli` (the subprocess-driving
  seam, exercised with a mocked `subprocess.run` so neither
  ProteinMPNN nor torch need be installed — covering command
  assembly, `chains_to_design` / `fixed_positions` / `ca_only` /
  `use_soluble_model` flag pass-through, the public `generate()`
  entry point, and `CalledProcessError` → `RuntimeError` translation).
  Edge-case tests for the `_parse_metadata` header parser (numeric vs.
  string values, tokens without `=`) were also added. 12 new tests
  (17 → 29 in the file).
- **Performance benchmark suite (`tests/benchmarks/`).** Baseline
  timings for the five structural-analysis functions most likely to
  sit in a pipeline inner loop: RMSD (with and without Kabsch
  superposition), DSSP, lDDT, distance/contact maps, and global /
  local sequence alignment. 8 benchmarks total, run against a
  synthetic 200-residue protein generated parametrically (an
  idealized alpha-helix with valid per-residue backbone geometry —
  reproducible, no large fixture files). Built on `pytest-benchmark`
  (added to the `[dev]` extra). The benchmarks are marked
  `benchmark` *and* `slow`, so a normal `pytest` run and the CI
  `test` job (`-m "not slow"`) skip them; run them explicitly with
  `pytest -m benchmark`. A new non-blocking CI `benchmark` job
  exercises them on every push so a broken benchmark is caught,
  while timing variance on shared runners never gates the build —
  for real regression tracking, save a local baseline with
  `pytest -m benchmark --benchmark-save=baseline` and compare. The
  suite skips cleanly (rather than erroring) when `pytest-benchmark`
  isn't installed.
- **`io.fetch` is now implemented.** `molforge.io.fetch` — exported
  but previously a `NotImplementedError` stub — now downloads
  structures from the RCSB Protein Data Bank
  (`source="rcsb"`, the default) or the AlphaFold Protein Structure
  Database (`source="alphafold"`), in PDB or mmCIF format. It uses
  only the standard library (`urllib`), so it adds no dependency.
  Network and HTTP-404 failures surface as a clear `OSError` with
  the failing URL; bad arguments raise `ValueError`. 7 new tests
  (argument validation + mocked-network success and failure paths).
  Surfaced by the API audit.
- **`docs/architecture/api-stability.md` — API stability reference.**
  New documentation page recording the pre-1.0 API audit: which
  parts of the public surface are committed (semver-protected) vs.
  tentative (may still change), the audit-driven changes, and the
  contract for engine-private fields. Added to the docs nav under
  Architecture.
- **`molforge.core.metadata_keys` — documented vocabulary for
  `Protein.metadata`.** `Protein.metadata` remains a free-form
  `dict[str, Any]` (no breaking change), but the keys molforge's own
  parsers and engine wrappers produce are now a documented, stable
  contract. The new module provides string constants for every such
  key (`ENGINE`, `MEAN_CONFIDENCE`, `PDB_ID`, `PAE_INTER`, ...), a
  `ProteinMetadata` TypedDict (`total=False`) re-exported from
  `molforge.core` for editor/mypy support, and a `DOCUMENTED_KEYS`
  frozenset. Keys cover three groups: structural-IO header keys (PDB
  / mmCIF parsers), uniform folding-engine keys (set by every folding
  wrapper), and engine-specific folding keys. The PDB/mmCIF parsers
  and all four folding wrappers now write metadata via these
  constants, so a key typo is an import-time `NameError` rather than
  a silently-missing key. Keys outside the documented vocabulary are
  still permitted but carry no cross-version stability guarantee. 15
  new tests, including consistency checks that the parsers only emit
  documented keys and that the TypedDict matches `DOCUMENTED_KEYS`.
- **RoseTTAFold All-Atom folding wrapper.** New file
  `src/molforge/wrappers/folding/rosettafold.py` implements a real
  wrapper around the Baker lab's RoseTTAFold-All-Atom (Krishna et
  al. 2024, *Science* 384: eadl2528). Like Boltz, RFAA is driven via
  subprocess — invocation is `python -m rf2aa.run_inference
  --config-name <name>` from inside the cloned repo with a Hydra
  config the wrapper writes to a temporary directory. Constructor
  resolves the repo via explicit `repo_dir=` or the `RFAA_HOME`
  environment variable; checks for both directory existence and an
  `rf2aa/` subdirectory before invocation. Supports custom Python
  executable (for callers whose conda env is separate from
  molforge's), the `loader_params.MAXCYCLE` override RFAA recommends
  for hard cases, custom job naming, and arbitrary Hydra-style
  overrides via `extra_overrides=`. Output post-processing parses
  the PDB (per-atom pLDDT in the B-factor column) plus the
  `*_aux.pt` PyTorch confidence file when torch is importable —
  torch tensors converted to NumPy on the way out. Surfaces the
  uniform folding-engine metadata keys
  (`confidence_per_residue`, `confidence_per_atom`,
  `mean_confidence`) plus RFAA-specific tensors (`pae`, `pde`,
  `mean_pae`, `pae_prot`, `pae_inter` — the last is RFAA's headline
  metric, <10 = high-quality interface). Degrades gracefully when
  torch isn't installed or the aux file is malformed: PDB-derived
  confidence is still populated. v1 scope is single-chain protein
  prediction matching the rest of the folding wrappers; protein-
  ligand and covalent-modification co-folding (RFAA's headline
  capability) need a separate `predict_complex()` surface and
  remain planned. 
- 47 new tests (45 passing + 2 correctly skipped:
  one for the torch tensor conversion when torch isn't installed,
  one @slow end-to-end requiring `$RFAA_HOME`). Total test count:
  830 → 875 passed + 11 skipped.
- **Boltz / Boltz-2 folding wrapper.** Real implementation replacing
  the `boltz.py` stub. Drives the `boltz predict` CLI via subprocess
  against a temporary directory and parses the resulting mmCIF +
  confidence JSON sidecar. Supports both `boltz1` and `boltz2`
  (default `boltz2`), MSA server toggling (`use_msa_server=True`
  default; pass `False` for fast single-sequence inference),
  configurable recycling steps, diffusion samples, sampling steps,
  CPU/GPU routing via `--accelerator`, custom executable path, and
  custom weights cache via `BOLTZ_CACHE`. Lazy CLI detection
  (`shutil.which("boltz")`) — construction never touches the binary;
  the first `predict()` call resolves it or raises a
  `FoldingEngineNotInstalledError` with install hints. Output
  metadata follows the uniform folding-engine convention
  (`confidence_per_residue`, `confidence_per_atom`, `mean_confidence`)
  and additionally surfaces Boltz-specific `ptm`, `iptm`, and
  `confidence_score` from the JSON sidecar. 
- 47 new tests (46 passing
  + 1 correctly skipped @slow end-to-end), structured as a series of
  testable seams: construction, sequence validation, YAML input
  construction, command-line assembly, environment setup, output
  collection, subprocess invocation (with mocked `subprocess.run`),
  and CIF post-processing in isolation. Total test count: 784 → 830
  passed + 9 skipped.
- **`molforge.ensembles` — weighted statistics over pose ensembles.**
  New top-level subpackage with seven public functions covering the
  four standard analyses run against docking output:

  - **Weighting:** `boltzmann_weights` (numerically-stable softmax
    over scores, with physical defaults — kT at 298 K, 0.593 kcal/mol
    — and a `lower_is_better` flag for ML confidence scores) and
    `resample` (weighted bootstrap of pose objects, reproducible
    with explicit `rng`).
  - **Geometry:** `pairwise_rmsd` (N×N heavy-atom RMSD matrix over
    ligand poses, vectorized in NumPy) and `pose_diversity` (summary
    statistics over the upper triangle — min/max/mean/median/std —
    for "did the docking actually explore?" diagnostics).
  - **Clustering:** `pose_clusters` (hierarchical average-linkage
    clustering at a user-specified RMSD cutoff, pure NumPy with no
    scipy dependency; returns a `PoseClusteringResult` with cluster
    labels, ordered `PoseCluster` objects with medoid index and
    intra-cluster mean RMSD, and the underlying RMSD matrix).
  - **Spatial:** `binding_site_density` (3D histogram of ligand
    heavy-atom positions, auto-sized bounding box with configurable
    padding or explicit `origin`/`shape` for comparative grids,
    Boltzmann-weightable; returns a `DensityGrid` with a
    `coordinate_of(ijk)` helper).
  - **Consensus:** `consensus_pose` (medoid pick — returns one of
    the input poses by reference — or weighted-mean synthesis —
    returns a new `Pose` with averaged coords and weighted-average
    score, marked in `metadata`).

  Designed as a top-level subpackage (not under `docking`) because
  the primitives generalize to MD trajectories and other structural
  ensembles; v1 focuses on docking poses since that's the immediately
  useful case. 1094 source lines across 5 modules, 120 new tests
  across 5 test files. Total test count: 664 → 784 passed + 8 skipped.

  Limitations documented in the module docstring and user guide:
  pose RMSD is order-sensitive (upper-bound for symmetric ligands),
  receptor is treated as fixed, and clustering is O(n³) and best
  suited for ensembles of n ≲ 200 (single-docking-run sizes; MD-scale
  ensembles would benefit from scipy's optimized linkage in a future
  enhancement).
- **Docs: ensembles user guide + API reference.** `docs/guide/ensembles.md`
  walks through the canonical workflow (score → weights → diversity →
  clusters → density → consensus); `docs/reference/ensembles.md`
  renders the full API via mkdocstrings. Added to mkdocs nav.
- **Notebook rendering via mkdocs-jupyter.** All six walkthrough
  notebooks (`notebooks/walkthroughs/01_sequences.ipynb` through
  `06_plugin_authoring.ipynb`) and all three example notebooks
  (`cross_engine_validation`, `de_novo_design`, `end_to_end_design`)
  now render as proper docs pages alongside the rest of the site.
  Notebooks live at their canonical `notebooks/` location (where CI
  executes them); they're symlinked into `docs/walkthroughs/` and
  `docs/examples/` so mkdocs-jupyter can find them inside `docs_dir`
  without duplicating files. `execute: false` in the plugin config —
  the docs build never re-runs notebooks; it renders the pre-baked
  outputs that are already committed to the repo (matching the CI
  setup that catches notebook drift separately). A new
  `docs/examples/index.md` landing page gives a 1-line summary of
  each example. Total site size now ~7.6 MB across 24 pages; the
  notebook pages average ~700 KB each (mkdocs-jupyter bundles
  notebook CSS/JS per page). Build time ~6 s.
- **Docs CI + GitHub Pages deployment.** `.github/workflows/docs.yml`
  rewritten from the placeholder `echo` into a real two-job workflow:
  `build` runs `mkdocs build --strict` on every push and PR (catches
  broken nav links, unresolved cross-references, missing
  mkdocstrings symbols), and `deploy` runs only on pushes to
  `main`/`master`, using the modern `actions/deploy-pages@v4` flow
  (no `gh-pages` orphan branch). The deploy job has `pages: write`
  and `id-token: write` permissions, is gated behind a
  `github-pages` environment, and uses a `pages` concurrency group
  with `cancel-in-progress: false` so concurrent deploys queue
  rather than thrash. CI install is just `pip install -e
  ".[docs]" ruff` — molforge's lazy-import discipline means no
  torch / scipy / openmm / biopython is needed at doc-build time,
  which keeps the docs job under a minute. Verified locally by
  building strictly in a fresh venv with only `[docs]` installed.
- **API reference pages live, strict mkdocs build green.** Eleven
  reference pages now render real API content via mkdocstrings, with
  `molforge.wrappers` split into a router landing page plus four
  per-subcategory pages (folding, docking, md, generative) — totalling
  ~1.2 MB of rendered API HTML across 15 reference pages. `mkdocs
  build --strict` is the local + CI check, with zero warnings.
- **mkdocs site skeleton (`docs/`, `mkdocs.yml`).** First end-to-end
  buildable docs site, replacing the half-finished biocore-era stub.
  Material for MkDocs theme with light/dark toggle, indigo palette,
  navigation tabs, edit-on-GitHub links, and snippets-driven content
  reuse. `mkdocstrings[python]` wired up against `src/` with
  Google-style docstring parsing, source-order member listing, and
  underscore filtering. Site is organized into five top-level
  sections — Getting started, User guide, Architecture, API
  reference, Project — and `mkdocs build` runs in ~2 s producing
  17 pages including 11 stubbed API-reference pages (one per
  subpackage, each rendering the live `__all__`). API-reference
  content fills out in the next commit; this commit lands the
  structure, theme, configuration, and all hand-written guide
  prose. Stale `docs/source/` directory removed.
- **Realistic PDB fixtures + 29 new integration tests.**
  Three new fixtures handcrafted from canonical bond lengths and
  angles (Engh & Huber 1991) to exercise real-PDB code paths that
  synthetic fixtures structurally can't:
  - **`real_small_protein.pdb`** (193 atoms, 24 residues): mixed
    helix/loop/strand topology with **all 20 standard amino acids
    plus PRO**. Every residue carries its full canonical atom set,
    so aromatic-ring parsing (PHE/TYR/TRP/HIS), branched side
    chains (LEU/ILE/VAL), and the PRO ring-closure case all get
    exercised. B-factors vary realistically (edges higher than
    core, ~16-45 Å²). The helix is left-handed due to the NeRF
    sign convention — documented honestly in a REMARK and in the
    relevant test — which doesn't affect DSSP H-bond detection or
    most other geometric analyses.
  - **`real_with_altloc_sidechains.pdb`** (97 atoms, 12 residues):
    same backbone as the first 12 residues of `real_small_protein`,
    but with **A/B alternative conformations spanning the full
    side chain** at LEU 2 (CB/CG/CD1/CD2) and SER 9 (CB/OG).
    Occupancies 0.60/0.40. Replaces the prior 8-atom
    `with_altloc.pdb` for any test that needs multi-atom altloc
    context (the old one is kept for parser-level smoke tests).
  - **`real_with_ligand_realistic.pdb`** (76 atoms, 13 residues
    across 3 chains): 8-residue helix + a **benzene ligand** (BNZ,
    6 aromatic carbons in a proper hexagonal ring at canonical 1.4 Å
    spacing) + **a zinc ion** (ZN, properly classified as ion not
    ligand thanks to the corrected element column) + **3
    crystallographic waters** (HOH, classified as water). Replaces
    `mini_with_ligand.pdb`'s fake imidazole + waters with proper
    multi-chain hetero-atom chemistry.
- 29 integration tests in `tests/integration/test_real_fixtures.py`
  organized by code path (fixture loading, entity-type classification,
  full side-chain atom counts, multi-atom alt-loc handling under all
  four `altloc=` modes, write-then-read round-trip preservation,
  structural-analysis algorithms exercised on the realistic protein,
  ML featurization on full side chains, and sequence mutation through
  `mutate_protein`). Each test class is named after the code path
  it exercises rather than the fixture, so failures point at the
  broken behavior rather than the test fixture.
- Test coverage now stands at **664 passing + 8 correctly skipped**
  (+29 from the new fixtures), with the integration suite growing
  from 19 to 48 tests.
- **[`notebooks/examples/cross_engine_validation.ipynb`](notebooks/examples/cross_engine_validation.ipynb)**:
  20-cell worked example of the cross-validator consensus pattern.
  Uses two deterministic synthetic validators (mimicking
  ESMFold-like and AlphaFold-like output) to walk through:
  single-validator `cross_validate`, the strict / permissive /
  majority consensus modes, drilling into a borderline design to
  see which validator disagreed, and ranking the survivors. End-
  to-end executable without GPU; the validator stubs are designed
  so the cross-architecture-disagreement pattern (one model
  overconfident, one model rejecting) is clearly visible on a
  single sample design.
- Both this new notebook and the
  [`05_ml_featurization`](notebooks/walkthroughs/05_ml_featurization.ipynb)
  one are now in the CI's executable allowlist (so any drift
  between the notebook outputs and library behavior breaks CI).
- **CI now executes runnable notebooks.** A new `notebooks` job in
  `.github/workflows/ci.yml` parse-validates every notebook in
  `notebooks/` and executes the four that don't require external
  engines (`01_sequences`, `02_structures`,
  `05_ml_featurization`, `06_plugin_authoring`) top-to-bottom
  against the freshly-installed library. Catches the class of bug
  where a notebook's outputs go silently out of sync with the
  library — if any cell raises, CI fails.
- **`scripts/execute_notebooks.py`**: the underlying executor.
  Usable locally as `python scripts/execute_notebooks.py` (or
  `--check-only` for parse-only). Maintains explicit allowlists
  for executable vs. parse-only notebooks; updating either list is
  the only thing required when adding a new notebook.
- `nbclient>=0.10` and `ipykernel>=6.29` added to the `[dev]`
  extra so the script is runnable in any dev environment via
  `pip install -e ".[dev]"`.
- **`molforge.plugins.discover()` implemented.** Was previously
  raising `NotImplementedError`; now walks Python entry points
  under the `molforge.plugins` group via
  `importlib.metadata.entry_points`. Each entry point's registration
  function is called once. Broken plugins (failed import, register
  function raises) are tolerated and silently skipped so one bad
  plugin can't break every downstream user of molforge. Returns the
  list of successfully-loaded entry-point names so callers can
  introspect what's available. Companion `clear()` exported for
  test isolation.
- **[`notebooks/walkthroughs/06_plugin_authoring.ipynb`](notebooks/walkthroughs/06_plugin_authoring.ipynb)**:
  the last walkthrough stub from the v0.0.1 skeleton is now live.
  14-cell tour of the plugin registry: when to use it vs. direct
  imports, how to register engines / parsers / scorers, the
  inline-vs-entry-point distinction, and how the
  `pyproject.toml` entry-point declaration translates into
  auto-discovery. Includes a runnable `RandomFolder` toy engine,
  a minimal `.xyz` parser, and a `hydrophobic_fraction` scorer
  registered inline so the notebook executes end-to-end without
  installing anything extra.
- 11 new plugin-registry tests bringing total registry coverage
  to 12: basic register / available / get round-trip, all three
  kinds (engine / parser / scorer), `clear()` isolation, and
  `discover()` against a mocked `importlib.metadata.entry_points`
  covering the multi-plugin case, the broken-plugin tolerance, and
  the empty-entry-points fallthrough.

### Fixed
- **`load_alphafold` now emits the uniform confidence metadata keys.**
  `molforge.io.load_alphafold` previously wrote only AlphaFold-specific
  keys (`plddt`, `plddt_per_residue`, `mean_plddt`, `source`), while
  the AlphaFold *wrapper* wrote the cross-engine-uniform keys
  (`confidence_per_atom`, `confidence_per_residue`, `mean_confidence`,
  `engine`). Downstream code reading confidence uniformly across
  engines silently missed AlphaFold structures loaded from disk.
  `load_alphafold` now populates both sets (uniform keys preferred,
  legacy keys retained for backward compatibility); the two carry
  identical values. Surfaced by the API audit.
- **`GROMACS` and `DiffDock` are now coherent stubs.** Both are
  exported (committed import paths) but unimplemented. Previously
  they were *incoherent*: `GROMACS` didn't implement its `MDEngine`
  abstract methods at all, so `GROMACS()` failed with a cryptic
  "Can't instantiate abstract class" `TypeError` rather than a
  meaningful message; both engines' methods raised a bare
  `NotImplementedError` with no text. They are now coherent stubs —
  instantiable, satisfying their respective engine ABCs
  (`MDEngine` / `DockingEngine`), with every method raising
  `NotImplementedError` carrying a clear message that points at the
  working alternative (`OpenMM` / `Vina`) and the tracking issue.
  10 new tests. Surfaced by the API audit.
- - **Lint drift from a Ruff version bump cleared; CI lint job green
  again.** `.pre-commit-config.yaml` pinned `ruff-pre-commit` at
  `v0.5.0`, but the `[dev]` extra installs `ruff>=0.5` unpinned, so
  CI resolved a much newer Ruff (0.15.x) whose added rules flagged
  33 pre-existing issues — meaning the CI `lint` job was effectively
  red. All 33 are now resolved: a genuine dead variable in
  `ensembles.clustering` removed, an unused `shutil` import dropped,
  five `pytest.raises(match=...)` patterns with unescaped regex
  metacharacters made explicit (raw strings / escaped dots), a
  `zip()` given an explicit `strict=`, four nested `with` statements
  collapsed, a `getattr()` call with a string literal in
  `ensembles.weighting` replaced by a `cast`-backed direct attribute
  access (dropping a now-misplaced `# noqa`), and a Ruff-version
  formatting refresh applied across 24 files (cosmetic line-joining
  only). Two intentional-notation cases
  are configured rather than rewritten: `allowed-confusables`
  permits `×`, `σ`, and `–` in docstrings (matrix dimensions, the
  standard deviation, prose dashes), and `RUF022` is per-file-ignored
  for the two modules whose `__all__` is deliberately grouped by
  category with section comments. The `ruff` and `mypy` pre-commit
  pins are bumped to the versions CI resolves, so the two stay in
  lock-step and this drift cannot silently recur. No source-behaviour
  or test-count change (918 pass + 11 skipped, unchanged).
- **Docs notebooks no longer use symlinks.** The walkthrough and
  example notebooks were previously symlinked from `docs/` into the
  canonical `notebooks/` directory. Symlinks broke two things: (1)
  extracting a release tarball on Windows failed with "a required
  privilege is not held by the client" because creating symlinks
  needs a privilege normal accounts lack, and (2) the GitHub Pages
  docs build failed in strict mode because `actions/checkout`
  didn't preserve the links, leaving nine dangling `nav` references
  (13 strict-mode warnings, non-zero exit). Replaced with a build
  hook (`docs/_hooks/copy_notebooks.py`, registered via the
  `hooks:` key in `mkdocs.yml`) that copies the notebooks from
  `notebooks/` into `docs/` during `on_config`, before mkdocs's
  file discovery runs so mkdocs-jupyter renders them normally. The
  copies are git-ignored; the notebooks remain single-source in
  `notebooks/`. No symlinks anywhere in the repo, and the tarball
  extracts cleanly on every platform.
- **`docs/guide/data-model.md` field names.** Two rows in the
  `AtomArray` schema table used pre-rename names (`res_name`,
  `res_id`); corrected to `residue_name`, `residue_id` to match
  the actual public attributes. Discovered while writing ensemble
  test fixtures.

### Changed
- **BREAKING: `cross_validate` now defaults to `on_error="raise"`.**
  Previously `cross_validate` defaulted to `on_error="record"` —
  exceptions raised by individual validators were silently caught,
  recorded in verdict metadata, and the verdict marked
  `passed=False`. The problem: a validator that throws on every
  design (a misconfigured engine, a missing dependency, a bad
  input) produced a full list of `passed=False` verdicts that
  *looked* like a real result, hiding the bug. The new default
  fails loud. Code that genuinely wants a batch to survive
  individual validator failures must now pass `on_error="record"`
  explicitly. Flagged and resolved by the API audit.
- **The entire `molforge` package is now `mypy --strict` clean.**
  With `wrappers` and `plugins` brought up to strict, all 77 source
  modules across every subpackage pass `mypy --strict` with zero
  errors. The CI `typecheck` job is correspondingly simplified: the
  previous two-step arrangement (a strict gate on the clean
  subpackages plus a non-blocking informational full-tree run)
  collapses to a single `mypy src` gate that fails the build on any
  type error. The `tests/unit/test_typing.py` regression test is
  likewise simplified to one whole-package check. 31 errors fixed in
  this final tranche: 20 stale `# type: ignore` comments (made
  redundant when the optional heavy dependencies were added to the
  mypy `ignore_missing_imports` override), four deliberate engine-
  method `# type: ignore[override]` annotations (the concrete
  engine wrappers refine the permissive `**kwargs` signatures of
  their `DockingEngine` / `MDEngine` / `GenerativeEngine` abstract
  bases — an intentional, documented refinement that mypy's strict
  Liskov check cannot model), `cast`s for the opaque
  `Simulation.engine_handle` inside the OpenMM wrapper and for the
  unstubbed-dependency return values, and `Vina.dock`'s receptor
  narrowing switched from `hasattr` to `isinstance` (a more correct
  check that mypy can also narrow on).
- **`molforge.ml` is now `mypy --strict` clean.** The ML subpackage
  (sequence/structure featurization, protein-language-model
  embeddings) joins the strict gate — eight strict-clean
  subpackages in total, 51 source files. Six errors fixed: the four
  numpy-widening `no-any-return`s in `embeddings.py` (resolved with
  `cast`s), and two real type bugs in `structure_features.py` —
  `pair_distances` and `pair_distance_features` declared
  `atom_choice: str` but pass it to `distance_map`, which requires
  the `Literal["ca","cb","heavy","all"]` the docstrings already
  specify, and a coordinate feature array silently upcast to
  float64 by a division. The `torch` and `transformers` (and
  `colabfold`, `meeko`, `vina`) optional heavy dependencies, which
  ship no type stubs, are added to the mypy `ignore_missing_imports`
  override alongside the existing `Bio` / `biotite` / `mdtraj` /
  `openmm` / `rdkit` entries. CI strict gate and the
  `tests/unit/test_typing.py` regression test updated; only
  `plugins` and `wrappers` remain outside the gate.
- **Six more subpackages are now `mypy --strict` clean.**
  `molforge.io`, `molforge.sequence`, `molforge.structure`,
  `molforge.metrics`, `molforge.ensembles`, and
  `molforge.validation` now pass `mypy --strict` with zero errors,
  joining `molforge.core` — seven strict-clean subpackages in total,
  46 source files. The 12 errors fixed were mostly numpy operations
  mypy widens to `Any` (resolved with explicit `cast`s that document
  the known array dtype) and two stale `type: ignore` comments; two
  were genuine annotation bugs — `_place_hydrogens` in `dssp.py` was
  declared to return a single array but actually returns a
  `(coords, mask)` tuple, and `_score` in `alignment.py` was
  declared `NDArray[np.int_]` but builds an `int32` array (`np.int_`
  is `int64` on 64-bit platforms). The CI strict gate now covers all
  seven subpackages; the regression test
  (`tests/unit/test_typing.py`, moved up from `tests/unit/core/` and
  parametrized) checks each one in-suite. The remaining subpackages
  (`ml`, `plugins`, `wrappers`) are still tracked by the
  non-blocking informational `mypy src` CI step.
- **`molforge.core` is now `mypy --strict` clean, and CI enforces
  it.** The `core` subpackage — the data model the rest of the
  library is built on — now passes `mypy --strict` with zero
  errors (fixed: two missing `NDArray` type arguments in
  `AtomArray`, an `Any`-return in `Atom.coord`, and an untyped
  `Chain.__iter__` that was suppressed with a `type: ignore`). The
  CI `typecheck` job now runs `mypy --strict src/molforge/core/`
  as a hard gate, with a separate non-blocking full-tree `mypy src`
  step that keeps the remaining (out-of-`core`) type errors visible
  while they're worked through. A new `slow`-marked regression test
  (`tests/unit/core/test_typing.py`) runs the strict check in-suite
  so a `core` type regression is caught locally too.

### Documented
- **`Simulation.engine_handle` contract clarified.** The attribute
  type (`object | None`) is correct — it really is an opaque,
  engine-specific handle — but the contract was under-specified.
  The docstring now states explicitly that `engine_handle` is
  engine-private (callers must not inspect it or set it), is **not
  serialized** (it typically wraps unpicklable C-extension state;
  persistence layers must drop it and let the engine wrapper
  rebuild it on resume), and carries **no semver guarantee**. For
  inspectable per-simulation data, `Simulation.metadata` is the
  supported field. No code change. Flagged by the API audit.

### Deprecated
- **`molforge.wrappers.folding.Rosetta` is now a deprecated alias
  for `RoseTTAFold`.** The original `rosetta.py` placeholder was
  ambiguous about whether it referred to PyRosetta (the Baker lab's
  classical sequence-design library) or RoseTTAFold (the deep-
  learning model). The new real wrapper lives at
  `RoseTTAFold` for clarity. `Rosetta` is retained as a thin
  subclass that emits `DeprecationWarning` on construction so
  existing imports / isinstance checks keep working through the
  next minor release. A PyRosetta wrapper, if added, would live in
  a separate module (`pyrosetta.py`) since PyRosetta's surface is
  much wider than the `FoldingEngine` contract.

### Removed
- **`tests/unit/core/test_core_types.py`.** A pre-existing fossil
  from before the view-based data-model refactor: it imported from
  `biocore.core` (the pre-rename namespace) and called constructors
  like `Chain(chain_id="A")` and `Residue(name="ALA", seq_id=1)`
  that no longer match the current view-based signatures (`Chain`,
  `Residue`, and `Atom` are now views over an `AtomArray` and take
  `(array, start, end)` not standalone keyword arguments). The
  assertions it made were already fully covered by
  `test_hierarchy.py` (260 lines, with dedicated `TestAtom`,
  `TestResidue`, `TestChain`, `TestProtein`, and `TestConsistency`
  classes) and `test_atom_array.py` (210 lines). Removing the
  fossil unblocks the full test suite from running cleanly under
  `pytest` (previously needed `--ignore` for that one file).
  Headline test count unchanged: 664 + 8 skipped.

## [v0.1.0] 2026-05-20 

- **`molforge.validation`: cross-validation utilities for protein
  design.** Captures the common pattern of "score designs across
  multiple validators and combine results" that was previously
  hand-rolled as list comprehensions in user code.
  - **`Criterion`**: declarative success conditions (e.g.
    `Criterion.gt("plddt", 80)`). Atomic comparisons via the six
    standard operators (`gt` / `ge` / `lt` / `le` / `eq` / `ne`),
    composable with the standard logical operators (`&` for AND,
    `|` for OR, `~` for NOT) to express arbitrarily complex success
    rules. Tracks which metrics it references via
    `criterion.metric_names`, useful for upstream validation.
  - **`CriteriaSet`**: a named collection of criteria evaluated
    together, returning per-criterion pass/fail for diagnostics
    rather than just an opaque boolean. Implicit AND across criteria.
  - **`Verdict`**: per-design result combining metric values,
    per-criterion results, an overall pass/fail, and a sortable
    score. Exposes `failed_criteria` / `passed_criteria` properties
    for inspection.
  - **`cross_validate(designs, validators, criteria)`**: the
    workhorse. Runs every design through every validator, namespaces
    metrics by validator name (`"esmfold.plddt"` not just `"plddt"`),
    applies criteria, returns ranked verdicts. Handles validator
    exceptions gracefully (default: record in metadata + mark
    failed; opt-in propagation via `on_error="raise"`).
  - **`consensus(verdict_lists, mode=...)`**: merges verdict lists
    across validators under a chosen rule (`"all"` / `"any"` /
    `"majority"` / explicit `"threshold"` count). Joins by
    `design_id`; metric values from every validator are preserved
    in the merged `Verdict.values`; per-criterion pass/fail is
    namespaced by validator to keep diagnostics distinguishable.
  - **`rank_verdicts(verdicts, only_passed=..., by=...)`**: ranking
    helper. Defaults to ascending score (lower-is-better); can
    filter to only-passed; can sort by an arbitrary metric name
    instead of the score.

### Changed
- **Docstring normalization for griffe.** Twelve docstrings across
  `metrics/{dockq,gdt,lddt}`, `ml/{graph,structure_features}`,
  `sequence/alignment`, `structure/{contacts,dihedrals,rmsd,sasa}`,
  and `wrappers/folding/{alphafold,esmfold}` rewritten so each
  parameter gets its own line in the `Args:` block (instead of
  comma-grouping like `a, b: ...`), and continuation lines under
  bullet points are re-indented to 8 spaces. No behavior or
  signature changes; griffe was the only consumer mis-parsing them,
  but the fix also makes the rendered tables clearer (each
  parameter gets its own row). 664 + 8 skipped tests, unchanged.

### Removed
- **`tests/unit/core/test_core_types.py`.** A pre-existing fossil
  from before the view-based data-model refactor: it imported from
  `biocore.core` (the pre-rename namespace) and called constructors
  like `Chain(chain_id="A")` and `Residue(name="ALA", seq_id=1)`
  that no longer match the current view-based signatures (`Chain`,
  `Residue`, and `Atom` are now views over an `AtomArray` and take
  `(array, start, end)` not standalone keyword arguments). The
  assertions it made were already fully covered by
  `test_hierarchy.py` (260 lines, with dedicated `TestAtom`,
  `TestResidue`, `TestChain`, `TestProtein`, and `TestConsistency`
  classes) and `test_atom_array.py` (210 lines). Removing the
  fossil unblocks the full test suite from running cleanly under
  `pytest` (previously needed `--ignore` for that one file).
  Headline test count unchanged: 664 + 8 skipped.

## [v0.0.3] 2026-05-20 

- **De novo design notebook updated**: the `de_novo_design.ipynb`
  example now includes a section demonstrating the validation
  utilities — declarative `CriteriaSet`, `cross_validate` against a
  reproducible stub validator, ranking, and `consensus` semantics
  for the multi-validator case. The old hand-rolled filter
  expressions in section 4 stay as a baseline for contrast.
- 69 unit tests covering: all six atomic operators with edge cases
  (strict-vs-loose comparison, None values, missing metrics), the
  three logical compositions (AND / OR / NOT) including complex
  multi-level expressions, `metric_names` aggregation, repr formats,
  `NamedCriterion`, `CriteriaSet` evaluation and chaining, `Verdict`
  field defaults and properties, `rank_verdicts` with score / metric
  / passed-only sorting and tie-stability, `cross_validate` with
  single and multiple validators, namespacing, score-from-metric vs.
  score-from-failed-count, error handling (record / raise modes,
  partial validator failure still marks verdict failed), custom
  `design_id` functions, and `consensus` across all four modes plus
  the ID-mismatch and threshold-validation error paths.
- **[`notebooks/examples/de_novo_design.ipynb`](notebooks/examples/de_novo_design.ipynb)**:
  showcase notebook for the full *de novo* design loop. Walks
  RFdiffusion backbone generation → ProteinMPNN sequence design →
  ESMFold validation → scoring with `molforge.metrics`
  (TM-score, lDDT, RMSD) → filtering by the standard
  "successful design" criterion (pLDDT > 80, TM > 0.5, RMSD < 2 Å).
  Heavy cells are marked and unexecuted (both engines require
  GPU + multi-GB weights); the API calls and expected outputs are
  documented inline with representative numbers from real runs.
  Demonstrates the four main RFdiffusion modes (unconditional /
  motif scaffolding / binder design / symmetric) and the main
  ProteinMPNN options (fixed positions, multi-chain design,
  sampling control).
- Updated `notebooks/README.md` (added the de novo notebook to the
  examples table) and `README.md` (top-level callout block now
  surfaces it first).
- **`molforge.wrappers.generative`: protein design wrappers.**
  - **`RFdiffusion`** — backbone generation via the RoseTTAFold
    diffusion model (Watson et al. 2023, *Nature* 620:1089-1100).
    Supports unconditional monomer generation (just specify a
    length), motif scaffolding (preserve specific residues from an
    input PDB), binder design (with hotspot residues on the
    target), and symmetric design (cyclic / dihedral / tetrahedral).
    The wrapper translates Python kwargs into RFdiffusion's Hydra
    `key=value` syntax internally so users don't need to learn the
    Hydra config dialect. Invokes the official `scripts/
    run_inference.py` via subprocess for reliability.
  - **`ProteinMPNN`** — sequence design (inverse folding) via the
    message-passing neural network from Dauparas et al. 2022
    (*Science* 378:49-56). Given a backbone `Protein`, samples
    sequences that should fold to it. Supports monomer and multi-
    chain design, fixed-position constraints (preserve a known
    motif), all four official model variants (`v_48_002` /
    `v_48_010` / `v_48_020` / `v_48_030`), soluble-only and
    CA-only checkpoints, configurable sampling temperature and
    omitted amino acids. Writes the helper-script JSONL formats
    internally rather than shelling out to the helper scripts,
    keeping the wrapper self-contained.
  - **Shared infrastructure**: `molforge.generative.GenerativeEngine`
    abstract base; `DesignedSequence` dataclass (sequence + score
    + optional recovery + metadata) returned by sequence-design
    engines; `GenerativeEngineNotInstalledError` mirroring the
    folding / docking / MD error conventions.
  - **Together with the existing wrappers, this completes the
    *de novo* protein design loop in molforge**: RFdiffusion to
    generate a backbone, ProteinMPNN to design sequences for it,
    ESMFold / AlphaFold to validate the designs fold back to the
    input shape, OpenMM to refine, Vina to dock against a target,
    and `molforge.metrics` to score everything.
- 33 unit tests covering: construction with parameter validation
  (model name allowlist, temperature range, num_seqs minimum),
  install-path resolution (env-var fallback, explicit-arg
  override, sanity-check for missing scripts), missing-dependency
  error paths, Hydra-arg building for all five RFdiffusion modes
  (unconditional / motif / binder / symmetric / extra args),
  FASTA output parsing in isolation (sample ProteinMPNN headers
  with the score/T/sample/model_name/git_hash convention,
  sort-by-score, native-record skip), fixed-positions JSONL
  writing, and the `DesignedSequence` repr.
- **[`notebooks/walkthroughs/03_md_simulations.ipynb`](notebooks/walkthroughs/03_md_simulations.ipynb)**:
  the last walkthrough stub from the v0.0.1 skeleton is now a real
  22-cell tour of the OpenMM MD wrapper. Covers the
  `prepare → minimize → run` flow with the bundled force-field
  registry, what the `Simulation` and `Trajectory` dataclasses
  look like in practice, and how to fold a trajectory back into
  molforge's analysis layer (RMSD-vs-frame-0, DSSP-per-frame as
  stability indicators). Heavy `engine.prepare(...)` /
  `engine.run(...)` cells are marked `# 🐢 SLOW` and show their
  call signatures with documented expected outputs rather than
  executing — the notebook reads cleanly on GitHub without OpenMM
  installed.
- A synthetic-trajectory demonstration shows the analysis pattern
  using a `Trajectory` constructed from Gaussian-perturbed copies of
  the helix fixture, with an explicit note clarifying that real MD
  trajectories would behave very differently — the point is the
  API, not the dynamics.
- Updated `notebooks/README.md` (status table) and `README.md` (top-
  level callout block) to mark `03_md_simulations.ipynb` as live and
  list it alongside the other walkthroughs. All six walkthroughs in
  the v0.0.1 stub set now have at least one live entry — only
  `06_plugin_authoring.ipynb` remains as a stub.
- **`molforge.metrics`: evaluation metrics for protein prediction
  quality.**
  - **TM-score** (`tm_score`) — Zhang & Skolnick 2004 length-normalized
    structural similarity. Configurable normalization length
    (`reference` / `model` / `shorter` / `longer`); > 0.5 ≈ same fold.
  - **GDT-TS / GDT-HA** (`gdt_ts`, `gdt_ha`) — CASP's gold-standard
    metric. Average fraction of residues within 1/2/4/8 Å (TS) or
    0.5/1/2/4 Å (HA) after optimal superposition. `gdt_per_cutoff`
    exposes the underlying per-cutoff fractions for custom analyses.
  - **lDDT** (`lddt`, `lddt_per_residue`) — Mariani et al. 2013
    alignment-free local Distance Difference Test. Translation /
    rotation invariant by construction (no superposition required).
    This is the metric AlphaFold's pLDDT and ESMFold's pLDDT
    confidence scores estimate; molforge's `lddt_per_residue` is
    what you'd compare a pLDDT prediction against given a native
    structure.
  - **DockQ** (`dockq`, `fnat`, `irms`, `lrms`) — Basu & Wallner 2016
    single-number protein-protein complex quality, breaking down
    into Fnat (fraction of native interface contacts recovered),
    iRMS (interface backbone RMSD), and LRMS (ligand-chain RMSD
    after receptor superposition). Includes the CAPRI quality
    thresholds in the docstring.
  - All metrics are pure NumPy — no `tmalign` / `lddt` / DockQ
    binaries required.
  - Replaces the previous `tm_score` / `lddt` / `gdt_ts` stubs that
    raised `NotImplementedError`.
- New test fixtures for DockQ: `tests/fixtures/pdb/mini_complex_native.pdb`
  (2-chain helix-helix complex, 56 atoms), `mini_complex_good.pdb`
  (small 0.3 Å backbone noise applied to both chains), and
  `mini_complex_bad.pdb` (chain B placed 30 Å away — no interface
  preserved). These exercise the metric across the full quality
  range (DockQ ≈ 1.0, ≈ 0.8, < 0.3 respectively).
- 44 unit tests covering: TM-score `_d0` length-dependence, perfect-
  match / translation-invariance / rotation-invariance (after
  superposition), graded noise degradation, normalization modes,
  error paths for mismatched lengths; GDT-TS / HA identity + noise
  + monotonicity-across-cutoffs invariants; lDDT identity + small/
  large noise + the alignment-free property under translation and
  rotation; DockQ Fnat / iRMS / LRMS individual measures plus the
  combined score across the three quality bands.
- **Three new live walkthrough notebooks** (replacing the stubs that
  came with the v0.0.1 repo skeleton):
  - [`notebooks/walkthroughs/01_sequences.ipynb`](notebooks/walkthroughs/01_sequences.ipynb)
    (19 cells) — composition / properties, Needleman-Wunsch and
    Smith-Waterman alignment with BLOSUM62/PAM250, the
    protein-engineering mutation notation (`A123V`, `A1V/T56K`,
    `H:K42N`), wild-type validation, and `mutate_protein` on a real
    structure.
  - [`notebooks/walkthroughs/02_structures.ipynb`](notebooks/walkthroughs/02_structures.ipynb)
    (24 cells) — geometry primitives, Kabsch superposition + RMSD
    across atom subsets, per-residue RMSD that localizes structural
    differences, contact / distance maps, DSSP 8-state and 3-state,
    Shrake-Rupley SASA, backbone dihedrals (φ/ψ/ω) and Ramachandran.
  - [`notebooks/walkthroughs/05_ml_featurization.ipynb`](notebooks/walkthroughs/05_ml_featurization.ipynb)
    (16 cells) — every layer of `molforge.ml`: sequence featurizers
    (one-hot, BLOSUM, positional encoding, compose), structure
    featurizers (RBF-binned distances, pair orientations, local
    environment, combined node features), graph construction in the
    PyTorch Geometric / DGL convention, and ESM-2 embeddings via
    `ESM2Embedder` (heavy cells marked `# 🐢 SLOW` with code shown
    but not executed).
- **`notebooks/README.md`**: updated index with a status table
  showing which walkthroughs are live and which remain stubs.
- All notebook outputs are pre-baked from real runs against the
  bundled `mini_mixed.pdb` / `tripeptide.pdb` / `helix.pdb` fixtures
  so the notebooks render correctly on GitHub without requiring
  `torch`, `colabfold`, `vina`, or `openmm` to be installed.
- **`molforge.ml`: featurization for protein ML.**
  - **Sequence featurizers** (no structure required): `one_hot`
    (21-dim with `X` for unknowns), `blosum_embed` (BLOSUM62/PAM250
    rows as embeddings — surprisingly strong baseline featurization),
    `positional_encoding` (sinusoidal Vaswani-style), `compose_features`
    (concatenate any combination of per-residue featurizers).
  - **Structure featurizers** (need 3D coordinates):
    `pair_distances` (float32 distance map), `pair_distance_features`
    (Gaussian-RBF binned distances — the standard input for
    distance-based protein GNNs), `pair_orientations` (CA-CA unit
    vectors + cosines against local-frame forward direction),
    `local_environment` (atomic counts by element within a radius),
    `per_residue_features` (combined one-hot + environment + DSSP
    node features for GNNs).
  - **Protein language model embeddings**: `ESM2Embedder` wraps
    Meta's ESM-2 via HuggingFace transformers. Per-residue, batched,
    and pooled (`mean` / `max` / `cls`) extraction modes.
    Configurable model size, layer, device, dtype. Lazy imports so
    `import molforge.ml` stays light.
  - **Graph construction**: `to_graph` builds a `ProteinGraph` with
    `(node_features, edge_index, edge_features)` in the PyTorch
    Geometric / DGL convention. Configurable cutoff distance, self-
    loops, edge distance binning, and which node features to include.
  - Replaces the previous `featurize` / `to_tensor` / `to_graph` stubs
    that raised `NotImplementedError`.
- 54 unit tests covering: every featurizer (one-hot with/without
  unk, BLOSUM/PAM matrices, positional encoding values and odd-dim
  validation, compose_features shape compatibility); pair distances
  / RBF features / orientations / local environment / per-residue
  features against the helix fixture; graph construction (node
  count, dim correctness, self-loop handling, cutoff respect,
  edge feature dimensionality, bidirectional edges, empty-protein
  edge case); ESM2Embedder construction, lazy load, missing-dep
  error path, and a slow end-to-end test on the 8M model gated
  on torch availability.
- **`molforge.wrappers.md.OpenMM`: first MD-engine wrapper.**
  - Wraps [OpenMM](https://openmm.org/) (Eastman et al. 2017), the
    Python-first MD engine. GPU-accelerated, installable via pip
    (Linux/macOS) or conda-forge (Windows), supports the major
    modern force fields out of the box.
  - Same `prepare -> minimize -> run` flow as the abstract
    :class:`MDEngine` contract: `engine.prepare(protein,
    force_field="amber14-all")` returns a :class:`Simulation`,
    `engine.minimize(sim)` energy-minimizes in place,
    `engine.run(sim, n_steps=50_000, save_every=500)` returns a
    :class:`Trajectory` with recorded frames.
  - Configurable: platform (`CUDA`/`CPU`/`OpenCL`/auto), precision,
    nonbonded cutoff and method, bond constraints, force field,
    temperature, timestep.
  - Curated force-field name registry (`amber14-all`, `amber99sb`,
    `charmm36`, `amber99sb-ildn`) shipped with sensible defaults
    including matching water-model XMLs; any other OpenMM-recognized
    XML name passes through.
  - The OpenMM `Simulation` object is exposed at
    :attr:`Simulation.engine_handle` so users can drop down to the
    full OpenMM API when the wrapper's surface isn't enough.
  - Completes the **wrapper triad** (folding ✓, docking ✓, MD ✓).
    Combined with ESMFold + Vina, this is a full
    fold → minimize → equilibrate → dock loop in one library.
- **`molforge.md.Trajectory`, `Simulation`, `MDEngine`**: replaces
  the previous stubs that raised `NotImplementedError` on
  construction. `Trajectory` is iterable, indexable by frame, and
  exposes `topology`, `coordinates`, `times`, `energies`,
  `temperatures`, `metadata`. `Simulation` carries the engine
  handle, current coordinates/velocities/time, and force-field /
  integrator settings.
- **`MDEngineNotInstalledError`**: dedicated exception type for
  missing OpenMM, mirroring the folding/docking error conventions.
- 21 unit tests covering: Trajectory shape and iteration semantics,
  frame snapshots returning deep copies (mutating a frame must not
  corrupt the trajectory), Simulation construction and parameter
  passthrough, ABC contract enforcement (cannot instantiate
  `MDEngine` directly), end-to-end dummy-engine flow, OpenMM
  wrapper construction with various parameter combinations, lazy
  import behavior, missing-dependency error paths, run-parameter
  validation, and force-field registry correctness.
- **`molforge.wrappers.folding.AlphaFold`: second fully-implemented
  folding engine wrapper.**
  - Wraps [ColabFold](https://github.com/sokrypton/ColabFold)
    (Mirdita et al. 2022), the streamlined practical interface to
    AlphaFold (Jumper et al. 2021). Uses MMseqs2 remote MSA search
    so users don't need to host the ~3 TB AlphaFold MSA database
    locally.
  - Configurable AlphaFold parameters: number of models (1-5),
    number of recycling iterations (default 3), MSA mode
    (`mmseqs2_uniref_env` for full pipeline, `single_sequence` for
    MSA-free fast mode, comparable to ESMFold), model type
    (`AlphaFold2-ptm` or `AlphaFold2`), device.
  - Lazy import of `colabfold` keeps `import molforge` cheap;
    missing-dependency errors point at `pip install colabfold` with
    a link to ColabFold's platform-specific setup notes.
  - **Validates the wrapper pattern across two engines.** The
    uniform-confidence convention (`metadata["confidence_per_residue"]`,
    `mean_confidence`, `confidence_per_atom`) is identical to ESMFold's,
    so downstream code that filters or ranks by confidence reads from
    the same keys regardless of which engine produced the output. A
    test in `test_alphafold.py` explicitly verifies this.
  - Server mode is stubbed (raises `NotImplementedError` with a
    clear message) — coming in a future release.
- 12 unit tests covering construction, lazy loading, sequence
  validation, missing-deps error path, post-processing in isolation,
  and a cross-engine metadata-convention test that fails CI if either
  engine drifts away from the shared contract.
- **CI / release-pipeline hardening.**
  - `.github/workflows/release.yml` now verifies that the pushed tag
    matches `src/molforge/__init__.py`'s `__version__` before
    building, runs `twine check` on the built sdist + wheel, and
    smoke-installs the wheel in a clean venv before publishing.
    Mismatched-version tags now fail loudly rather than silently
    publishing the wrong version.
  - Workflow_dispatch trigger added with a `dry_run` input so the
    release pipeline can be exercised manually without uploading to
    PyPI.
  - `.github/workflows/ci.yml` adds a `smoke-install` job that
    installs the built wheel in a clean venv (no editable install,
    no source layout) and runs the test suite against the
    installed package. Catches packaging bugs — missing data files,
    wrong package layout — that the regular CI doesn't see.
  - The new smoke-install job already caught one bug during
    development: the smoke-check tried `from molforge import load`,
    which doesn't work (the top-level package surface is
    intentionally minimal). Fixed to use `from molforge.io import
    load, save` instead, matching the documented public API.
- **`docs/RELEASING.md`**: complete release procedure with SemVer
  guidance, step-by-step tagging instructions, the dry-run flow,
  one-time PyPI trusted-publishing setup, and a troubleshooting
  section for the most common pitfalls (version-tag mismatches,
  "file already exists", missing data files).
- **Real-world fixture suite for integration testing.** Four new
  hand-built PDB fixtures under `tests/fixtures/pdb/` exercise
  realistic structural patterns that the idealized helix and
  tripeptide didn't cover:
  - `mini_beta_sheet.pdb` (48 atoms, 12 residues) — two adjacent
    beta-strand-geometry segments.
  - `mini_mixed.pdb` (60 atoms, 15 residues) — alpha helix + loop +
    beta strand topology; DSSP correctly assigns `CHHHEEEEEEEECCC`.
  - `mini_ensemble.pdb` (96 atoms, 3 NMR-style models with random
    noise) — exercises multi-model parsing, model selection, and
    round-trip preservation.
  - `mini_with_ligand.pdb` (27 atoms: 5 protein residues + 5-atom
    imidazole ligand + 2 waters) — exercises the entity-type
    classifier across protein/ligand/water in a single file.
- **`tests/integration/test_fixtures.py`**: 19 integration tests that
  exercise the full IO -> data-model -> structural-analysis pipeline
  on each fixture. Catches integration bugs that pure unit tests
  miss (e.g. a regression in `entity_type` propagation, or in how
  multi-model files round-trip through `write_pdb`). Covers DSSP
  detection of mixed topology, phi/psi recovery, SASA pipeline,
  NMR ensemble multi-model handling, ligand vs. protein vs. water
  classification, `protein_only` and `remove_water` filtering,
  parametrized round-trip preservation, and a complete
  load -> analyze -> mutate -> compare end-to-end chain.
- **`molforge.structure.sasa`: solvent-accessible surface area (Shrake-Rupley).**
  - `sasa(protein)` — per-atom SASA in Å² via the standard
    Shrake-Rupley algorithm. Configurable probe radius (default 1.4 Å,
    water) and sphere-point count (default 100; 960 matches NACCESS).
  - `sasa_per_residue(protein)` — sum across atoms in each residue.
  - `total_sasa(protein)` — single-scalar shortcut.
  - Default van-der-Waals radii from the Bondi 1964 set with
    biomolecular adjustments (NACCESS / FreeSASA). Uses the
    golden-spiral / Fibonacci method for uniform sphere-point
    distribution.
  - Pure-NumPy implementation; no FreeSASA / mkdssp / Biopython
    dependency. ~1-2 s on a 3000-atom structure with 100 points.
- **`molforge.structure.dihedrals`: backbone dihedrals.**
  - `phi(protein)`, `psi(protein)`, `omega(protein)` — per-residue
    backbone dihedral angles in degrees, with `NaN` at chain termini
    or where backbone atoms are missing.
  - `phi_psi_omega(protein)` — all three at once (cheapest path).
  - `ramachandran(protein)` — `(n_res, 2)` φ/ψ pairs for plotting.
  - `dihedral(p1, p2, p3, p4)` — scalar dihedral via the standard
    `atan2(b1·(b2×b3), (b1×b2)·(b2×b3))` formula, no acos
    near-singular issues.
  - `dihedrals_batch(quartets)` — fully vectorized over an `(N, 4, 3)`
    array, matches the scalar function bit-for-bit.
- 24 unit tests for SASA (sphere-point uniformity, isolated-atom full
  exposure, two-atom occlusion, far-apart atoms full exposure,
  fixture-based per-atom / per-residue / total shape and non-
  negativity) and dihedrals (scalar at 0°, 90°, 180°; degenerate
  geometry returns NaN; batch matches scalar; chain termini are NaN;
  helix fixture has |φ| ≈ 60°, |ψ| ≈ 45°, |ω| ≈ 180°; empty-protein
  edge case).
- Replaces the previous `sasa` stub that raised `NotImplementedError`.
- **Automatic receptor / ligand preparation for Vina via meeko + RDKit.**
  - `molforge.wrappers.docking.prepare_receptor` / `prepare_ligand`:
    convert any of the common chemistry file formats (.pdb, .mmcif,
    .sdf, .mol, .mol2) into the PDBQT files Vina consumes, with
    Gasteiger charges, AutoDock atom types, and rotatable-bond
    identification. SMILES is supported for ligands via the
    `from_smiles=True` flag (uses RDKit's ETKDG to generate a 3D
    conformer first).
  - `Vina().dock(receptor=protein, ligand="ligand.sdf", ...)` now
    just works — meeko is invoked transparently when the input
    isn't already a PDBQT file. Previously this raised
    `NotImplementedError`.
  - Lazy imports: meeko and RDKit are only imported when prep is
    actually needed. Constructing a `Vina()` engine stays free.
  - Missing-dep errors point at `pip install meeko` and
    `pip install 'molforge[docking]'` so users can fix the setup
    without grepping the docs.
- 17 unit tests for the prep module: PDBQT passthrough for already-
  prepared files, missing-dep error paths for both meeko and RDKit,
  unsupported-extension validation, and a slow end-to-end SMILES-prep
  test gated on meeko/rdkit availability.
- **End-to-end design-loop notebook** (`notebooks/examples/end_to_end_design.ipynb`).
  A 20-cell worked example walking the full pipeline: sequence input
  with composition stats → ESMFold prediction → DSSP secondary
  structure + radius of gyration → point mutation → re-fold → per-
  residue RMSD, contact-map overlap, and DSSP-diff comparison
  between wild-type and mutant. Cells call out the heavy ESMFold
  inference steps as `# 🐢 SLOW` so the notebook renders correctly
  without GPU; the rest run against the bundled `helix.pdb` fixture.
  Every code cell ships pre-baked outputs that match the real
  runtime values so it makes sense to read straight on GitHub.
- **Vina docking walkthrough** (`notebooks/walkthroughs/04_docking.ipynb`).
  Replaces the previous stub with a real 15-cell walkthrough:
  engine construction, search-box specification, dock invocation,
  result iteration, pose-to-pose RMSD with `molforge.structure`,
  and run-metadata round-trip. Documents the current receptor /
  ligand prep requirement and points at the `meeko` integration on
  the roadmap.

## [v0.0.1] 2026-05-14

- **`molforge.structure.dssp`: Kabsch-Sander secondary-structure assignment.**
  - Pure-NumPy implementation of the canonical DSSP algorithm
    (Kabsch & Sander 1983) with no external dependencies — no DSSP
    binary required, no Biopython, no mkdssp install.
  - Returns both the full 8-state DSSP alphabet (`H` α-helix,
    `G` 3-10 helix, `I` π-helix, `E` β-strand, `B` β-bridge,
    `T` turn, `S` bend, `-` coil) and the 3-state collapse
    (`H` / `E` / `C`) via :func:`dssp_3state`.
  - Geometric backbone amide-H placement (no need for explicit H atoms
    in input), Kabsch-Sander electrostatic H-bond energy model, both
    parallel and antiparallel β-bridge detection.
  - Non-protein residues (water, ligands, ions) and residues with
    incomplete backbones get `-` rather than crashing.
  - Result dict also exposes the full ``(n_res, n_res)`` H-bond energy
    matrix for downstream analyses (custom topology metrics, contact-
    map enrichment, etc.).
  - Replaces the previous stub that raised `NotImplementedError`.
- New test fixture `tests/fixtures/pdb/helix.pdb` — an idealized
  15-residue poly-alanine α-helix built from canonical (φ, ψ) values
  via NeRF placement. Produces the expected DSSP `CHHHHHHHHHHHHHC`
  pattern.
- 12 unit tests covering empty / tiny inputs, helix recognition (≥7 of
  the middle 9 residues classified as H), 3-state collapse, alphabet
  validity, residue labels, H-bond matrix shape, and graceful handling
  of non-protein residues.
- **`molforge.wrappers.docking.Vina`: second fully-implemented engine wrapper.**
  - Wraps AutoDock Vina via the
    [`vina`](https://pypi.org/project/vina/) PyPI package, which bundles
    the Vina binary so no manual install is required.
  - Configurable scoring function (`vina` or `vinardo`), seed, CPU
    thread count, and verbosity.
  - Takes either a prepared `.pdbqt` file path or (eventually) a
    `Protein` plus charges; the receptor / ligand preparation path for
    `Protein` and `.pdb` / `.sdf` inputs raises a clear
    `NotImplementedError` pointing users at meeko / AutoDockTools.
  - Search box specified by `center` and `box_size` in Å.
  - Multi-pose PDBQT output parsed back into `DockingResult` / `Pose`
    objects with score (kcal/mol), RMSD lower/upper bounds vs the
    best pose, rank, and the ligand atoms as a `Protein`.
- **`molforge.docking`: completed ABC and result types.**
  - `Pose` and `DockingResult` dataclasses with `best`, `top_n`,
    iteration, and length helpers — replacing the previous stub classes
    that raised `NotImplementedError` on construction.
  - `DockingEngine` ABC with the formal `dock` contract; mirrors
    `FoldingEngine` for API consistency across wrapper categories.
  - `DockingEngineNotInstalledError` for missing-dependency error paths.
- 28 unit tests (1 marked `@pytest.mark.slow` for the real engine):
  construction is dependency-free, lazy import behavior, materialization
  helpers (path passthrough plus clear errors for unsupported inputs),
  and exhaustive PDBQT output parsing (multi-MODEL, single-pose-no-MODEL,
  score/RMSD extraction, best-first sorting, rank reassignment, empty
  input).
- **`molforge.sequence`: sequence operations subpackage.**
  - **Pairwise alignment** (`align`, `needleman_wunsch`, `smith_waterman`,
    `Alignment`, `identity`): pure-NumPy Needleman-Wunsch (global) and
    Smith-Waterman (local) with affine gap penalties, BLOSUM62 / PAM250
    substitution matrices, and a `format()` method for human-readable
    alignment blocks. No external dependencies (no Biopython, no
    parasail) so it works in the minimal install.
  - **Substitution matrices** (`BLOSUM62`, `PAM250`, `get_matrix`,
    `available_matrices`): hardcoded as NumPy arrays from the NCBI BLAST
    distribution. No runtime data-file dependency.
  - **Mutations** (`Mutation`, `parse_mutations`, `apply_mutation`,
    `apply_mutations`, `mutate_protein`): protein-engineering notation
    (`A123V`, `A123V/T56K`, chain-prefixed `H:K42N`) with parsing,
    sequence-level application, and `Protein`-level application that
    updates the canonical `AtomArray` (sequence-only — atoms stay put;
    side-chain rebuilding is a job for Rosetta or OpenMM).
  - **Composition / properties** (`composition`, `length`,
    `molecular_weight`, `gravy`, `aromaticity`): the everyday sequence
    stats — per-residue counts/fractions, monoisotopic MW with
    terminal water, Kyte-Doolittle GRAVY score, aromatic fraction.
- 65 unit tests covering alignment correctness (identity, gaps, full and
  local coverage), matrix lookup and symmetry, mutation parsing
  (including chain prefix and multi-mutant syntax), `Protein` mutation
  with original-unchanged semantics, and all composition helpers.
- **`molforge.structure`: structural analysis subpackage.**
  - **Superposition** (`superpose`, `kabsch_rmsd`, `SuperpositionResult`):
    Kabsch / Umeyama optimal rigid-body alignment via SVD with proper-
    rotation guarantee (no reflections), optional per-point weights for
    masking outliers, returns rotation + translation + aligned coords +
    RMSD.
  - **RMSD** (`rmsd`, `rmsd_raw`, `rmsd_per_residue`): structure-to-
    structure RMSD with five atom-subset selectors (``ca``, ``backbone``,
    ``backbone_o``, ``all_heavy``, ``all``), optional alignment, and
    per-residue breakdown for localizing structural differences.
  - **Contacts** (`contact_map`, `distance_map`, `residue_contacts`):
    binary contact maps at configurable cutoff, continuous distance
    maps, and all-atom inter-residue contact listings with chain-pair
    filtering for interface analysis.
  - **Geometry** (`centroid`, `center_of_mass`, `radius_of_gyration`,
    `bounding_box`, `translate`, `rotate`, `center_at_origin`): bulk
    geometric properties (mass-weighted or geometric) and in-place
    coordinate transforms that mutate the canonical `AtomArray` directly.
  - Stubs (still `NotImplementedError`): `sasa`, `dssp`.
- 43 unit tests covering superposition correctness (identity, translation,
  rotation, noisy alignment, proper-rotation guarantee, weighted),
  RMSD computations across atom subsets, contact / distance map
  symmetry and chain filtering, and all geometry operations.

- **`molforge.io.read_cif` / `write_cif`: mmCIF / PDBx implementation.**
  - Full read/write of the ``_atom_site`` loop, the only mmCIF block
    that holds atomic coordinate data.
  - Hand-written tokenizer handles quoted strings, comments,
    semicolon-bounded multi-line text fields, and the ``.``/``?`` sentinel
    values for missing/unknown.
  - Header metadata extracted: ``_entry.id``, ``_struct.title``,
    ``_exptl.method``, ``_refine.ls_d_res_high``.
  - Preference for ``auth_*`` columns (matching PDB conventions) with
    fallback to ``label_*``, so PDB↔mmCIF round-trips preserve
    author-assigned chain IDs and residue numbers.
  - Reuses the same altloc resolution strategies and entity-type
    classification as the PDB parser for behavioural consistency.
  - ``CIFParseError`` and ``CIFWriteError`` for typed error handling.
  - Wired into the top-level :func:`load` / :func:`save` dispatcher so
    ``.cif`` and ``.mmcif`` extensions just work.
- 27 unit tests covering the tokenizer, parsing, write, round-trip
  (CIF→CIF and PDB→CIF→Protein), dispatch, and error paths.
- **`molforge.wrappers.folding.ESMFold`: first fully-implemented engine wrapper.**
  - Wraps Meta AI's ``facebook/esmfold_v1`` via HuggingFace
    ``transformers``. Single-sequence folding (no MSA needed), fast,
    GPU-friendly.
  - Lazy import of ``torch`` and ``transformers`` keeps ``import
    molforge`` cheap; missing-dependency errors point users at the
    correct ``pip install 'molforge[ml]'`` extra.
  - Configurable device (``cuda``/``cpu``/``mps``/auto), chunk size for
    long-sequence memory management, and dtype (``float32``/``float16``).
  - pLDDT exposed uniformly: per-atom in
    ``metadata["confidence_per_atom"]``, per-residue in
    ``metadata["confidence_per_residue"]``, scalar mean in
    ``metadata["mean_confidence"]``.
- **`FoldingEngine` ABC**: full contract definition with ``predict`` (abstract),
  ``predict_many`` (overridable batch), and a uniform per-residue
  confidence convention so downstream code reads engine output the same
  way regardless of which engine produced it.
- **`FoldingEngineNotInstalledError`**: dedicated exception type for missing
  heavy dependencies, with actionable error messages.
- 17 unit tests covering construction, lazy loading, sequence
  validation, missing-dependency error paths, and post-processing
  (PDB-to-Protein conversion with confidence metadata). The end-to-end
  fold test is marked ``@pytest.mark.slow`` and skipped unless ``torch``
  is installed.

### Changed
- **Project renamed from `biocore` to `molforge`** (PyPI name collision; the
  `biocore` GitHub organization is a separate, established scientific
  Python community). Import path is now `molforge`.
- README rewritten around the cross-tool workflow thesis: molforge is
  positioned as connective tissue between docking, MD, folding, design,
  and experimental tools, rather than primarily as a data-representation
  library.

### Added
- **`molforge.io`: file I/O subsystem.**
  - **PDB reader and writer** (`read_pdb`, `write_pdb`, plus their
    `*_string` variants). Handles the full wwPDB v3.30 column layout,
    HEADER/TITLE/EXPDTA/REMARK 2 metadata, NMR multi-model files,
    alternate locations (with three resolution strategies:
    `highest_occupancy`, `first`, `all`, or a specific altloc id),
    insertion codes, gzipped input/output, hydrogen filtering, and
    automatic entity-type classification (protein / dna / rna / water /
    ion / ligand) per residue.
  - **FASTA reader and writer** (`read_fasta`, `write_fasta`, `*_string`
    variants) with `FastaRecord` dataclass. Tolerant of multi-line
    sequences, embedded digits, comments, and blank lines.
  - **AlphaFold helpers** (`load_alphafold`, `is_alphafold_pdb`). Lifts
    pLDDT out of the B-factor column into `protein.metadata["plddt"]`
    (per atom), `protein.metadata["plddt_per_residue"]`, and
    `protein.metadata["mean_plddt"]`. B-factor column preserved for
    downstream-tool compatibility.
  - **Top-level `load()`, `save()`, `fetch()`** dispatch by file
    extension or explicit `format=` keyword. `fetch()` is stubbed
    pending an HTTP utility.
  - Format stubs (raising `NotImplementedError` with clear pointers):
    `mmcif`, `pdbqt`, `pqr`, `sdf`, `mol2`. The API surface is committed
    so user code targeting these formats won't break when
    implementations land.
  - `PDBParseError`, `PDBWriteError` for typed error handling.
- 73 unit tests covering PDB parsing, writing, round-trip correctness on
  real fixtures (dipeptide, NMR ensemble, altloc, insertion-coded),
  FASTA edge cases (multiline, digits, comments, malformed input),
  AlphaFold detection and pLDDT extraction, and dispatch behavior.
- `ACKNOWLEDGEMENTS.md` crediting Protkit, Biotite, Biopython,
  BioPandas, MDAnalysis, OpenMM, RDKit, and the file-format
  specifications we implement.
- **`molforge.core`: full implementation of the canonical data model.**
  - `AtomArray`: flat, NumPy-backed source of truth with 15 typed fields
    (coords, element, atom_name, residue_name, residue_id, insertion_code,
    chain_id, b_factor, occupancy, charge, serial, record_type, entity_type,
    altloc, model_id). Supports construction from a dict of arrays, boolean
    selection (`select`, `where`), slicing, fancy indexing, and concatenation
    (`append`). Lazily-computed and cached residue / chain boundary indices
    (`residue_starts`, `chain_starts`) with explicit invalidation.
  - `Atom`, `Residue`, `Chain`, `Protein`: lightweight hierarchical *views*
    that hold a reference to a shared `AtomArray` plus an index range.
    Mutations on the hierarchical side write through to the array, so the
    two views never go out of sync.
  - `Protein.select(**filters)`, `protein_only()`, `remove_water()` for
    common substructure operations.
  - First-class support for heterogeneous content: HETATM records,
    ligands, waters, and ions are represented in the same array via
    `record_type` and `entity_type` fields.
  - Insertion codes, alternate locations (altloc), and multi-model
    (NMR / trajectory) structures are modeled from day one.
  - Constants module with three-letter ↔ one-letter mappings for the 20
    canonical amino acids, common non-canonical residues (MSE, SEC,
    phospho-S/T/Y, force-field-specific His variants), DNA / RNA
    nucleotides, waters, and ions. Helper functions `three_to_one`,
    `is_standard_amino_acid`, `is_water`, `is_ion`.
- 74 unit tests covering `AtomArray`, hierarchical views, constants, and
  cross-cutting hierarchical ↔ linear consistency invariants.
- Initial repository skeleton with src-layout package structure.
- Hierarchical data model stubs for `Protein`, `Chain`, `Residue`, `Atom`.
- Linear / array view stubs (`AtomArray`) alongside hierarchical views.
- Top-level subpackages: `core`, `sequence`, `structure`, `md`, `docking`,
  `ml`, `io`, `plugins`, `metrics`.
- Wrapper interface stubs for folding (AlphaFold/ColabFold, ESMFold, Boltz),
  docking (AutoDock Vina, DiffDock), and MD (OpenMM, GROMACS).
- Plugin registry stub with entry-point discovery.
- `pyproject.toml` with PEP 621 metadata, extras (`structure`, `sequence`,
  `md`, `docking`, `ml`, `io`, `all`, `dev`, `docs`), and tool config for
  ruff, mypy, pytest, and coverage.
- GitHub Actions workflows for CI (lint, type-check, tests on Python 3.10-3.12),
  documentation build, and release-to-PyPI on tag.
- Issue templates (bug, feature, question) and PR template.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CODEDoctorDeanS`.
- Walkthrough notebook stubs for sequences, structures, MD, and docking.
- Pinned requirements files per extra under `requirements/`.

[Unreleased]: https://github.com/DoctorDean/molforge/compare/HEAD...HEAD
