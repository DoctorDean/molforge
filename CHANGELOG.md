# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
