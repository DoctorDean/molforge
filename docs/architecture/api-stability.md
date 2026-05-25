# API stability

This page records which parts of molforge's public API are
**committed** (stable, semver-protected) versus **tentative** (may
change before 1.0). It's the output of the pre-1.0 API audit.

molforge follows [Semantic Versioning](https://semver.org). Once 1.0
ships, anything documented here as *committed* will not break within
the 1.x series without a major-version bump.

## What "public API" means

The public API is everything reachable from a module's `__all__`,
across the package: `molforge.core`, `molforge.io`,
`molforge.sequence`, `molforge.structure`, `molforge.ml`,
`molforge.metrics`, `molforge.validation`, `molforge.md`,
`molforge.docking`, `molforge.ensembles`, `molforge.plugins`, and the
`molforge.wrappers.*` subpackages.

Names prefixed with an underscore are private and may change at any
time, with no deprecation cycle.

## Committed surface

The following are considered stable and ready to freeze for 1.0:

- **Core data model** — `Protein`, `Chain`, `Residue`, `Atom`,
  `AtomArray`, and the `ATOM_FIELDS` schema. The hierarchical-view /
  flat-array design is settled.
- **`Protein.metadata` key vocabulary** — `Protein.metadata` is a
  free-form `dict`, but the *documented keys* in
  [`molforge.core.metadata_keys`](../reference/core.md) are a stable
  contract. See "Metadata" below.
- **I/O** — `load`, `save`, `fetch`, and the format-specific
  `read_*` / `write_*` functions for PDB, mmCIF, and FASTA. The
  `FastaRecord` type.
- **Sequence, structure, metrics, ML featurization** — the analysis
  functions (alignment, RMSD, contacts, SASA, dihedrals, DSSP, lDDT,
  GDT, graph and structure featurization) have stable signatures.
- **Validation** — `Criterion`, `CriteriaSet`, `Verdict`,
  `cross_validate`, `consensus`, `rank_verdicts`. Note the
  `cross_validate` error-handling change below.
- **Ensembles** — the seven `molforge.ensembles` functions and the
  `DensityGrid` / `PoseCluster` / `PoseClusteringResult` dataclasses.
- **Engine-wrapper contracts** — the `FoldingEngine`, `MDEngine`,
  and `DockingEngine` abstract bases, and the uniform metadata
  conventions their implementations follow.

## Type checking

The **entire `molforge` package** is verified under `mypy --strict` —
all 77 source modules, every subpackage. CI enforces it: the
`typecheck` job's `Mypy (strict)` step runs `mypy src` (the
`[tool.mypy]` config sets `strict = true`) and fails the build on any
new type error. A `slow`-marked regression test
(`tests/unit/test_typing.py`) runs the same check in-suite, so a type
regression is caught in a local test run too.

This was reached incrementally — `core` first, then the analysis
subpackages, then `ml`, and finally the `wrappers` and `plugins`
subpackages — but the whole tree is now strict-clean and gated as a
single check.

## Tentative surface

The following are intentionally **not** frozen yet. They are exported
so the import path is stable, but the behavior or signature may still
change:

- **Unimplemented format stubs** — `read`/`write` for SDF, MOL2,
  PDBQT, and PQR raise `NotImplementedError` with a clear pointer.
  The *import paths* are committed; the implementations are planned.
- **`DiffDock`** (`molforge.wrappers.docking`) — committed import
  path and `DockingEngine` contract, but `dock()` raises
  `NotImplementedError`. Use `Vina` for working docking.
- **`GROMACS`** (`molforge.wrappers.md`) — committed import path and
  `MDEngine` contract, but all methods raise `NotImplementedError`.
  Use `OpenMM` for working MD.
- **`Rosetta`** (`molforge.wrappers.folding`) — a deprecated alias
  for `RoseTTAFold`, retained for backward compatibility. It emits
  `DeprecationWarning` and will be removed in a future minor
  release.
- **`Simulation.engine_handle`** — see "Engine-private fields" below.

## Audit-driven changes

The pre-1.0 audit made these changes to lock down the surface:

### `cross_validate` error handling

`cross_validate` previously defaulted to `on_error="record"`, which
silently caught validator exceptions. A validator that threw on every
design produced a full list of `passed=False` verdicts that looked
like a real result, hiding the bug.

**The default is now `on_error="raise"`** — a validator exception
propagates immediately. Code that wants a long batch to survive
individual failures must pass `on_error="record"` explicitly. This
changed in 0.2.

### `Protein.metadata`

`Protein.metadata` stays a free-form `dict[str, Any]` — that's
deliberate, since parsers attach open-ended data (PDB `REMARK`
records, engine-specific extras). What changed is that the keys
molforge's own parsers and wrappers produce are now a *documented
vocabulary*: string constants and a `ProteinMetadata` TypedDict in
[`molforge.core.metadata_keys`](../reference/core.md).

Code can rely on the documented keys being stable across 1.x. Keys
outside the vocabulary are still permitted but carry no stability
guarantee.

The audit also fixed a real inconsistency: `load_alphafold` wrote
only AlphaFold-specific keys while the AlphaFold *wrapper* wrote the
cross-engine-uniform keys. `load_alphafold` now writes both.

### `fetch`

`io.fetch` was previously a `NotImplementedError` stub despite being
exported. It is now implemented (RCSB and AlphaFold DB, over stdlib
`urllib` — no new dependency) and is part of the committed surface.

### Stub coherence

`GROMACS` and `DiffDock` were exported but incoherent — `GROMACS`
couldn't even be instantiated (it didn't implement its ABC), and
both raised bare `NotImplementedError` with no message. They are now
*coherent* stubs: instantiable, satisfying their engine ABCs, and
raising `NotImplementedError` with a message that points at the
working alternative (`OpenMM` / `Vina`).

## Engine-private fields

Some fields are exported (they're attributes of public dataclasses)
but are explicitly **not** part of the API contract:

- **`Simulation.engine_handle`** — an opaque, engine-specific handle
  (an OpenMM `Simulation` object, a GROMACS run-directory handle,
  ...). Its type is `object` deliberately. Callers must not inspect
  it, depend on its type, or set it. It is not serialized — it
  typically wraps unpicklable C-extension state — and carries no
  semver guarantee. For inspectable per-simulation data, use
  `Simulation.metadata`.

## Metadata

For the full documented key vocabulary, see
[`molforge.core.metadata_keys`](../reference/core.md). In brief:

- **Structural-IO header keys** — `pdb_id`, `title`,
  `classification`, `deposition_date`, `experimental_method`,
  `resolution`. Set by the PDB and mmCIF parsers.
- **Uniform folding-engine keys** — `engine`, `source_sequence`,
  `confidence_per_residue`, `confidence_per_atom`,
  `mean_confidence`. Set by *every* folding-engine wrapper, so
  downstream code can read confidence without knowing which engine
  ran.
- **Engine-specific folding keys** — `ptm`, `iptm`, `pae`,
  `pae_inter`, and others. Set by some engines but not all.
