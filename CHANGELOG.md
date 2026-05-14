# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
