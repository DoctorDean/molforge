# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
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
