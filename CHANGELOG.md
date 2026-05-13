# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added 

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
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CODEOWNERS`.
- Walkthrough notebook stubs for sequences, structures, MD, and docking.
- Pinned requirements files per extra under `requirements/`.

[Unreleased]: https://github.com/DoctorDean/molforge/compare/HEAD...HEAD
