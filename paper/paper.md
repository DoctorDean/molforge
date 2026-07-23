---
title: "molforge: a unified, reproducible interface for computational protein science"
tags:
  - Python
  - structural bioinformatics
  - protein structure prediction
  - molecular docking
  - molecular dynamics
  - protein design
  - reproducibility
authors:
  - name: "Dean Sherry"
    orcid: 0000-0003-2582-0868
    affiliation: 1
affiliations:
  - name: "Independent, South Africa"
    index: 1
date: 1 August 2026                  # TODO: submission date
bibliography: paper.bib
---

# Summary

Modern protein science is inherently multi-tool: a single study may fold a
sequence with a deep-learning model, dock a ligand with a search engine,
relax the complex with a molecular-dynamics integrator, design new sequences
with a generative model, and score the result with a learned potential. Each
tool speaks its own dialect — different file formats, atom-naming
conventions, and notions of what "the structure" even is — so much of a
researcher's effort is spent on format conversion and glue code rather than
science, and end-to-end workflows are rarely reproducible.

`molforge` is a Python library that gives these tools a shared vocabulary. At
its core is a canonical, NumPy-backed `Protein` data model [@numpy] that
exposes both a hierarchical view (`protein.chains["A"].residues[42]`) and a
flat, machine-learning-friendly array view (`protein.atom_array.coords`) over
the same data, with no conversion between them (\autoref{fig:datamodel}).
Around this substrate, `molforge` wraps sixteen widely used engines across six
modalities behind consistent interfaces, adds a from-scratch and independently
validated structural-analysis stack, and layers on reproducibility
infrastructure — provenance tracking, content-addressed caching, and citable,
re-executable pipeline manifests. Higher-level workflows built on this
foundation include cross-engine structure ensembles, an iterative
protein-design loop, and a direction-aware scoring interface
(\autoref{fig:pipeline}).

![molforge's canonical data model. A single `Protein`/`AtomArray` object that
every engine reads from and writes to, exposing a hierarchical view (chain →
residue → atom) for biology and a flat coordinate-array view for machine
learning over the *same* data with no conversion, alongside first-class
support for heterogeneous content — protein, ligand, water and ions, glycans
and lipids.\label{fig:datamodel}](molforge_datamodel.pdf)

# Statement of need

Existing libraries solve parts of this problem well but not the whole.
Biopython [@biopython] and Biotite [@biotite] provide structure parsing and
manipulation; MDAnalysis [@mdanalysis] focuses on trajectory analysis; RDKit
handles small-molecule cheminformatics. None offers a single data model that
spans the *current* engine landscape — deep-learning structure prediction
[@alphafold; @esmfold; @boltz], diffusion-based docking and design
[@diffdock; @rfdiffusion; @proteinmpnn], classical docking [@vina], and
molecular dynamics [@openmm] — together with the reproducibility machinery a
multi-engine workflow needs.

`molforge` targets researchers who compose *several* tools and are tired of
writing and re-writing the connective code between them. Rather than
reimplementing the engines, it provides a common representation they all read
from and write to, so swapping one folding model for another, or feeding a
docked pose into a molecular-dynamics run, is a one-line change instead of a
format-conversion project. The design is a library, not a framework: there is
no orchestrator or workflow runtime to adopt — it is plain imports, usable
inside Snakemake, Nextflow, a notebook, or a shell script.

# Features

- **Canonical data model.** One `Protein`/`AtomArray` representation with
  hierarchical and linear views, first-class support for heterogeneous
  content (ligands, ions, nucleic acids, glycans), and I/O for the common
  structural-biology formats (PDB, mmCIF, FASTA, PDBQT, PQR, SDF, MOL2).
- **Engine wrappers** across folding (ESMFold, AlphaFold/ColabFold, Boltz,
  Chai-1, RoseTTAFold), docking (AutoDock Vina, Gnina, DiffDock), molecular
  dynamics (OpenMM, GROMACS, AMBER), generative design (RFdiffusion,
  ProteinMPNN, ESM-IF1), binding free energy (MM-PB(GB)SA, FEP/TI ingestion),
  and pocket detection (fpocket, P2Rank [@p2rank]) — each behind a consistent
  interface.
- **Validated analysis stack.** From-scratch implementations of RMSD,
  superposition, SASA, DSSP, contact/distance maps, backbone dihedrals, and
  the standard comparison metrics — TM-score [@tmscore], GDT, lDDT
  [@lddt], and DockQ [@dockq]. Each is validated against an independent
  reference implementation — TM-align [@tmalign], MDTraj [@mdtraj], the
  reference DockQ package [@dockq], and Biopython [@biopython] — and the
  golden values are committed as regression tests, so the numbers match the
  literature.
- **Reproducibility.** Every output carries a `Provenance` record — engine,
  version, parameters, inputs, and a pointer to the step it consumed — that
  can be emitted as a citable `pipeline.yaml` manifest and re-executed with a
  single `replay()` call (\autoref{fig:pipeline}). A content-addressed cache
  makes repeated steps instant.
- **Cross-engine ensembles.** `cross_engine_fold` folds one sequence with
  several engines, superposes the models, and returns their pairwise
  TM-score/RMSD spread, a consensus structure, and a per-residue map of where
  the engines disagree — a model-agnostic confidence signal that complements
  any single model's self-reported score.
- **Design and scoring.** A `DesignLoop` runs the generate → fold → dock →
  score → iterate loop with a ranked design table, and `molforge.scoring`
  gives every score an explicit direction so heterogeneous scorers (docking
  affinity, folding confidence, learned potentials) rank and compare
  uniformly.

![A molforge workflow. At each stage — fold, consensus, assemble, simulate,
score — the shared `Protein` object is handed to any of several
interchangeable engines (listed beneath each stage), while a provenance trail
along the bottom records every step so the whole run can be emitted as a
citable `pipeline.yaml`, cached, and re-executed exactly with
`replay()`.\label{fig:pipeline}](molforge_pipeline.pdf)

# Design and quality

`molforge` is written in typed Python with a numpy-only core (heavy
dependencies are opt-in extras), so it installs without a compiler. The
project ships an extensive automated test suite run on a three-OS ×
three-Python-version matrix, static type checking, and continuous
integration, including an opt-in nightly workflow that exercises the
CPU-installable engines against their real implementations rather than mocks.
Third parties can register additional engines, parsers, or scorers through a
plugin entry-point group without forking the library.

# Related software

`molforge` is complementary to, rather than a replacement for, the libraries
above: it reads and writes their formats and is designed to interoperate.
Where Biopython, Biotite, MDAnalysis, or RDKit already solve a problem well,
`molforge` builds on their formats rather than competing, and its distinct
contribution is the shared data model and reproducibility layer that let a
workflow span many engines at once.

# Acknowledgements

`molforge` wraps and builds upon a large body of open-source scientific
software; we are grateful to the developers of the engines and libraries it
depends on.

# References
