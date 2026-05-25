# molforge

[![CI](https://github.com/DoctorDean/molforge/actions/workflows/ci.yml/badge.svg)](https://github.com/DoctorDean/molforge/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/molforge.svg)](https://pypi.org/project/molforge/)
[![Python versions](https://img.shields.io/pypi/pyversions/molforge.svg)](https://pypi.org/project/molforge/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

![Molforge Logo](molforge.png)

> **A forge for protein workflows.** One Python script, every tool in your stack: docking, MD, folding, antibody and nanobody engineering, de novo design — without the format-conversion tax.

`molforge` is an open-source Python library that lets you compose protein workflows across the tools you already use. Bring your structures and sequences in, plug in your engines of choice (Vina, OpenMM, ESMFold, AlphaFold, RFdiffusion, ProteinMPNN, your own model), and walk out with a coherent pipeline instead of five incompatible Python environments and a graveyard of conversion scripts.

**Documentation:** [doctordean.github.io/molforge](https://doctordean.github.io/molforge/)

---

## Why molforge exists

Modern protein work is multi-tool by nature. A real antibody-design loop might fold a sequence with ESMFold, identify CDR loops with anarci, score binding with FoldX, dock against a target with AutoDock Vina, relax with OpenMM, then evaluate with Rosetta. **Each of those tools speaks its own dialect**: different file formats, different atom-naming conventions, different ideas of what "the structure" is. Most of an engineer's day disappears into glue code.

`molforge` is the connective tissue. It provides:

- A **canonical, NumPy-backed data model** that's cheap to convert in and out of — so every engine in your pipeline reads from and writes to the same representation.
- **Thin wrappers** around the engines you already trust, with consistent interfaces (so swapping ESMFold for AlphaFold is one line, not a refactor).
- **First-class IO** for the messy reality of structural-bio files: PDB, mmCIF, FASTA, PDBQT, PQR, SDF, MOL2, and AlphaFold predictions with pLDDT.
- A **plugin registry** so the next docking engine, folding model, or scoring function can slot into your pipeline without forking molforge.

Built as a library, not a framework: there's no orchestrator, no DAG runtime, no decorators you have to import to make things work. Use whatever workflow tool you like — Snakemake, Nextflow, Prefect, a shell script — molforge is just imports.

## Design principles

1. **Workflows over silos.** Every design decision is judged by "does this make it easier to chain N tools together?"
2. **Wrappers, not reimplementations.** We don't rebuild OpenMM or AutoDock. We give them a shared vocabulary.
3. **One data model, two views.** Hierarchical (`protein.chains["A"].residues[42]`) for biology, linear (`protein.atom_array.coords`) for ML — same data, no conversion.
4. **Heterogeneous content is first-class.** Antibodies have glycans. Drug targets have ligands and ions. Membrane proteins have lipids. The data model handles all of it without an awkward special case for "non-protein."
5. **Typed, tested, documented.** Strict mypy, ruff-clean, >90% coverage target.

## Installation

```bash
# minimal core (data model + sequence + basic IO)
pip install molforge

# with structure analysis (RMSD, SASA, contacts)
pip install "molforge[structure]"

# with ML wrappers (torch, transformers, esm)
pip install "molforge[ml]"

# with MD support (openmm, mdtraj)
pip install "molforge[md]"

# with docking (rdkit for small molecules)
pip install "molforge[docking]"

# everything
pip install "molforge[all]"

# development
git clone https://github.com/DoctorDean/molforge.git
cd molforge
pip install -e ".[dev,all]"
```

## Quickstart

The smallest end-to-end example that shows the cross-tool point:

```python
import molforge as mf
from molforge.wrappers.folding import ESMFold
from molforge.wrappers.docking import Vina
from molforge.wrappers.md import OpenMM

# 1. Fold a sequence
folded = ESMFold().predict("MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVS...")

# 2. Save as PDB, save as mmCIF, hand to anything
mf.save(folded, "candidate.pdb")
mf.save(folded, "candidate.cif")

# 3. Dock a ligand against it (Vina-prepared PDBQT files)
result = Vina().dock(
    receptor="receptor.pdbqt",
    ligand="ligand.pdbqt",
    center=(10.0, 5.0, -2.0),
    box_size=(20.0, 20.0, 20.0),
)
top_pose = result.best

# 4. Drop into MD for relaxation
trajectory = OpenMM().simulate(top_pose.complex, steps=10_000)

# 5. Inspect — hierarchical or linear, your call
print(folded.sequence)                            # one-letter per chain
print(folded.atom_array.coords.shape)             # (N, 3) NumPy array
ca = folded.chains["A"].residues[42].atoms["CA"]  # specific atom
```

Notice what *isn't* there: file-format conversions, atom-name remapping, hand-rolled PDB parsers, custom data classes per engine. molforge does that work so your script reads like the science you're actually doing.

> **Worked examples and walkthroughs** ([`notebooks/`](notebooks/)):
> - [`de_novo_design.ipynb`](notebooks/examples/de_novo_design.ipynb) — *de novo* design loop: RFdiffusion → ProteinMPNN → ESMFold → scoring.
> - [`cross_engine_validation.ipynb`](notebooks/examples/cross_engine_validation.ipynb) — two-validator consensus pattern in detail (ESMFold + AlphaFold).
> - [`end_to_end_design.ipynb`](notebooks/examples/end_to_end_design.ipynb) — full mutation loop: fold → analyze → mutate → re-fold → compare.
> - [`01_sequences.ipynb`](notebooks/walkthroughs/01_sequences.ipynb) — alignment, mutations, composition.
> - [`02_structures.ipynb`](notebooks/walkthroughs/02_structures.ipynb) — RMSD, contacts, DSSP, SASA, dihedrals.
> - [`03_md_simulations.ipynb`](notebooks/walkthroughs/03_md_simulations.ipynb) — OpenMM `prepare → minimize → run` flow, trajectory analysis.
> - [`04_docking.ipynb`](notebooks/walkthroughs/04_docking.ipynb) — Vina with automatic ligand prep.
> - [`05_ml_featurization.ipynb`](notebooks/walkthroughs/05_ml_featurization.ipynb) — one-hot, RBF distances, ESM-2 embeddings, graph construction.
> - [`06_plugin_authoring.ipynb`](notebooks/walkthroughs/06_plugin_authoring.ipynb) — register custom engines, parsers, and scorers.

## Repository structure

```
molforge/
├── src/molforge/             # Library source (src-layout)
│   ├── core/                 # Hierarchical + linear data model
│   ├── sequence/             # Sequence operations, alignment, mutations
│   ├── structure/            # RMSD, SASA, contacts, geometry
│   ├── md/                   # MD trajectories and analysis
│   ├── docking/              # Docking abstractions and pose handling
│   ├── ml/                   # ML utilities, featurizers, tensor views
│   ├── io/                   # PDB, mmCIF, FASTA, PDBQT, PQR, SDF, MOL2
│   ├── plugins/              # Plugin registry and entry-point discovery
│   ├── metrics/              # TM-score, lDDT, GDT-TS, docking metrics
│   └── wrappers/             # Thin interfaces to external engines
│       ├── folding/          # AlphaFold, ESMFold, Boltz, Rosetta
│       ├── docking/          # AutoDock Vina, DiffDock
│       └── md/               # OpenMM, GROMACS
├── tests/                    # pytest suite (909 passing + skips)
│   ├── fixtures/pdb/         # synthetic mini_*.pdb + realistic real_*.pdb fixtures
│   ├── unit/                 # per-subpackage unit tests
│   ├── integration/          # end-to-end tests against the realistic fixtures
│   └── benchmarks/           # performance benchmarks (pytest -m benchmark)
├── docs/                     # Architecture docs and reference
├── notebooks/                # Walkthroughs and worked examples
├── plugins/                  # Example external plugins
├── pyproject.toml            # Build config, deps, tool config
└── ACKNOWLEDGEMENTS.md       # Prior art and intellectual debts
```

A deeper architecture walkthrough is in [`docs/architecture/overview.md`](docs/architecture/overview.md).

## Status

molforge is **pre-1.0** and under active development. What's working today:

- **Core data model** — `Protein` / `Chain` / `Residue` / `Atom` over a canonical NumPy-backed `AtomArray`, with first-class heterogeneous content (ligands, water, ions, modified residues).
- **File I/O** — full read/write for **PDB** (with NMR ensembles, altlocs, insertion codes) and **mmCIF** (the modern format for large structures); **FASTA** sequence I/O; **AlphaFold** loader that surfaces pLDDT as a first-class field. PDBQT, PQR, SDF, MOL2 are stubbed with committed APIs.
- **Sequence operations** — pairwise **alignment** (Needleman-Wunsch / Smith-Waterman with BLOSUM62 / PAM250), point **mutations** with protein-engineering notation (`A123V`, `A123V/T56K`, `H:K42N`), composition and property helpers (MW, GRAVY, aromaticity).
- **Structural analysis** — Kabsch/Umeyama **superposition**, **RMSD** (whole-structure and per-residue, multiple atom subsets), **contact and distance maps**, **radius of gyration**, **centroid / center of mass**, in-place **translate / rotate**, **DSSP** secondary-structure assignment (8-state and 3-state, no external binary), **SASA** (Shrake-Rupley, no FreeSASA dependency), and **backbone dihedrals** (φ, ψ, ω, Ramachandran).
- **Validation utilities** — `molforge.validation` orchestrates the "score designs across multiple validators and combine results" pattern. Declarative `Criterion` (composable with `&` / `|` / `~`), `CriteriaSet` for per-criterion diagnostics, `cross_validate` to run designs through one or more validators, `consensus` to merge verdict lists ("ESMFold AND AlphaFold both pass" / "majority of validators pass" / threshold rules).
- **Evaluation metrics** — `molforge.metrics` ships the standard structural-prediction metrics: **TM-score** (Zhang & Skolnick), **GDT-TS / GDT-HA** (CASP), **lDDT** (alignment-free, what AlphaFold's pLDDT estimates), and **DockQ** (Basu & Wallner, for protein-protein complexes). NumPy-only — no tmalign/lddt binaries required.
- **ML featurization** — sequence featurizers (one-hot, BLOSUM/PAM, positional encoding), structure featurizers (RBF-binned distances, pair orientations, local environment), **ESM-2 protein language model embeddings**, and graph construction (`to_graph` → PyTorch Geometric / DGL).
- **Four engine-wrapper categories live end-to-end** — folding (**ESMFold** + **AlphaFold/ColabFold**), docking (**AutoDock Vina** with automatic meeko/RDKit prep), MD (**OpenMM** with full `prepare → minimize → run` flow), and now **generative design** (**RFdiffusion** for backbone generation, **ProteinMPNN** for sequence design). The full *de novo* design loop is in one library.

Coming next: DiffDock wrapper, GROMACS MD wrapper, explicit-solvent prep helpers, ML featurization for downstream models. See [`CHANGELOG.md`](CHANGELOG.md) for the full picture.

## Acknowledgements

molforge is inspired by [Protkit](https://github.com/silicogenesis/protkit) (SilicoGenesis), which pioneered the idea of a unified, hierarchical representation for protein structures in Python. molforge extends that direction toward cross-tool, cross-format workflows and a different internal architecture (NumPy-backed linear store, hierarchical views as accessors). See [`ACKNOWLEDGEMENTS.md`](ACKNOWLEDGEMENTS.md) for the longer list of projects we've learned from.

## Contributing

We welcome contributions. See [`CONTRIBUTING.md`](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md) before opening an issue or PR.

## License

MIT — see [`LICENSE`](LICENSE).

## Citation

If you use `molforge` in academic work, please cite us (BibTeX coming with the first tagged release).
