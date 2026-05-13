# molforge

[![CI](https://github.com/DoctorDean/molforge/actions/workflows/ci.yml/badge.svg)](https://github.com/DoctorDean/molforge/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/molforge.svg)](https://pypi.org/project/molforge/)
[![Python versions](https://img.shields.io/pypi/pyversions/molforge.svg)](https://pypi.org/project/molforge/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> **A unified, composable Python library for structural bioinformatics, molecular dynamics, protein engineering, and machine learning.**

`molforge` is an open-source library — **not a framework** — designed for researchers and engineers working on proteins. Import only what you need. Compose modules freely. Plug in your favorite folding, docking, or MD engine through a clean wrapper interface.

---

## Why molforge?

The protein/structural-bio Python ecosystem is fragmented: Biopython, MDAnalysis, RDKit, OpenMM, PyMOL, BioPandas, ProDy, ESM, OpenFold, and dozens of model-specific repos all use their own data representations. Moving a structure between two libraries usually means lossy conversion, re-parsing PDB files, or writing glue code.

`molforge` aims to fix this with **one principled, hierarchical data model** for protein structures — Protein → Chain → Residue → Atom — paired with first-class support for sequences, MD trajectories, and ML-ready tensor views. Everything else (folding, docking, simulation, scoring) is a thin wrapper around that shared representation.

## Design principles

1. **Library, not framework.** No runtime, no orchestration, no required entry point. Just import what you need.
2. **One data model, many views.** Hierarchical (`protein.chains[0].residues[12].atoms`) and linear (`protein.atom_array`, `protein.sequence`) views of the same data, kept in sync.
3. **Wrappers, not reimplementations.** We do not reimplement OpenMM, AutoDock, or AlphaFold. We provide consistent, typed interfaces around them.
4. **Plugin-friendly.** A registry pattern lets third parties ship docking engines, folding models, or scoring functions as separate packages that integrate seamlessly.
5. **Typed, tested, and documented.** Full type hints, >90% coverage target, MkDocs-based reference docs, runnable notebooks.

## Installation

```bash
# minimal core (data model + IO + sequence)
pip install molforge

# with structure analysis extras
pip install "molforge[structure]"

# with ML wrappers (torch, transformers, esm)
pip install "molforge[ml]"

# with MD support (openmm, mdtraj)
pip install "molforge[md]"

# everything
pip install "molforge[all]"

# development
git clone https://github.com/DoctorDean/molforge.git
cd molforge
pip install -e ".[dev,all]"
```

## Quickstart

```python
import molforge as bc

# Load a protein from PDB, mmCIF, or fetch from RCSB
protein = bc.load("1ubq.pdb")

# Hierarchical access
chain_a = protein.chains["A"]
residue_42 = chain_a.residues[42]
ca_atom = residue_42.atoms["CA"]

# Linear / tensor views (zero-copy where possible)
coords = protein.atom_array.coords          # (N_atoms, 3) numpy array
sequence = protein.sequence                  # str, one-letter code
chain_ids = protein.atom_array.chain_id      # (N_atoms,) array

# Compose with engines via wrappers
from molforge.wrappers.folding import ESMFold
from molforge.wrappers.docking import Vina

folded = ESMFold().predict("MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG")
poses  = Vina().dock(receptor=protein, ligand="ligand.sdf")
```

## Repository structure

```
molforge/
├── src/molforge/              # Library source (src-layout)
│   ├── core/                 # Hierarchical data model (Protein, Chain, Residue, Atom)
│   ├── sequence/             # Sequence operations, alignment, mutations
│   ├── structure/            # Structural analysis (RMSD, SASA, contacts, geometry)
│   ├── md/                   # MD trajectory I/O and analysis
│   ├── docking/              # Docking abstractions and pose handling
│   ├── ml/                   # ML utilities, featurizers, tensor views
│   ├── io/                   # Parsers and writers (PDB, mmCIF, FASTA, MOL2, SDF, ...)
│   ├── plugins/              # Plugin registry and discovery
│   ├── metrics/              # Scoring, evaluation, benchmarking metrics
│   └── wrappers/             # Thin interfaces to external engines
│       ├── folding/          # AlphaFold, ESMFold, Boltz, ...
│       ├── docking/          # AutoDock Vina, DiffDock, ...
│       └── md/               # OpenMM, GROMACS, ...
├── tests/                    # pytest test suite (unit + integration)
├── docs/                     # MkDocs / Sphinx documentation source
├── notebooks/                # Walkthrough + example notebooks
├── plugins/                  # Reference / example external plugins
├── scripts/                  # Maintenance and release scripts
├── .github/                  # CI workflows, issue/PR templates, CODEOWNERS
├── pyproject.toml            # Build config, deps, tool config (PEP 621)
├── requirements/             # Pinned requirement files per extra
├── CHANGELOG.md              # Keep-a-Changelog format
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
└── LICENSE
```

A more detailed walkthrough of the architecture lives in [`docs/architecture/overview.md`](docs/architecture/overview.md).

## Documentation

- **Tutorials & walkthroughs:** [`notebooks/walkthroughs/`](notebooks/walkthroughs/)
- **API reference:** [molforge.readthedocs.io](https://molforge.readthedocs.io) (coming soon)
- **Architecture:** [`docs/architecture/`](docs/architecture/)

## Contributing

We welcome contributions! Please read [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md) before opening an issue or PR.

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use `biocore` in academic work, please cite us (BibTeX coming soon).
