# molforge

[![CI](https://github.com/DoctorDean/molforge/actions/workflows/ci.yml/badge.svg)](https://github.com/DoctorDean/molforge/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/molforge.svg)](https://pypi.org/project/molforge/)
[![Python versions](https://img.shields.io/pypi/pyversions/molforge.svg)](https://pypi.org/project/molforge/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> **A forge for protein workflows.** One Python script, every tool in your stack: docking, MD, folding, antibody and nanobody engineering, de novo design — without the format-conversion tax.

`molforge` is an open-source Python library that lets you compose protein workflows across the tools you already use. Bring your structures and sequences in, plug in your engines of choice (Vina, OpenMM, ESMFold, AlphaFold, RFdiffusion, ProteinMPNN, your own model), and walk out with a coherent pipeline instead of five incompatible Python environments and a graveyard of conversion scripts.

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

# 3. Dock a ligand against it
result = Vina().dock(receptor=folded, ligand="ligand.sdf")
top_pose = result.poses[0]

# 4. Drop into MD for relaxation
trajectory = OpenMM().simulate(top_pose.complex, steps=10_000)

# 5. Inspect — hierarchical or linear, your call
print(folded.sequence)                            # one-letter per chain
print(folded.atom_array.coords.shape)             # (N, 3) NumPy array
ca = folded.chains["A"].residues[42].atoms["CA"]  # specific atom
```

Notice what *isn't* there: file-format conversions, atom-name remapping, hand-rolled PDB parsers, custom data classes per engine. molforge does that work so your script reads like the science you're actually doing.

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
├── tests/                    # pytest suite (unit + integration)
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
- **Structural analysis** — Kabsch/Umeyama **superposition**, **RMSD** (whole-structure and per-residue, multiple atom subsets), **contact and distance maps**, **radius of gyration**, **centroid / center of mass**, in-place **translate / rotate**.
- **First engine wrapper** — **ESMFold** end-to-end from sequence string to `Protein` (`pip install 'molforge[ml]'`); the wrapper pattern is now proven and the other folding/docking/MD engines can follow the same template.

Coming next: SASA, DSSP, sequence alignment, additional engine wrappers (AlphaFold, Vina, OpenMM). See [`CHANGELOG.md`](CHANGELOG.md) for the full picture.

## Acknowledgements

molforge is inspired by [Protkit](https://github.com/silicogenesis/protkit) (SilicoGenesis), which pioneered the idea of a unified, hierarchical representation for protein structures in Python. molforge extends that direction toward cross-tool, cross-format workflows and a different internal architecture (NumPy-backed linear store, hierarchical views as accessors). See [`ACKNOWLEDGEMENTS.md`](ACKNOWLEDGEMENTS.md) for the longer list of projects we've learned from.

## Contributing

We welcome contributions. See [`CONTRIBUTING.md`](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md) before opening an issue or PR.

## License

MIT — see [`LICENSE`](LICENSE).

## Citation

If you use `molforge` in academic work, please cite us (BibTeX coming with the first tagged release).
