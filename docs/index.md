# molforge

**A unified Python library for structural bioinformatics, MD, protein
engineering, and ML — without the format-conversion tax.**

`molforge` is the connective tissue between the tools you already use.
Bring your structures and sequences in, plug in your engines of choice
(Vina, OpenMM, ESMFold, AlphaFold, RFdiffusion, ProteinMPNN, or your
own model), and walk out with a coherent pipeline instead of five
incompatible Python environments and a graveyard of conversion
scripts.

It is a *library*, not a framework: there is no orchestrator, no DAG
runtime, no decorators you have to import to make things work. Use
whatever workflow tool you like — Snakemake, Nextflow, Prefect, a
shell script — `molforge` is just imports.

[Install →](getting-started/installation.md){ .md-button .md-button--primary }
[Quickstart →](getting-started/quickstart.md){ .md-button }
[API reference →](reference/core.md){ .md-button }

---

## What's in the box

| Subpackage          | What it does                                                                                              |
| ------------------- | --------------------------------------------------------------------------------------------------------- |
| `molforge.core`     | Canonical data model: `Protein`, `Chain`, `Residue`, `Atom`, `AtomArray`.                                 |
| `molforge.io`       | Structure I/O (PDB, mmCIF, FASTA, SDF, MOL2, PDBQT, PQR) and trajectory I/O (XTC, TRR, DCD, NetCDF, HDF5). |
| `molforge.sequence` | Alignment, mutations, composition, substitution matrices.                                                 |
| `molforge.structure`| RMSD, SASA, contacts, DSSP, dihedrals, superposition.                                                     |
| `molforge.ml`       | Sequence/structure featurization, graph construction, ESM-2 embeddings.                                   |
| `molforge.metrics`  | TM-score, GDT-TS/HA, lDDT, DockQ.                                                                         |
| `molforge.validation`| Composable acceptance criteria for protein design candidates.                                            |
| `molforge.md`       | `Trajectory` and `Simulation` containers, plus the `MDEngine` interface for engine wrappers.              |
| `molforge.prep`     | MD system preparation: heterogen removal, missing-atom completion, ACE/NME capping, pH-aware protonation. |
| `molforge.docking`  | Pose handling and engine-agnostic docking abstractions.                                                   |
| `molforge.plugins`  | Entry-point discovery for third-party engines, parsers, and scorers.                                      |
| `molforge.wrappers` | Thin wrappers around external engines (folding, docking, MD, generative).                                 |

## Design principles

1. **Workflows over silos.** Every design decision is judged by
   *"does this make it easier to chain N tools together?"*
2. **Wrappers, not reimplementations.** We don't rebuild OpenMM or
   AutoDock. We give them a shared vocabulary.
3. **One data model, two views.** Hierarchical
   (`protein.chains["A"].residues[42]`) for biology, linear
   (`protein.atom_array.coords`) for ML — same data, no conversion.
4. **Heterogeneous content is first-class.** Antibodies have glycans.
   Drug targets have ligands and ions. Membrane proteins have lipids.
   The data model handles all of it without an awkward special case
   for *"non-protein."*
5. **Typed, tested, documented.** Strict mypy, ruff-clean, 1,100+ tests
   in CI, every public symbol has a Google-style docstring.

## Where to go next

- **New here?** Start with [Installation](getting-started/installation.md)
  and the [Quickstart](getting-started/quickstart.md).
- **Want to understand the design?** Read the
  [Architecture overview](architecture/overview.md).
- **Looking for a specific function?** Browse the
  [API reference](reference/core.md) or use the search box (top right).
- **Want to see real workflows?** The
  [walkthrough notebooks](https://github.com/DoctorDean/molforge/tree/main/notebooks/walkthroughs)
  cover each subpackage end-to-end.

## License & contributing

`molforge` is MIT-licensed. Issues and pull requests are welcome at
[github.com/DoctorDean/molforge](https://github.com/DoctorDean/molforge);
see [CONTRIBUTING.md](https://github.com/DoctorDean/molforge/blob/main/CONTRIBUTING.md)
for the workflow.
