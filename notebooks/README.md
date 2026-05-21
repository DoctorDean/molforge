# Notebooks

Runnable walkthroughs and examples.

## Walkthroughs (`walkthroughs/`)

Short, didactic notebooks focused on a single capability area:

| Notebook | Subpackage | Status |
|---|---|---|
| [`01_sequences.ipynb`](walkthroughs/01_sequences.ipynb) | `molforge.sequence` | ✅ Live |
| [`02_structures.ipynb`](walkthroughs/02_structures.ipynb) | `molforge.structure` | ✅ Live |
| [`03_md_simulations.ipynb`](walkthroughs/03_md_simulations.ipynb) | `molforge.wrappers.md` | ✅ Live |
| [`04_docking.ipynb`](walkthroughs/04_docking.ipynb) | `molforge.wrappers.docking` | ✅ Live |
| [`05_ml_featurization.ipynb`](walkthroughs/05_ml_featurization.ipynb) | `molforge.ml` | ✅ Live |
| [`06_plugin_authoring.ipynb`](walkthroughs/06_plugin_authoring.ipynb) | `molforge.plugins` | ✅ Live |

## Examples (`examples/`)

Longer real-world examples combining multiple subpackages:

| Notebook | What it shows |
|---|---|
| [`end_to_end_design.ipynb`](examples/end_to_end_design.ipynb) | Full protein-design loop: sequence → ESMFold → DSSP/Rg analysis → mutation → re-fold → per-residue RMSD + DSSP-diff comparison. |
| [`de_novo_design.ipynb`](examples/de_novo_design.ipynb) | Full *de novo* design loop: generate backbone with RFdiffusion → design sequences with ProteinMPNN → fold with ESMFold → score with TM-score/lDDT/RMSD → filter to successful designs. |
| [`cross_engine_validation.ipynb`](examples/cross_engine_validation.ipynb) | Cross-engine consensus pattern: scoring candidate designs against two folding validators (ESMFold + AlphaFold), combining results with `consensus(mode="all" / "any" / "majority")`, and inspecting which validator disagreed for borderline designs. |

## Notebook conventions

All notebooks should:

- Run top-to-bottom on a fresh kernel.
- Print the molforge version at the top (`import molforge; print(molforge.__version__)`).
- Cells that require heavy dependencies (`torch`, `vina`, `openmm`, `colabfold`) are marked with `# 🐢 SLOW` and include pre-baked outputs so the notebook renders on GitHub without forcing those installs.
- Be small enough to render on GitHub. For longer notebooks with heavy outputs, clear them before committing.

If you're adding a new notebook, run it once locally and commit with executed outputs so readers can follow along without setup.
