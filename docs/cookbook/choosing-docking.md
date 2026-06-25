# Choosing a docking engine

molforge wraps two docking engines that take different approaches to
the same problem: given a receptor and a ligand, where does the
ligand bind?

## Side-by-side

| Engine    | Method                              | Manual binding site? | Speed (1 ligand)        | When to pick it                                                          |
| --------- | ----------------------------------- | -------------------- | ----------------------- | ------------------------------------------------------------------------ |
| **Vina**  | Force-field scoring + search        | Yes (box centre + size) | Seconds                 | Known binding site; virtual screening; reproducible scores.             |
| **DiffDock** | Diffusion-model pose prediction  | No                   | Minutes (GPU)           | Unknown binding site; ML-driven pose proposal; novel ligand classes.    |

These differ in two fundamental ways: **how they search**, and
**how much you need to know about the site upfront**.

## How to choose

### You know the binding site

**Use Vina.** A defined box + an empirical scoring function is the
right tool for a well-defined search. Vina's strengths:

- **Fast** — seconds per ligand on a single core, scales easily to
  millions of compounds.
- **Reproducible** — same seed, same pose, same score. Essential
  for ranking compounds.
- **Interpretable scoring** — the score is an estimate of binding
  free energy (kcal/mol). The absolute number isn't trustworthy,
  but the rank ordering across a series is meaningful.

Vina's weaknesses:

- **You must specify the box.** Wrong centre = nothing useful.
- **Rigid receptor.** Vina samples ligand conformations but not
  receptor motions (with caveats around `--exhaustiveness` and
  some flexibility options).
- **Empirical scoring function.** Calibrated on PDBbind-class
  data; out-of-distribution ligand classes (large macrocycles,
  covalent binders, metal-coordinating compounds) get unreliable
  scores.

### You don't know the binding site

**Use DiffDock** — it predicts the pose without needing a box.
Trained on the PDB, the model learned what binding poses look like
without explicit thermodynamics.

- **No site selection needed.** Hand it a receptor and a ligand;
  it proposes poses anywhere on the surface.
- **ML scoring is closer to "how confident is the model"** than to
  "what's the binding free energy." Better for ranking poses for
  a given ligand than for comparing different ligands.
- **Slower.** Seconds-to-minutes per ligand, plus GPU memory
  pressure for large receptors.
- **Distribution sensitivity.** DiffDock works best for ligand
  classes well-represented in its training data.

### You're virtual-screening millions of compounds

**Use Vina.** DiffDock is too slow at this scale. Vina against a
pre-prepared receptor with `exhaustiveness=8` runs hundreds of
ligands per CPU core per hour; scale across cores and you can do
millions overnight.

### You want pose proposals to inspect by hand

**Use DiffDock.** Its ML scoring + uncertainty are more meaningful
than Vina's per-ligand absolute scores when you're triaging top
hits with a structural biologist's eye.

### You want both

A reasonable two-stage workflow: DiffDock for pocket discovery,
Vina for tight scoring. molforge makes this straightforward — both
engines return the same `DockingResult` shape:

```python
from molforge.wrappers.docking import DiffDock, Vina

# Stage 1: where does it bind?
explore = DiffDock().dock(receptor=receptor, ligand=ligand)
discovered_site = explore.poses[0].ligand.atom_array.coords.mean(axis=0)

# Stage 2: score precisely at the discovered site.
score = Vina(seed=42).dock(
    receptor=receptor,
    ligand=ligand,
    center=tuple(discovered_site.tolist()),
    box_size=(20.0, 20.0, 20.0),
    exhaustiveness=32,
)
```

## Common dimensions

### Input formats

Both accept the same range:

- Receptor: `Protein` object, PDB / mmCIF path, or pre-prepared
  PDBQT.
- Ligand: SDF / MOL / MOL2 / PDB / PDBQT path, or a SMILES string
  (Vina's `dock` accepts SMILES directly; for DiffDock, prepare
  the ligand first).

### Installation footprint

| Engine    | Install                                                            |
| --------- | ------------------------------------------------------------------ |
| Vina      | `pip install "molforge[docking]"` + `pip install vina` + Open Babel installed system-wide. |
| DiffDock  | Manual clone + install of gcorso/DiffDock repo. `DIFFDOCK_HOME` env var. GPU strongly recommended. |

Vina is by far the easier install — pip + a system package.
DiffDock needs the upstream repository present.

### Reproducibility

Vina with a fixed `seed=N` produces bit-identical output across
runs on the same hardware. DiffDock has multiple sources of non-
determinism (PyTorch RNG, CUDA non-determinism) and is harder to
pin to byte-identical output even with seeds set.

For *workflow* reproducibility — same engine, same parameters, same
inputs — both engines write a `Provenance` to the `DockingResult`
that lets downstream code verify the inputs match. See
[Inspect provenance](inspect-provenance.md).

### Receptor preparation

Vina needs the receptor in PDBQT format. molforge handles this
automatically: pass a `Protein` or a `.pdb` and Vina prepares it
with meeko. If you have lots of ligands to dock against the same
receptor, prepare the receptor once and pass the `.pdbqt` path
directly to skip re-preparation.

DiffDock has its own preparation pipeline inside the upstream repo;
molforge just shells out to it.

## What molforge doesn't wrap (yet)

- **GNINA** (CNN-rescored Vina) — on the roadmap.
- **Glide, GOLD, Surflex** — commercial; no plans.
- **AutoDock GPU** — the GPU-accelerated descendant of Vina;
  considered for the roadmap.

For an engine that's not yet wrapped, write a plugin and call it
through the standard `DockingEngine` interface — see
[Plugin authoring](../guide/plugins.md).
