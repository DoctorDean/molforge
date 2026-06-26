# Choosing a docking engine

molforge wraps three docking engines that take different approaches
to the same problem: given a receptor and a ligand, where does the
ligand bind?

## Side-by-side

| Engine    | Method                              | Manual binding site? | Speed (1 ligand)        | When to pick it                                                          |
| --------- | ----------------------------------- | -------------------- | ----------------------- | ------------------------------------------------------------------------ |
| **Vina**  | Force-field scoring + search        | Yes (box centre + size) | Seconds                 | Known binding site; virtual screening; reproducible scores.             |
| **Gnina** | Vina search + CNN rescoring         | Yes (box centre + size) | Tens of seconds         | Known binding site, where CNN scoring beats Vina's empirical function. |
| **DiffDock** | Diffusion-model pose prediction  | No                   | Minutes (GPU)           | Unknown binding site; ML-driven pose proposal; novel ligand classes.    |

These differ in two fundamental ways: **how they search**, and
**how they score**. Vina and Gnina share the same Monte-Carlo
search (gnina is a fork of Vina) but differ in scoring — Vina uses
an empirical function, Gnina rescores with a CNN. DiffDock does
the search itself with a diffusion model.

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

### You know the binding site but want better scoring than Vina

**Use Gnina.** Same Monte-Carlo search as Vina (it's a Vina fork),
but rescores poses with a 3D CNN trained on PDBbind. The CNN
typically ranks poses more accurately than Vina's empirical
scoring function — gnina's own benchmarks show Top-1 redocking
accuracy going from ~58% (Vina) to ~73% (gnina). The cost is
tens-of-seconds per call rather than seconds: the CNN forward
pass is the bottleneck.

Gnina returns three scores per pose:

- `vina_affinity` — the Vina-style empirical energy (kcal/mol).
- `cnn_score` — the learned pose-quality score (0–1, higher is
  better). This is the default ranking criterion.
- `cnn_affinity` — the learned binding affinity (pK units).

```python
from molforge.wrappers.docking import Gnina

result = Gnina(seed=42).dock(
    receptor=receptor,
    ligand=ligand,
    center=(10.0, 5.0, -2.0),
    box_size=(20.0, 20.0, 20.0),
)
# Default: poses sorted by CNN score, best first.
top = result.poses[0]
print(f"CNN score: {top.score:.2f}    "
      f"CNN affinity: {top.metadata['cnn_affinity']:.2f}    "
      f"Vina: {top.metadata['vina_affinity']:.2f}")
```

For ranking compounds in a virtual screen, you can override
`sort_order="CNNaffinity"` to rank by predicted binding strength
rather than pose quality:

```python
result = Gnina(sort_order="CNNaffinity").dock(...)
```

Gnina's reproducibility is the same as Vina's — same seed produces
the same poses (the CNN scoring is deterministic given fixed
inputs). For virtual screening where Vina-only speed matters more
than CNN accuracy, pass `cnn_scoring="none"` to make Gnina behave
like smina (Vina with extra command-line ergonomics).

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

### A different pre-step: detect pockets, then dock

If you don't know the binding site *and* want Vina's reproducible
scoring rather than DiffDock's ML-driven search, molforge also wraps
**fpocket**, a fast Voronoi-based pocket detector. It runs in seconds
and gives you ranked pocket candidates with druggability scores; pick
the top one and pass its centre to Vina:

```python
from molforge.wrappers.pockets import detect_pockets
from molforge.wrappers.docking import Vina

pockets = detect_pockets(receptor)         # ranked best-first
result = Vina(seed=42).dock(
    receptor=receptor,
    ligand=ligand,
    center=tuple(pockets[0].center.tolist()),
    box_size=(20.0, 20.0, 20.0),
)
```

fpocket is much cheaper than DiffDock (CPU-only, seconds) but its
ranking is heuristic rather than learned — for tough cases (allosteric
sites, cryptic pockets) DiffDock's exploration is usually better. See
the [folding-then-docking](folding-then-docking.md) recipe for more
on pocket-centre selection.

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
| Gnina     | Not pip-installable. `brew install gnina` on macOS, download a release binary from `github.com/gnina/gnina/releases`, or build from source. The binary bundles all CNN models. |
| DiffDock  | Manual clone + install of gcorso/DiffDock repo. `DIFFDOCK_HOME` env var. GPU strongly recommended. |

Vina is the easiest install — pip + a system package. Gnina is a
single binary download. DiffDock needs the upstream repository
present.

### Reproducibility

Vina and Gnina with a fixed `seed=N` produce bit-identical output
across runs on the same hardware. (Gnina's CNN scoring is
deterministic given fixed inputs, so the seed only affects the
Vina-style Monte-Carlo search.) DiffDock has multiple sources of
non-determinism (PyTorch RNG, CUDA non-determinism) and is harder
to pin to byte-identical output even with seeds set.

For *workflow* reproducibility — same engine, same parameters, same
inputs — all three engines write a `Provenance` to the
`DockingResult` that lets downstream code verify the inputs match.
See [Inspect provenance](inspect-provenance.md).

### Receptor preparation

Vina needs the receptor in PDBQT format; molforge handles this
automatically via meeko. If you have lots of ligands to dock against
the same receptor, prepare the receptor once and pass the `.pdbqt`
path directly to skip re-preparation.

Gnina handles format conversion internally via Open Babel — pass a
`Protein`, a `.pdb`, a `.pdbqt`, or a `.mol2` and it figures things
out. No molforge-side preparation needed.

DiffDock has its own preparation pipeline inside the upstream repo;
molforge just shells out to it.

## What molforge doesn't wrap (yet)

- **Smina** — Vina's direct ancestor. Gnina with `cnn_scoring="none"`
  is effectively smina, so a dedicated wrapper isn't urgent.
- **Glide, GOLD, Surflex** — commercial; no plans.
- **AutoDock GPU** — the GPU-accelerated descendant of Vina;
  considered for the roadmap.

For an engine that's not yet wrapped, write a plugin and call it
through the standard `DockingEngine` interface — see
[Plugin authoring](../guide/plugins.md).
