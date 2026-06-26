# Fold then dock

You have a sequence and a small-molecule ligand (a SMILES string or
an SDF file) and want to predict where the ligand binds. This recipe
folds the receptor with ESMFold, then docks the ligand against it
with AutoDock Vina.

## Requirements

```bash
pip install "molforge[ml,docking]"
# Plus the vina Python package and Open Babel for ligand prep.
# On macOS:  brew install open-babel  &&  pip install vina
# On Linux:  apt install openbabel    &&  pip install vina
```

If you already have a receptor structure (a PDB file, an RCSB ID),
skip the fold step and pass the file/ID to `dock` directly.

## The recipe

```python
from molforge.wrappers.folding import ESMFold
from molforge.wrappers.docking import Vina

receptor_sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK..."
ligand_smiles     = "CC(=O)OC1=CC=CC=C1C(=O)O"   # aspirin

# 1. Fold the receptor.
receptor = ESMFold().predict(receptor_sequence)

# 2. Pick a docking site. For a real workflow you'd use a known
#    binding-site residue, a pocket detector, or the centre of mass
#    of a co-crystallised ligand. Here we use the receptor's centroid
#    as a placeholder — replace with your actual site.
import numpy as np
center = tuple(np.asarray(receptor.atom_array.coords).mean(axis=0).tolist())

# 3. Dock.
result = Vina().dock(
    receptor=receptor,
    ligand=ligand_smiles,           # SMILES string; meeko prepares it
    center=center,
    box_size=(20.0, 20.0, 20.0),    # angstroms; widen for larger sites
    exhaustiveness=8,               # 8 is the Vina default; 32 for production
    n_poses=9,
)

# `result.poses` is sorted best-first (lowest binding energy first).
top = result.poses[0]
print(f"Top pose: score = {top.score:.2f} kcal/mol")
print(f"Generated {len(result.poses)} poses")

# Save the receptor + top pose for visualisation.
from molforge.io import save
save(receptor, "receptor.pdb")
save(top.ligand, "ligand_top_pose.sdf")
```

## What you get back

A [`DockingResult`](../reference/docking.md) with:

- `result.poses` — a list of `Pose` objects sorted best-first by
  Vina's binding-energy score (more negative = better).
- `result.receptor` — the receptor used for docking (the same
  `Protein` you passed in, or one parsed from the path).
- `result.metadata["provenance"]` — a `Provenance` whose `parent` is
  the receptor's `Provenance` from ESMFold. Calling
  `result.metadata["provenance"].chain()` yields the full sequence-to-
  pose history: see [Inspect provenance](inspect-provenance.md).

Each `Pose` has:

- `pose.ligand` — a `Protein` for the docked ligand conformation.
- `pose.score` — Vina's binding-energy estimate (kcal/mol).
- `pose.metadata` — Vina-specific extras (RMSD to top pose, etc.).

## Picking the docking site

Vina docks into a box; the box's `center` and `box_size` matter
enormously. Get the centre wrong and you'll dock to nothing.

Common ways to pick the centre:

```python
# (a) Around a specific residue (e.g. an active-site His):
his57 = next(r for r in receptor.iter_residues()
             if r.id == 57 and r.name == "HIS")
center = tuple(his57.atom_array.coords.mean(axis=0).tolist())

# (b) Around a co-crystallised ligand in a reference structure:
from molforge.io import fetch
ref = fetch("1ABC")
ligand_atoms = ref.atom_array.select(ref.atom_array.entity_type == "ligand")
center = tuple(ligand_atoms.coords.mean(axis=0).tolist())

# (c) The geometric centre of the whole protein (last resort,
#     usually wrong for anything but tiny proteins):
center = tuple(receptor.atom_array.coords.mean(axis=0).tolist())
```

For *systematic* pocket detection, molforge wraps **fpocket**:

```python
from molforge.wrappers.pockets import detect_pockets

pockets = detect_pockets(receptor)
print(f"Found {len(pockets)} pockets; top druggability "
      f"{pockets[0].druggability:.2f}")

# Use the top pocket's centre as the docking box centre.
result = Vina().dock(
    receptor=receptor,
    ligand=ligand_smiles,
    center=tuple(pockets[0].center.tolist()),
    box_size=(20.0, 20.0, 20.0),
)
```

`detect_pockets` returns a list of `Pocket` objects ranked by
fpocket's score. Each pocket has a `center` you can pass straight
to a docking engine's `center=` argument, plus `volume`, `score`,
`druggability` (fpocket's 0–1 drug-likeness estimate), and a list
of lining residues.

`fpocket` itself isn't pip-installable; install via your system
package manager (`brew install fpocket` on macOS,
`apt install fpocket` on Linux) or build from
`https://github.com/Discngine/fpocket`. P2Rank (an ML-based
alternative) isn't wrapped yet.

## Higher-confidence results

The recipe above uses Vina's defaults, which are fine for a first
pass but undersample for production work. For higher-confidence
results:

```python
result = Vina(seed=42).dock(
    ...,
    exhaustiveness=32,    # 4× the default; ~4× slower
    n_poses=20,           # generate more poses to choose from
    energy_range=4.0,     # keep poses within 4 kcal/mol of the best
)
```

`seed=42` makes the run deterministic — useful when comparing
multiple ligands or runs.

## When to pick a different docking engine

For SMILES + receptor → pose, Vina is the workhorse default. Switch
when:

- **You know the site and want CNN scoring** → Gnina. Same Vina
  search but each pose is rescored by a 3D CNN; typically more
  accurate ranking than Vina's empirical function, at ~10× the
  per-call latency.
- **You have a ligand and want fast, ML-based pose prediction
  without specifying a box** → DiffDock. Slower per-call but the
  search is learned, no manual site selection.
- **You're scoring many ligands against one site (virtual
  screening)** → Vina, with the receptor pre-prepared once.

See [Choosing a docking engine](choosing-docking.md) for the full
trade-off table.
