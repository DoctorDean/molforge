# MD and RMSD

You have a prepared protein and want to run a short MD simulation
and look at how much it moves. This recipe runs OpenMM end-to-end —
prepare → minimise → run — and computes RMSD against the starting
structure across the trajectory.

This is a "hello world" simulation — 10 picoseconds, just enough to
see equilibration. Production simulations are usually nanoseconds to
microseconds long; the API is the same.

## Requirements

```bash
pip install "molforge[md,prep]"   # openmm + pdbfixer + mdtraj
```

## The recipe

```python
from molforge.io import fetch
from molforge.prep import prepare_for_md
from molforge.wrappers.md import OpenMM
from molforge.structure import rmsd

# 1. Get a starting structure and prepare it.
protein = fetch("1AKE")
ready = prepare_for_md(protein)

# 2. Build an OpenMM Simulation.
engine = OpenMM(platform="CPU")          # "CUDA" if you have a GPU
sim = engine.prepare(
    ready,
    force_field="amber14-all",           # AMBER ff14SB + TIP3P
    temperature=300.0,                   # K
    timestep=0.002,                      # ps (2 fs)
)

# 3. Minimise the energy. Removes clashes from the prepared structure.
sim = engine.minimize(sim, max_iterations=100)

# 4. Run the simulation. With timestep=0.002 ps, 5,000 steps = 10 ps.
trajectory = engine.run(sim, n_steps=5_000, save_every=100)
print(f"Recorded {trajectory.n_frames} frames over "
      f"{trajectory.times[-1]:.1f} ps")

# 5. RMSD against the starting structure, per frame.
import numpy as np

reference = ready                         # the energy-minimised starting state
rmsds = np.empty(trajectory.n_frames)
for i in range(trajectory.n_frames):
    rmsds[i] = rmsd(trajectory.frame(i), reference, subset="ca", align=True)

print(f"RMSD range: {rmsds.min():.2f} – {rmsds.max():.2f} Å")
print(f"Final RMSD (CA, aligned): {rmsds[-1]:.2f} Å")
```

## What's happening

The four-step pipeline mirrors how every MD engine works:

| Step          | What it does                                                              |
| ------------- | ------------------------------------------------------------------------- |
| `prepare`     | Builds a `Simulation` — topology, system, force field, integrator.        |
| `minimize`    | Steepest-descent energy minimisation; squeezes out clashes.               |
| `run`         | Advances the integrator `n_steps` steps, recording every `save_every`th.  |
| Analysis      | The returned `Trajectory` is just NumPy arrays — work with it directly.   |

`OpenMM(...)`'s default platform picks CUDA if available, else CPU.
Explicitly pass `platform="CUDA"` if you want to fail loudly when no
GPU is available.

## What's in the Trajectory

A [`Trajectory`](../reference/md.md) is a thin dataclass:

```python
trajectory.coordinates    # (n_frames, n_atoms, 3) float32, in Å
trajectory.times          # (n_frames,) float, in ps
trajectory.energies       # (n_frames,) float, potential energy in kJ/mol
trajectory.topology       # the Protein this trajectory belongs to
trajectory.metadata       # engine + run config + Provenance
```

It's not magic — once you have it, just slice the NumPy arrays.
Average coordinate position of CA atoms over the last 5 frames:

```python
ca_mask = (trajectory.topology.atom_array.atom_name == "CA")
late_frames_ca = trajectory.coordinates[-5:, ca_mask, :]
mean_pos = late_frames_ca.mean(axis=0)
```

## Saving the trajectory to disk

Trajectories are big. Save them in a compact binary format:

```python
from molforge.io import write_trajectory
write_trajectory("run.xtc", trajectory)             # GROMACS XTC, lossy
write_trajectory("run.dcd", trajectory)             # CHARMM DCD
write_trajectory("run.h5",  trajectory)             # HDF5, carries topology
```

XTC is the common choice — small files, slight precision loss
(quantised to 0.001 nm). DCD is a hair larger but lossless. Reload
later:

```python
from molforge.io import read_trajectory
traj = read_trajectory("run.xtc", topology=ready)
```

## Production-scale runs

For real simulations the same recipe scales up by changing two
numbers:

```python
trajectory = engine.run(
    sim,
    n_steps=50_000_000,    # 100 ns at 2 fs/step
    save_every=5_000,      # 10,000 frames; one every 10 ps
)
```

For nanosecond-scale runs you want GPU (`platform="CUDA"`) and
typically a HDF5 or DCD writer that streams to disk rather than
holding all frames in memory. The current implementation holds the
full trajectory in memory; expect that to change as longer
simulations become a first-class workflow.

## When to pick GROMACS instead

OpenMM is the right default — Python-native, simple to drive,
faster than GROMACS for many setups on modern GPUs. Switch to
GROMACS when:

- You need a feature OpenMM doesn't have (specific enhanced-
  sampling methods, complex topology manipulation, certain
  membrane setups).
- You're running on a cluster where GROMACS is the established
  workflow (job scripts, restart files, etc.).

The molforge interface is the same — `GROMACS(...).prepare().minimize().run()`
returns the same `Trajectory` shape. See the
[Engine wrappers guide](../guide/wrappers.md).

## RMSD subset choices

The recipe uses `subset="ca"` — alpha-carbon only — which is the
standard for protein-conformational-change analyses. Other choices:

- `subset="backbone"` (N, CA, C): more atoms, slightly tighter
  fits, similar story.
- `subset="all_heavy"`: every non-hydrogen atom. Sensitive to
  side-chain motions too — useful for binding-site analyses, noisy
  for overall fold.
- `subset="all"`: includes hydrogens, almost never what you want
  for cross-trajectory comparison.

For RMSF (per-residue fluctuations) and other geometric analyses
across frames, see [`molforge.structure`](../reference/structure.md).
