# Ensembles

`molforge.ensembles` provides weighted statistics over collections of
related structures — the most common case being **docked poses**: one
ligand against one receptor, returned by a docking engine as N
plausible binding modes ranked by score.

The seven public functions, organized by concern:

| Concern         | Function                       | Returns                              |
| --------------- | ------------------------------ | ------------------------------------ |
| Weighting       | [`boltzmann_weights`](../reference/ensembles.md) | `(n,)` weights summing to 1 |
| Weighting       | [`resample`](../reference/ensembles.md) | Bootstrap of pose objects |
| Geometry        | [`pairwise_rmsd`](../reference/ensembles.md) | `(n, n)` RMSD matrix         |
| Geometry        | [`pose_diversity`](../reference/ensembles.md) | Dict of summary stats        |
| Clustering      | [`pose_clusters`](../reference/ensembles.md) | `PoseClusteringResult`        |
| Spatial         | [`binding_site_density`](../reference/ensembles.md) | `DensityGrid`           |
| Consensus       | [`consensus_pose`](../reference/ensembles.md) | A single `Pose`               |

## The canonical pipeline

```python
from molforge.ensembles import (
    boltzmann_weights, pose_diversity, pose_clusters,
    binding_site_density, consensus_pose,
)

# 1. Got a DockingResult from somewhere.
from molforge.wrappers.docking import Vina
result = Vina().dock(receptor, ligand)
poses = result.poses

# 2. Weight by score (lower = better, Vina convention).
weights = boltzmann_weights(poses)

# 3. Did the docking actually explore? Quick diagnostic.
stats = pose_diversity(poses)
print(f"mean pairwise RMSD: {stats['mean']:.2f} Å")

# 4. How many distinct binding modes?
clusters = pose_clusters(poses, cutoff=2.0)
print(f"{clusters.n_clusters} distinct modes; biggest has {clusters.clusters[0].size}")

# 5. Where does the ligand spend its mass in space?
grid = binding_site_density(poses, weights=weights, spacing=0.5)

# 6. One representative pose for downstream use.
representative = consensus_pose(poses, weights=weights, method="medoid")
```

## Boltzmann weights

The fundamental operation: turn a vector of scores into a probability
distribution. For Vina-style scores in kcal/mol with the default
temperature ``kT = 0.593`` kcal/mol (room temperature):

```python
weights = boltzmann_weights([-9.5, -8.2, -7.1])
# array([0.83, 0.10, 0.07])  — approximately
```

The temperature parameter controls softness:

```python
boltzmann_weights(scores, temperature=10.0)   # softer, more uniform
boltzmann_weights(scores, temperature=0.1)    # sharper, winner-takes-all
```

For ML-derived scores where larger is better (DiffDock confidence,
EquiDock scores), pass ``lower_is_better=False``.

The function accepts either a numeric sequence, a NumPy array, or a
sequence of [`Pose`](../reference/docking.md) objects (in which case
the ``score`` attribute is read automatically).

## Pose clustering

Hierarchical average-linkage RMSD clustering. Two poses end up in the
same cluster if their average-linkage RMSD stays below ``cutoff``
(default 2.0 Å, a common docking-community value for "same binding
mode"):

```python
result = pose_clusters(poses, cutoff=2.0)
result.n_clusters             # int
result.labels                 # (n_poses,) int array
result.clusters               # list of PoseCluster, biggest first

biggest = result.clusters[0]
biggest.size                  # 7
biggest.medoid                # 3 (pose index)
biggest.mean_intra_rmsd       # 0.85
```

The clusterer is pure NumPy with no scipy dependency. The algorithm
is O(n³) which is fine for ensemble sizes from a single docking run
(typically n ≲ 20).

## Density maps

`binding_site_density` accumulates ligand heavy-atom positions across
the ensemble into a 3D spatial grid. Each pose contributes
``weight[i]`` per atom; the default ``spacing`` is 1.0 Å and the box
is auto-sized to cover all poses with 4 Å padding:

```python
grid = binding_site_density(poses, weights=weights, spacing=0.5)
grid.density.shape           # (nx, ny, nz)
grid.origin                  # (3,) — Cartesian corner in Å
grid.spacing                 # float

# Find the highest-occupied cell:
import numpy as np
hot = np.unravel_index(grid.density.argmax(), grid.density.shape)
hot_xyz = grid.coordinate_of(hot)
```

For comparative analysis (e.g. ensemble A vs ensemble B on the same
grid), pass explicit ``origin`` and ``shape``.

## Consensus poses

Two strategies:

- ``method="medoid"`` (default) — pick an actual pose from the
  ensemble that minimizes its weighted summed RMSD to the rest. The
  output is a real pose, guaranteed chemically valid.
- ``method="mean"`` — synthesize a new pose by averaging coordinates
  across the ensemble. Bond geometries are not preserved, so this is
  only meaningful when the input ensemble has already been
  tight-clustered to similar conformations.

```python
medoid = consensus_pose(poses, weights=weights)               # safe default
synth  = consensus_pose(cluster_members, method="mean")       # for tight clusters
```

## What v1 doesn't do

A few known limitations worth flagging:

- **Pose RMSD is order-sensitive.** Atom orderings must be consistent
  across poses, which is true for poses from a single docking run.
  For symmetric ligands (benzene, p-phenyls) the reported RMSD is an
  upper bound on the symmetry-aware RMSD. A future enhancement could
  use RDKit graph matching when the ``[docking]`` extra is installed.
- **Receptor is treated as fixed.** Ensemble functions assume one
  receptor conformation; poses differ only in ligand placement.
  Mixed-receptor ensembles (e.g. one ligand across MD frames of the
  receptor) need a different API.
- **No scipy dependency.** Clustering is hand-rolled in NumPy. For
  very large ensembles (n > 200, e.g. MD-derived) scipy's optimized
  linkage would be faster; that's a natural future extension.

## Reference

- [`molforge.ensembles`](../reference/ensembles.md) — full API.
