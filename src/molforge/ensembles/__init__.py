"""Ensemble analysis: weighted statistics over collections of structures.

An *ensemble* is what you have when one input gives you many outputs:

- A docking run returns many plausible poses.
- An MD trajectory returns many time-snapshots of one system.
- A generative model (RFdiffusion, ProteinMPNN sampling) returns many
  candidates.

The questions you typically ask of an ensemble are the same regardless
of which kind it is:

1. **How should I weight the members?** Boltzmann-weight by score,
   uniform-weight, or use external weights.
2. **How diverse is the ensemble?** Pairwise-distance statistics tell
   you whether the sampling actually explored conformational space.
3. **Which members cluster together?** Reduces N noisy samples to k
   representative groups.
4. **What does the population look like in space?** Density maps,
   contact frequencies.
5. **Can I draw a single representative member?** Medoid pick or
   weighted average.

Two ensemble surfaces live here:

- **Pose ensembles** (the original focus — docking output: the same
  ligand in N binding-mode candidates against one receptor). The
  weighting / clustering / density / consensus functions below.
- **Cross-engine structural ensembles** (:func:`cross_engine_fold`):
  one sequence folded by several engines (ESMFold / AlphaFold / Boltz /
  RoseTTAFold), superposed, with a pairwise TM / RMSD spread, a medoid
  consensus, and a per-residue map of where the engines disagree. This
  is the structural analogue of the pose surface — "how much do my
  methods agree?" rather than "how much did my poses explore?".

The pose functions are deliberately general: `boltzmann_weights` takes
raw scores so it works for any kind of score, and the clustering /
density functions take generic coordinate arrays alongside the
convenience overloads that take `Pose` objects. MD trajectories remain a
natural future extension of the same machinery.

What's here:

**Weighting**:
    - :func:`boltzmann_weights` — softmax weights from scores.
    - :func:`resample` — weighted bootstrap.

**Geometry**:
    - :func:`pairwise_rmsd` — N×N RMSD matrix for ligand heavy atoms.
    - :func:`pose_diversity` — summary statistics over pairwise RMSDs.

**Clustering**:
    - :func:`pose_clusters` — hierarchical RMSD clustering with medoids.

**Spatial**:
    - :func:`binding_site_density` — 3D histogram of ligand atom
      positions, optionally Boltzmann-weighted.

**Consensus**:
    - :func:`consensus_pose` — pick a representative (medoid or
      weighted-mean) pose.

Limitations of v1:

- **Pose RMSD is order-sensitive.** Atom orderings are assumed to be
  consistent across poses (which is true for poses returned from a
  single docking run by Vina, DiffDock, etc.). For symmetric ligands
  the resulting RMSD is an *upper bound* on the true symmetry-aware
  RMSD; for the common case of asymmetric drug-like ligands this is
  fine. A symmetry-aware RMSD using RDKit graph matching would be a
  natural future addition.
- **Receptor is fixed.** Ensemble functions treat the receptor as a
  single conformation; poses differ only in ligand placement.
  Mixed-receptor ensembles (e.g. one ligand across an MD ensemble of
  receptor frames) need a separate, slightly different surface.
"""

from __future__ import annotations

from molforge.ensembles.clustering import pose_clusters
from molforge.ensembles.consensus import consensus_pose
from molforge.ensembles.cross_engine import CrossEngineEnsemble, cross_engine_fold
from molforge.ensembles.density import binding_site_density
from molforge.ensembles.geometry import pairwise_rmsd, pose_diversity
from molforge.ensembles.weighting import boltzmann_weights, resample

__all__ = [
    # Weighting
    "boltzmann_weights",
    "resample",
    # Geometry
    "pairwise_rmsd",
    "pose_diversity",
    # Clustering
    "pose_clusters",
    # Spatial
    "binding_site_density",
    # Consensus
    "consensus_pose",
    # Cross-engine structural ensembles
    "cross_engine_fold",
    "CrossEngineEnsemble",
]
