"""ML utilities: featurizers, embeddings, and graph representations.

Turn a :class:`molforge.core.Protein` into ML-ready tensors. Three layers:

**Sequence featurizers** (no structure needed):
    - :func:`one_hot` — 21-dim one-hot encoding.
    - :func:`blosum_embed` — BLOSUM62 / PAM250 rows as embeddings.
    - :func:`positional_encoding` — sinusoidal (Vaswani-style) positional.
    - :func:`compose_features` — concatenate featurizers.

**Structure featurizers** (need 3D coordinates):
    - :func:`pair_distances` — float32 distance map.
    - :func:`pair_distance_features` — Gaussian-RBF binned distances
      (the standard featurization for distance-based GNNs).
    - :func:`pair_orientations` — backbone orientation features
      (CA-CA vectors, distances, and local-frame cosines).
    - :func:`local_environment` — per-residue atomic environment counts
      by element.
    - :func:`per_residue_features` — combined node-feature vector
      (one-hot + environment + DSSP).

**Protein language model embeddings** (heavy, lazy):
    - :class:`ESM2Embedder` — wraps ESM-2 via HuggingFace transformers.

**Graph construction**:
    - :func:`to_graph` — build a :class:`ProteinGraph` ready for
      PyTorch Geometric / DGL.

The featurizers return plain NumPy float32 arrays. Convert to your
preferred tensor library downstream (``torch.from_numpy(arr)`` etc.).
"""

from __future__ import annotations

from molforge.ml.embeddings import EmbeddingNotInstalledError, ESM2Embedder
from molforge.ml.graph import ProteinGraph, to_graph
from molforge.ml.sequence_features import (
    blosum_embed,
    compose_features,
    one_hot,
    positional_encoding,
)
from molforge.ml.structure_features import (
    local_environment,
    pair_distance_features,
    pair_distances,
    pair_orientations,
    per_residue_features,
)

__all__ = [  # noqa: RUF022 — grouped by concern
    # Sequence
    "one_hot",
    "blosum_embed",
    "positional_encoding",
    "compose_features",
    # Structure
    "pair_distances",
    "pair_distance_features",
    "pair_orientations",
    "local_environment",
    "per_residue_features",
    # Embeddings
    "ESM2Embedder",
    "EmbeddingNotInstalledError",
    # Graph
    "to_graph",
    "ProteinGraph",
]
