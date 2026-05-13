"""ML utilities: featurizers, tensor views, model wrappers.

This subpackage provides utilities to turn a `Protein` into ML-ready
tensors (one-hot, distance maps, graph representations) and to wrap
common protein models behind a stable interface.
"""

from __future__ import annotations

__all__ = ["featurize", "to_graph", "to_tensor"]


def featurize(protein: object, scheme: str = "onehot") -> object:
    """Featurize a protein according to ``scheme``. TODO."""
    raise NotImplementedError


def to_graph(protein: object) -> object:
    """Convert a protein to a graph representation (e.g. for GNNs). TODO."""
    raise NotImplementedError


def to_tensor(protein: object) -> object:
    """Return a tensor representation of a protein's atoms / coords. TODO."""
    raise NotImplementedError
