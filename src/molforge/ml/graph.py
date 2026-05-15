"""Graph construction for protein GNNs.

Builds residue-graph representations (nodes = residues, edges between
residues within some distance cutoff) suitable for graph neural
networks. Compatible with PyTorch Geometric, DGL, and similar
frameworks via the standard ``(edge_index, node_features, edge_features)``
contract — but returns plain NumPy arrays so callers can convert to
whatever tensor library they prefer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from molforge.core import Protein


@dataclass(frozen=True)
class ProteinGraph:
    """A residue-graph representation of a protein structure.

    Attributes:
        node_features: ``(n_res, F_node)`` float32 array.
        edge_index: ``(2, n_edges)`` int64 array of (source, target)
            residue index pairs. Follows the PyTorch Geometric
            convention: ``edge_index[0]`` is the source row, edge_index[1]`
            is the target.
        edge_features: ``(n_edges, F_edge)`` float32 array.
        residue_labels: ``[(chain_id, residue_id)]`` for each node.
    """

    node_features: NDArray[np.float32]
    edge_index: NDArray[np.int64]
    edge_features: NDArray[np.float32]
    residue_labels: list[tuple[str, int]]

    @property
    def n_nodes(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def node_dim(self) -> int:
        return int(self.node_features.shape[1]) if self.node_features.size else 0

    @property
    def edge_dim(self) -> int:
        return int(self.edge_features.shape[1]) if self.edge_features.size else 0


def to_graph(
    protein: Protein,
    *,
    cutoff: float = 10.0,
    self_loops: bool = False,
    include_dssp: bool = True,
    include_environment: bool = True,
    edge_distance_bins: int = 16,
) -> ProteinGraph:
    """Build a residue-graph representation of ``protein``.

    Nodes: each protein residue with a CA atom.
    Edges: undirected (both directions present) connections between
    residues whose CA atoms are within ``cutoff`` Å of each other.

    Args:
        protein: input structure.
        cutoff: edge cutoff distance in Å. 10 Å is the typical default
            for protein GNNs (captures both direct and ~1-residue-away
            contacts).
        self_loops: include i->i edges. Some GNN architectures want
            these; most don't.
        include_dssp, include_environment: passed to
            :func:`per_residue_features`.
        edge_distance_bins: number of RBF basis functions for the
            edge-distance feature; pass 0 to use raw distance.

    Returns:
        A :class:`ProteinGraph`.

    Example:
        >>> from molforge.ml import to_graph
        >>> g = to_graph(my_protein, cutoff=8.0)
        >>> g.node_features.shape
        (76, 29)
        >>> g.edge_index.shape
        (2, 421)
        >>> g.edge_features.shape
        (421, 16)
    """
    from molforge.ml.structure_features import (
        _ca_coords_and_labels,
        pair_distance_features,
        per_residue_features,
    )

    # Node features
    node_features = per_residue_features(
        protein,
        include_environment=include_environment,
        include_dssp=include_dssp,
    )
    coords, labels = _ca_coords_and_labels(protein)
    n = coords.shape[0]

    if n == 0:
        return ProteinGraph(
            node_features=node_features,
            edge_index=np.zeros((2, 0), dtype=np.int64),
            edge_features=np.zeros((0, edge_distance_bins or 1), dtype=np.float32),
            residue_labels=labels,
        )

    # Pairwise CA distances
    diff = coords[None, :, :] - coords[:, None, :]
    dist = np.linalg.norm(diff, axis=-1)
    mask = dist < cutoff
    if not self_loops:
        np.fill_diagonal(mask, False)
    sources, targets = np.where(mask)
    edge_index = np.stack([sources, targets], axis=0).astype(np.int64)

    # Edge features — distance, encoded as either RBF or raw
    if edge_distance_bins > 0:
        all_rbf = pair_distance_features(
            protein,
            n_bins=edge_distance_bins,
        )
        edge_features = all_rbf[sources, targets].astype(np.float32)
    else:
        edge_features = dist[sources, targets][:, None].astype(np.float32)

    return ProteinGraph(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        residue_labels=labels,
    )
