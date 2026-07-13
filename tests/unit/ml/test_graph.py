"""Tests for protein-graph construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from molforge.core import Protein
from molforge.core.atom_array import AtomArray
from molforge.io import read_pdb
from molforge.ml import ProteinGraph, to_graph

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "pdb"


def _prepend_far_water(protein: Protein) -> Protein:
    """Copy of ``protein`` with one far-away water residue prepended."""
    arr = protein.atom_array
    water = AtomArray.from_dict(
        {
            "coords": np.array([[999.0, 999.0, 999.0]], dtype=np.float32),
            "atom_name": np.array(["O"], dtype="U4"),
            "element": np.array(["O"], dtype="U2"),
            "residue_name": np.array(["HOH"], dtype="U3"),
            "residue_id": np.array([1], dtype="int32"),
            "chain_id": np.array(["W"], dtype="U4"),
            "entity_type": np.array(["water"], dtype="U8"),
        }
    )
    return Protein(water.append(arr))


class TestProteinGraph:
    def test_dataclass_fields(self) -> None:
        g = ProteinGraph(
            node_features=np.zeros((3, 5), dtype=np.float32),
            edge_index=np.zeros((2, 4), dtype=np.int64),
            edge_features=np.zeros((4, 16), dtype=np.float32),
            residue_labels=[("A", 1), ("A", 2), ("A", 3)],
        )
        assert g.n_nodes == 3
        assert g.n_edges == 4
        assert g.node_dim == 5
        assert g.edge_dim == 16


class TestBasicConstruction:
    def test_returns_protein_graph(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        g = to_graph(p)
        assert isinstance(g, ProteinGraph)

    def test_correct_node_count(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        g = to_graph(p)
        # 15 protein residues in helix
        assert g.n_nodes == 15

    def test_default_node_feature_dim(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        g = to_graph(p)
        # Same as per_residue_features default: 29
        assert g.node_dim == 29

    def test_default_edge_feature_dim(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        g = to_graph(p, edge_distance_bins=16)
        assert g.edge_dim == 16

    def test_no_self_loops_by_default(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        g = to_graph(p, self_loops=False)
        # No i->i edges
        sources, targets = g.edge_index
        assert (sources != targets).all()

    def test_self_loops_when_requested(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        g = to_graph(p, self_loops=True)
        # Every node should have one self-loop
        sources, targets = g.edge_index
        self_loop_count = int((sources == targets).sum())
        assert self_loop_count == g.n_nodes


class TestEdgeCutoff:
    def test_smaller_cutoff_yields_fewer_edges(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        g_tight = to_graph(p, cutoff=5.0)
        g_loose = to_graph(p, cutoff=15.0)
        assert g_tight.n_edges <= g_loose.n_edges

    def test_zero_cutoff_no_edges(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        g = to_graph(p, cutoff=0.0, self_loops=False)
        assert g.n_edges == 0


class TestEdgeFeatures:
    def test_distance_bins_zero_uses_raw(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        g = to_graph(p, edge_distance_bins=0)
        # Raw distance has 1 column
        assert g.edge_dim == 1

    def test_distance_bins_custom(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        g = to_graph(p, edge_distance_bins=8)
        assert g.edge_dim == 8


class TestUndirectedness:
    def test_edges_are_bidirectional(self) -> None:
        """For each (i, j) edge there should be a matching (j, i)."""
        p = read_pdb(FIXTURES / "helix.pdb")
        g = to_graph(p)
        edges = {(int(s), int(t)) for s, t in zip(g.edge_index[0], g.edge_index[1], strict=True)}
        for s, t in edges:
            assert (t, s) in edges, f"missing reverse edge for ({s}, {t})"


class TestEmpty:
    def test_empty_protein(self) -> None:
        from molforge.core import AtomArray, Protein

        p = Protein(AtomArray(0))
        g = to_graph(p)
        assert g.n_nodes == 0
        assert g.n_edges == 0


class TestNonProteinInvariance:
    """A far non-protein residue must not change the protein graph. Before the
    fix, edge features were read from an all-residue distance matrix indexed by
    CA-only edge indices, so a single water corrupted every edge feature.
    """

    def test_graph_ignores_far_water(self) -> None:
        base = read_pdb(FIXTURES / "real_ubiquitin.pdb")
        g_base = to_graph(base)
        g_wat = to_graph(_prepend_far_water(base))
        assert g_wat.n_nodes == g_base.n_nodes
        assert np.array_equal(g_wat.edge_index, g_base.edge_index)
        np.testing.assert_allclose(g_wat.node_features, g_base.node_features)
        np.testing.assert_allclose(g_wat.edge_features, g_base.edge_features)
