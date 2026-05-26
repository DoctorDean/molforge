"""Smoke tests for the molforge.ensembles public surface."""

from __future__ import annotations

import molforge.ensembles as ens


class TestPublicSurface:
    def test_all_is_complete(self) -> None:
        """Everything in __all__ is importable from the top-level module."""
        for name in ens.__all__:
            assert hasattr(ens, name), f"missing from ensembles: {name}"

    def test_expected_functions_in_all(self) -> None:
        """The seven advertised functions are exported."""
        expected = {
            "boltzmann_weights",
            "resample",
            "pairwise_rmsd",
            "pose_diversity",
            "pose_clusters",
            "binding_site_density",
            "consensus_pose",
        }
        assert set(ens.__all__) == expected

    def test_no_private_leakage(self) -> None:
        """Underscore-prefixed names should not appear in __all__."""
        for name in ens.__all__:
            assert not name.startswith("_")


class TestEndToEndPipeline:
    """Smoke test for the canonical ensembles workflow."""

    def test_score_to_weights_to_clusters_to_consensus(self, two_clusters_poses) -> None:
        """Full pipeline: docking output → Boltzmann weights → clusters → consensus."""
        # 1. Weight by score.
        weights = ens.boltzmann_weights(two_clusters_poses)
        assert weights.shape == (5,)
        assert weights.sum() > 0.99

        # 2. Diversity statistics.
        stats = ens.pose_diversity(two_clusters_poses)
        assert stats["n_poses"] == 5.0
        assert stats["max"] > 5.0  # cluster separation

        # 3. Cluster.
        clusters = ens.pose_clusters(two_clusters_poses, cutoff=2.0)
        assert clusters.n_clusters == 2

        # 4. Density map.
        grid = ens.binding_site_density(two_clusters_poses, weights=weights)
        assert grid.density.sum() > 0

        # 5. Consensus.
        rep = ens.consensus_pose(two_clusters_poses, weights=weights)
        assert rep in two_clusters_poses
