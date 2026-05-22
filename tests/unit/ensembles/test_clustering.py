"""Tests for molforge.ensembles.clustering."""

from __future__ import annotations

import numpy as np
import pytest

from molforge.ensembles import pose_clusters
from molforge.ensembles.clustering import PoseCluster, PoseClusteringResult


class TestClusteringStructure:
    def test_returns_pose_clustering_result(self, three_collinear_poses) -> None:
        out = pose_clusters(three_collinear_poses)
        assert isinstance(out, PoseClusteringResult)

    def test_labels_length_matches_input(self, three_collinear_poses) -> None:
        out = pose_clusters(three_collinear_poses)
        assert out.labels.shape == (3,)
        assert out.labels.dtype == np.intp

    def test_rmsd_matrix_exposed(self, three_collinear_poses) -> None:
        out = pose_clusters(three_collinear_poses)
        assert out.rmsd_matrix.shape == (3, 3)
        # Symmetric, zero diagonal.
        np.testing.assert_allclose(out.rmsd_matrix, out.rmsd_matrix.T)
        np.testing.assert_array_equal(np.diag(out.rmsd_matrix), np.zeros(3))

    def test_cutoff_stored(self, three_collinear_poses) -> None:
        out = pose_clusters(three_collinear_poses, cutoff=1.5)
        assert out.cutoff == 1.5

    def test_n_clusters_matches_clusters_list(self, two_clusters_poses) -> None:
        out = pose_clusters(two_clusters_poses, cutoff=2.0)
        assert out.n_clusters == len(out.clusters)


class TestTwoClusterRecovery:
    """The headline use case: recover known cluster structure."""

    def test_finds_two_clusters_in_obvious_case(self, two_clusters_poses) -> None:
        """5 poses in 2 known clusters separated by ~10 Å."""
        out = pose_clusters(two_clusters_poses, cutoff=2.0)
        assert out.n_clusters == 2

    def test_cluster_membership_correct(self, two_clusters_poses) -> None:
        """Poses 0, 2, 4 are cluster A; 1, 3 are cluster B."""
        out = pose_clusters(two_clusters_poses, cutoff=2.0)
        labels = out.labels
        # All three pose indices in cluster A should share a label.
        assert labels[0] == labels[2] == labels[4]
        # Both pose indices in cluster B should share a label.
        assert labels[1] == labels[3]
        # The two cluster labels should differ.
        assert labels[0] != labels[1]

    def test_larger_cluster_listed_first(self, two_clusters_poses) -> None:
        """Cluster A has 3 members; it should come first in the sorted list."""
        out = pose_clusters(two_clusters_poses, cutoff=2.0)
        assert out.clusters[0].size == 3
        assert out.clusters[1].size == 2

    def test_medoid_belongs_to_its_cluster(self, two_clusters_poses) -> None:
        out = pose_clusters(two_clusters_poses, cutoff=2.0)
        for cluster in out.clusters:
            assert cluster.medoid in cluster.members


class TestSingleClusterFallback:
    def test_large_cutoff_collapses_everything(self, two_clusters_poses) -> None:
        """With cutoff >> inter-cluster distance, all poses merge."""
        out = pose_clusters(two_clusters_poses, cutoff=100.0)
        assert out.n_clusters == 1
        assert out.clusters[0].size == len(two_clusters_poses)

    def test_tiny_cutoff_yields_singletons(self, three_collinear_poses) -> None:
        """Cutoff smaller than minimum inter-pose RMSD → every pose alone."""
        out = pose_clusters(three_collinear_poses, cutoff=0.5)
        assert out.n_clusters == 3
        for cluster in out.clusters:
            assert cluster.size == 1
            assert cluster.mean_intra_rmsd == 0.0

    def test_identical_poses_in_one_cluster(self, two_identical_poses) -> None:
        out = pose_clusters(two_identical_poses, cutoff=0.5)
        assert out.n_clusters == 1
        assert out.clusters[0].size == 2
        assert out.clusters[0].mean_intra_rmsd == 0.0


class TestSinglePose:
    def test_singleton_makes_one_cluster(self, single_pose) -> None:
        out = pose_clusters(single_pose)
        assert out.n_clusters == 1
        assert out.clusters[0].size == 1
        assert out.clusters[0].medoid == 0
        assert out.clusters[0].mean_intra_rmsd == 0.0


class TestMedoidCorrectness:
    def test_medoid_is_central_pose(self, three_collinear_poses) -> None:
        """Three poses at x=0, 1, 2 → if all in one cluster, medoid is pose 1.

        Sum of distances: pose 0 → 1 + 2 = 3; pose 1 → 1 + 1 = 2; pose 2 → 2 + 1 = 3.
        Pose 1 has the smallest total distance.
        """
        out = pose_clusters(three_collinear_poses, cutoff=10.0)
        assert out.n_clusters == 1
        assert out.clusters[0].medoid == 1

    def test_medoid_breaks_ties_deterministically(self) -> None:
        """When two poses tie for medoid, the lower index wins."""
        # Build two identical poses where both are equally "central" to a third.
        from tests.unit.ensembles.conftest import make_pose

        c0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        c1 = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
        c2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)  # same as c0
        poses = [make_pose(c0, -9.0, 0), make_pose(c1, -8.0, 1), make_pose(c2, -7.0, 2)]
        out = pose_clusters(poses, cutoff=10.0)
        assert out.n_clusters == 1
        # Poses 0 and 2 are identical (tied medoids); 0 wins by argmin tie-break.
        assert out.clusters[0].medoid == 0


class TestMeanIntraRmsd:
    def test_mean_intra_rmsd_is_zero_for_singleton(self, single_pose) -> None:
        out = pose_clusters(single_pose)
        assert out.clusters[0].mean_intra_rmsd == 0.0

    def test_mean_intra_rmsd_is_zero_for_identical_poses(self, two_identical_poses) -> None:
        out = pose_clusters(two_identical_poses)
        assert out.clusters[0].mean_intra_rmsd == 0.0

    def test_mean_intra_rmsd_matches_pairwise(self, three_collinear_poses) -> None:
        """If all 3 poses cluster together, mean intra-RMSD = mean of [1, 2, 1] = 4/3."""
        out = pose_clusters(three_collinear_poses, cutoff=10.0)
        assert out.n_clusters == 1
        assert out.clusters[0].mean_intra_rmsd == pytest.approx(4 / 3)


class TestClusteringErrors:
    def test_empty_poses_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            pose_clusters([])

    def test_zero_cutoff_raises(self, three_collinear_poses) -> None:
        with pytest.raises(ValueError, match="cutoff must be > 0"):
            pose_clusters(three_collinear_poses, cutoff=0.0)

    def test_negative_cutoff_raises(self, three_collinear_poses) -> None:
        with pytest.raises(ValueError, match="cutoff must be > 0"):
            pose_clusters(three_collinear_poses, cutoff=-1.0)


class TestPoseClusterDataclass:
    def test_constructible(self) -> None:
        c = PoseCluster(members=[0, 1, 2], medoid=1, size=3, mean_intra_rmsd=0.5)
        assert c.members == [0, 1, 2]
        assert c.medoid == 1
        assert c.size == 3
        assert c.mean_intra_rmsd == 0.5
