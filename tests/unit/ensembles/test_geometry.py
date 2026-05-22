"""Tests for molforge.ensembles.geometry."""

from __future__ import annotations

import numpy as np
import pytest

from molforge.ensembles import pairwise_rmsd, pose_diversity


class TestPairwiseRmsdShape:
    def test_n_by_n_symmetric(self, three_collinear_poses) -> None:
        m = pairwise_rmsd(three_collinear_poses)
        assert m.shape == (3, 3)
        assert m.dtype == np.float32
        np.testing.assert_allclose(m, m.T)

    def test_zero_diagonal(self, three_collinear_poses) -> None:
        m = pairwise_rmsd(three_collinear_poses)
        np.testing.assert_array_equal(np.diag(m), np.zeros(3, dtype=np.float32))


class TestPairwiseRmsdValues:
    def test_identical_poses_have_zero_rmsd(self, two_identical_poses) -> None:
        m = pairwise_rmsd(two_identical_poses)
        np.testing.assert_allclose(m, np.zeros((2, 2)))

    def test_collinear_shifts_give_exact_rmsd(self, three_collinear_poses) -> None:
        """Pose i and pose j differ by (j-i) Å along x for every atom → RMSD = |j-i| Å."""
        m = pairwise_rmsd(three_collinear_poses)
        assert m[0, 1] == pytest.approx(1.0)
        assert m[0, 2] == pytest.approx(2.0)
        assert m[1, 2] == pytest.approx(1.0)

    def test_two_clusters_show_intra_vs_inter(self, two_clusters_poses) -> None:
        """Poses 0, 2, 4 are in cluster A; 1, 3 in cluster B at ~10 Å.

        Intra-cluster RMSDs should be small (~0.2 Å, the jitter scale);
        inter-cluster RMSDs should be ~10 Å.
        """
        m = pairwise_rmsd(two_clusters_poses)
        # Intra-A: (0,2), (0,4), (2,4)
        intra_a = [m[0, 2], m[0, 4], m[2, 4]]
        # Intra-B: (1,3)
        intra_b = [m[1, 3]]
        # Inter: everything between A and B
        inter = [m[0, 1], m[0, 3], m[2, 1], m[2, 3], m[4, 1], m[4, 3]]

        assert max(intra_a + intra_b) < 1.0
        assert min(inter) > 5.0


class TestPairwiseRmsdHydrogenHandling:
    def test_heavy_atoms_only_strips_hydrogens(self, poses_with_hydrogens) -> None:
        """Carbons identical; only hydrogens differ. With H stripped, RMSD = 0."""
        m = pairwise_rmsd(poses_with_hydrogens, heavy_atoms_only=True)
        assert m[0, 1] == pytest.approx(0.0, abs=1e-6)

    def test_all_atoms_includes_hydrogens(self, poses_with_hydrogens) -> None:
        """Same poses, but with H included, RMSD should be large (hydrogens differ)."""
        m = pairwise_rmsd(poses_with_hydrogens, heavy_atoms_only=False)
        assert m[0, 1] > 5.0


class TestPairwiseRmsdSingleton:
    def test_single_pose_returns_one_by_one_zero(self, single_pose) -> None:
        m = pairwise_rmsd(single_pose)
        assert m.shape == (1, 1)
        assert m[0, 0] == 0.0


class TestPairwiseRmsdErrors:
    def test_empty_poses_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            pairwise_rmsd([])

    def test_mismatched_atom_counts_raises(self, two_clusters_poses, single_pose) -> None:
        # Pose with 2 atoms vs poses with 3 atoms.
        mixed = [two_clusters_poses[0], single_pose[0]]
        with pytest.raises(ValueError, match="consistent atom ordering"):
            pairwise_rmsd(mixed)


class TestPoseDiversity:
    def test_returns_expected_keys(self, three_collinear_poses) -> None:
        result = pose_diversity(three_collinear_poses)
        expected_keys = {"min", "max", "mean", "median", "std", "n_poses"}
        assert set(result.keys()) == expected_keys

    def test_values_match_pairwise_matrix(self, three_collinear_poses) -> None:
        """For 3 collinear poses with gaps 1, 2, 1: upper-tri = [1, 2, 1]."""
        result = pose_diversity(three_collinear_poses)
        assert result["min"] == pytest.approx(1.0)
        assert result["max"] == pytest.approx(2.0)
        assert result["mean"] == pytest.approx(4 / 3)
        assert result["median"] == pytest.approx(1.0)
        assert result["n_poses"] == 3.0

    def test_singleton_has_zero_distance_stats(self, single_pose) -> None:
        result = pose_diversity(single_pose)
        for key in ("min", "max", "mean", "median", "std"):
            assert result[key] == 0.0
        assert result["n_poses"] == 1.0

    def test_tight_cluster_has_low_mean(self, two_identical_poses) -> None:
        result = pose_diversity(two_identical_poses)
        # Identical poses → all stats zero.
        assert result["mean"] == 0.0
        assert result["std"] == 0.0

    def test_two_clusters_has_high_max(self, two_clusters_poses) -> None:
        result = pose_diversity(two_clusters_poses)
        # Max should be the inter-cluster distance (~10 Å).
        assert result["max"] > 9.0
        # Min should be small (intra-cluster).
        assert result["min"] < 1.0


class TestPoseDiversityErrors:
    def test_empty_poses_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            pose_diversity([])
