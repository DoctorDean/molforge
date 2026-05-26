"""Tests for molforge.ensembles.consensus."""

from __future__ import annotations

import numpy as np
import pytest

from molforge.docking import Pose
from molforge.ensembles import boltzmann_weights, consensus_pose


class TestConsensusMedoid:
    def test_returns_pose(self, three_collinear_poses) -> None:
        p = consensus_pose(three_collinear_poses)
        assert isinstance(p, Pose)

    def test_default_method_is_medoid(self, three_collinear_poses) -> None:
        """Medoid of 3 collinear poses (shifts 0, 1, 2) should be the middle one."""
        p = consensus_pose(three_collinear_poses)
        # Pose 1 (rank=1) is geometrically central.
        assert p is three_collinear_poses[1]

    def test_medoid_returns_input_object_by_reference(self, three_collinear_poses) -> None:
        """Medoid mode should return one of the input poses unchanged."""
        p = consensus_pose(three_collinear_poses, method="medoid")
        assert p in three_collinear_poses

    def test_medoid_with_uniform_weights(self, three_collinear_poses) -> None:
        """Explicit uniform weights should give the same answer as no weights."""
        w = np.ones(3) / 3
        p_with = consensus_pose(three_collinear_poses, weights=w)
        p_without = consensus_pose(three_collinear_poses)
        assert p_with is p_without

    def test_medoid_biased_by_weights(self, three_collinear_poses) -> None:
        """Heavy weight on a non-central pose should not necessarily move medoid.

        Medoid still minimizes *weighted summed RMSD* — but with all weight on
        pose 2, the medoid becomes pose 2 (because its weighted sum-of-distances
        to itself is 0).
        """
        w = np.array([0.0, 0.0, 1.0])
        p = consensus_pose(three_collinear_poses, weights=w)
        assert p is three_collinear_poses[2]


class TestConsensusMean:
    def test_returns_new_pose_object(self, three_collinear_poses) -> None:
        """Mean mode synthesizes; output should be a distinct object."""
        p = consensus_pose(three_collinear_poses, method="mean")
        assert p is not three_collinear_poses[0]
        assert p is not three_collinear_poses[1]
        assert p is not three_collinear_poses[2]

    def test_uniform_mean_is_central(self, three_collinear_poses) -> None:
        """3 poses shifted by 0, 1, 2 Å along x → mean is shifted by 1 Å (= pose 1)."""
        p = consensus_pose(three_collinear_poses, method="mean")
        # Pose 1's coordinates should match the mean exactly.
        np.testing.assert_allclose(
            p.ligand.atom_array.coords,
            three_collinear_poses[1].ligand.atom_array.coords,
            atol=1e-5,
        )

    def test_weighted_mean(self, three_collinear_poses) -> None:
        """With weight 1.0 on pose 0, the mean should equal pose 0."""
        w = np.array([1.0, 0.0, 0.0])
        p = consensus_pose(three_collinear_poses, method="mean", weights=w)
        np.testing.assert_allclose(
            p.ligand.atom_array.coords,
            three_collinear_poses[0].ligand.atom_array.coords,
            atol=1e-5,
        )

    def test_mean_score_is_weighted_average(self, three_collinear_poses) -> None:
        """Mean score: -9*1/3 + -8.5*1/3 + -7*1/3 = -8.166..."""
        p = consensus_pose(three_collinear_poses, method="mean")
        assert p.score == pytest.approx(-8.1667, abs=1e-3)

    def test_mean_score_with_explicit_weights(self, three_collinear_poses) -> None:
        w = np.array([1.0, 0.0, 0.0])
        p = consensus_pose(three_collinear_poses, method="mean", weights=w)
        # Score should match pose 0 exactly.
        assert p.score == pytest.approx(-9.0)

    def test_mean_metadata_marks_synthesis(self, three_collinear_poses) -> None:
        p = consensus_pose(three_collinear_poses, method="mean")
        assert p.metadata.get("consensus_method") == "mean"
        assert p.metadata.get("consensus_n_poses") == 3

    def test_mean_does_not_mutate_inputs(self, three_collinear_poses) -> None:
        """Sanity check: consensus_pose must not modify the input poses."""
        original_coords = [p.ligand.atom_array.coords.copy() for p in three_collinear_poses]
        original_scores = [p.score for p in three_collinear_poses]

        _ = consensus_pose(three_collinear_poses, method="mean")

        for p, orig_c, orig_s in zip(
            three_collinear_poses, original_coords, original_scores, strict=True
        ):
            np.testing.assert_array_equal(p.ligand.atom_array.coords, orig_c)
            assert p.score == orig_s


class TestConsensusBoltzmannIntegration:
    def test_boltzmann_weighted_medoid(self, three_collinear_poses) -> None:
        """Real-world workflow: Boltzmann weights → medoid."""
        w = boltzmann_weights(three_collinear_poses)
        p = consensus_pose(three_collinear_poses, weights=w, method="medoid")
        # With Boltzmann weighting, the best-scoring pose pulls weight, but
        # the medoid is still geometrically determined. Pose 1 (central) is
        # likely chosen, or possibly pose 0 (best score and only 1 Å from 1).
        assert p in three_collinear_poses

    def test_boltzmann_weighted_mean_score(self, three_collinear_poses) -> None:
        """Boltzmann-weighted mean score should be lower (better) than uniform mean.

        Because Boltzmann puts more weight on the best-scoring pose, the
        weighted-average score should be closer to -9.0 than to the uniform
        mean of -8.167.
        """
        w = boltzmann_weights(three_collinear_poses)
        p_uniform = consensus_pose(three_collinear_poses, method="mean")
        p_boltzmann = consensus_pose(three_collinear_poses, method="mean", weights=w)
        # Lower (more negative) is better.
        assert p_boltzmann.score < p_uniform.score


class TestSinglePoseConsensus:
    def test_medoid_of_one(self, single_pose) -> None:
        p = consensus_pose(single_pose)
        assert p is single_pose[0]

    def test_mean_of_one(self, single_pose) -> None:
        p = consensus_pose(single_pose, method="mean")
        np.testing.assert_allclose(
            p.ligand.atom_array.coords,
            single_pose[0].ligand.atom_array.coords,
        )
        assert p.score == single_pose[0].score


class TestConsensusErrors:
    def test_empty_poses_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            consensus_pose([])

    def test_unknown_method_raises(self, three_collinear_poses) -> None:
        with pytest.raises(ValueError, match="method must be"):
            consensus_pose(three_collinear_poses, method="median")

    def test_wrong_weight_shape_raises(self, three_collinear_poses) -> None:
        with pytest.raises(ValueError, match=r"shape \(2,\), expected \(3,\)"):
            consensus_pose(three_collinear_poses, weights=np.array([0.5, 0.5]))

    def test_weights_dont_sum_to_one_raises(self, three_collinear_poses) -> None:
        with pytest.raises(ValueError, match=r"must sum to 1\.0"):
            consensus_pose(three_collinear_poses, weights=np.array([0.5, 0.5, 0.5]))

    def test_mean_with_mismatched_atom_counts_raises(
        self, three_collinear_poses, single_pose
    ) -> None:
        """Mean mode requires consistent atom counts across poses."""
        mixed = [three_collinear_poses[0], single_pose[0]]
        with pytest.raises(ValueError, match="consistent atom ordering"):
            consensus_pose(mixed, method="mean")
