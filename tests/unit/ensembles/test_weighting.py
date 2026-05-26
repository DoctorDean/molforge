"""Tests for molforge.ensembles.weighting."""

from __future__ import annotations

import numpy as np
import pytest

from molforge.ensembles import boltzmann_weights, resample
from molforge.ensembles.weighting import KT_298K_KCAL_PER_MOL


class TestBoltzmannWeightsScalars:
    def test_returns_array(self) -> None:
        w = boltzmann_weights([-9.0, -8.0, -7.0])
        assert isinstance(w, np.ndarray)
        assert w.dtype == np.float64
        assert w.shape == (3,)

    def test_sums_to_one(self) -> None:
        w = boltzmann_weights([-9.0, -8.0, -7.0])
        assert w.sum() == pytest.approx(1.0)

    def test_best_score_gets_highest_weight(self) -> None:
        w = boltzmann_weights([-9.0, -8.0, -7.0])
        assert w[0] > w[1] > w[2]

    def test_equal_scores_give_uniform_weights(self) -> None:
        w = boltzmann_weights([-5.0, -5.0, -5.0])
        np.testing.assert_allclose(w, [1 / 3, 1 / 3, 1 / 3])

    def test_higher_is_better_flag(self) -> None:
        # If we flip the convention, the highest score should win.
        w = boltzmann_weights([0.9, 0.5, 0.1], lower_is_better=False)
        assert w[0] > w[1] > w[2]
        assert w.sum() == pytest.approx(1.0)

    def test_lower_is_better_with_positive_scores(self) -> None:
        # Should still work for positive scores under "lower is better".
        w = boltzmann_weights([1.0, 2.0, 3.0])
        assert w[0] > w[1] > w[2]
        assert w.sum() == pytest.approx(1.0)


class TestBoltzmannWeightsTemperature:
    def test_high_temperature_smooths(self) -> None:
        """High T → weights approach uniform."""
        w_low = boltzmann_weights([-10.0, -5.0], temperature=0.1)
        w_high = boltzmann_weights([-10.0, -5.0], temperature=1000.0)
        # At very high T (kT >> score gap), weights should be ~0.5/0.5.
        # With kT=1000 and a 5 kcal/mol gap, residual asymmetry is ~0.0025.
        assert w_high[0] == pytest.approx(0.5, abs=0.01)
        # At very low T, almost all weight on the best score.
        assert w_low[0] > 0.99

    def test_low_temperature_sharpens(self) -> None:
        """At very low T, the best score should approach weight 1."""
        w = boltzmann_weights([-10.0, -9.0, -8.0], temperature=0.01)
        assert w[0] == pytest.approx(1.0, abs=1e-9)

    def test_zero_temperature_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature must be > 0"):
            boltzmann_weights([-9.0, -8.0], temperature=0.0)

    def test_negative_temperature_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature must be > 0"):
            boltzmann_weights([-9.0, -8.0], temperature=-1.0)

    def test_default_temperature_is_kt_at_room_temp(self) -> None:
        # Smoke check: room-temperature kT is approximately 0.593 kcal/mol.
        assert pytest.approx(0.5925, abs=1e-3) == KT_298K_KCAL_PER_MOL

    def test_default_temperature_gives_realistic_spread(self) -> None:
        """With kT=0.59 and a 1 kcal/mol gap, the spread should be ~5.4x.

        e^(1/0.5925) ≈ 5.39, so the better pose should get ~5.4x the weight.
        """
        w = boltzmann_weights([-10.0, -9.0])
        assert w[0] / w[1] == pytest.approx(5.39, abs=0.1)


class TestBoltzmannWeightsNumericalStability:
    def test_very_negative_scores(self) -> None:
        """Energies like -1000 kcal/mol shouldn't overflow."""
        w = boltzmann_weights([-1000.0, -999.0, -998.0])
        assert np.all(np.isfinite(w))
        assert w.sum() == pytest.approx(1.0)

    def test_very_positive_scores(self) -> None:
        """Mirror case: scores like +1000 shouldn't underflow to all zeros."""
        w = boltzmann_weights([1000.0, 1001.0, 1002.0])
        assert np.all(np.isfinite(w))
        assert w.sum() == pytest.approx(1.0)
        # 1000 is the lowest = best.
        assert w[0] > w[1] > w[2]

    def test_large_score_range(self) -> None:
        """Score range >> kT shouldn't cause issues."""
        w = boltzmann_weights([-100.0, 100.0])
        assert np.all(np.isfinite(w))
        assert w.sum() == pytest.approx(1.0)
        # Such a huge gap → essentially all weight on the better one.
        assert w[0] > 1 - 1e-10


class TestBoltzmannWeightsInputs:
    def test_accepts_numpy_array(self) -> None:
        arr = np.array([-9.0, -8.0, -7.0])
        w = boltzmann_weights(arr)
        assert w.shape == (3,)
        assert w.sum() == pytest.approx(1.0)

    def test_accepts_2d_numpy_array_ravels(self) -> None:
        # Should flatten; in practice users pass 1D but we handle 2D gracefully.
        arr = np.array([[-9.0, -8.0], [-7.0, -6.0]])
        w = boltzmann_weights(arr)
        assert w.shape == (4,)

    def test_accepts_tuple(self) -> None:
        w = boltzmann_weights((-9.0, -8.0, -7.0))
        assert w.sum() == pytest.approx(1.0)

    def test_accepts_pose_sequence(self, three_collinear_poses) -> None:
        """Should auto-extract .score from Pose objects."""
        w = boltzmann_weights(three_collinear_poses)
        assert w.shape == (3,)
        assert w.sum() == pytest.approx(1.0)
        # Pose 0 has score -9.0 (best) → highest weight.
        assert w[0] > w[1] > w[2]

    def test_empty_scores_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            boltzmann_weights([])

    def test_nan_score_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            boltzmann_weights([-9.0, float("nan"), -7.0])

    def test_inf_score_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            boltzmann_weights([-9.0, float("inf"), -7.0])


class TestResample:
    def test_returns_list_of_correct_length(self, three_collinear_poses) -> None:
        out = resample(three_collinear_poses, n_samples=10)
        assert isinstance(out, list)
        assert len(out) == 10

    def test_reproducible_with_seeded_rng(self, three_collinear_poses) -> None:
        rng1 = np.random.default_rng(seed=42)
        rng2 = np.random.default_rng(seed=42)
        out1 = resample(three_collinear_poses, n_samples=20, rng=rng1)
        out2 = resample(three_collinear_poses, n_samples=20, rng=rng2)
        # Same seed → same sequence of indices → same pose objects.
        assert [p.score for p in out1] == [p.score for p in out2]

    def test_uniform_weights_default(self, three_collinear_poses) -> None:
        """No weights → each pose should appear roughly evenly over many draws."""
        rng = np.random.default_rng(seed=0)
        out = resample(three_collinear_poses, n_samples=10_000, rng=rng)
        counts = np.bincount([p.rank for p in out], minlength=3)
        # Three poses, ~10000 draws, expect ~3333 each. 3σ tolerance.
        for c in counts:
            assert 3200 <= c <= 3500

    def test_weighted_sampling(self, three_collinear_poses) -> None:
        """With heavily biased weights, the favored pose should dominate."""
        weights = np.array([0.9, 0.05, 0.05])
        rng = np.random.default_rng(seed=0)
        out = resample(three_collinear_poses, n_samples=1000, weights=weights, rng=rng)
        counts = np.bincount([p.rank for p in out], minlength=3)
        assert 850 <= counts[0] <= 950

    def test_boltzmann_weighted_resample_round_trip(self, three_collinear_poses) -> None:
        """boltzmann_weights → resample produces population biased toward best score."""
        weights = boltzmann_weights(three_collinear_poses)
        rng = np.random.default_rng(seed=0)
        out = resample(three_collinear_poses, n_samples=1000, weights=weights, rng=rng)
        counts = np.bincount([p.rank for p in out], minlength=3)
        # Best pose (rank 0) should be most common.
        assert counts[0] > counts[1] > counts[2]

    def test_returns_same_pose_objects_by_reference(self, three_collinear_poses) -> None:
        rng = np.random.default_rng(seed=0)
        out = resample(three_collinear_poses, n_samples=5, rng=rng)
        # Every returned pose is the *same object* (not a copy) as one of the inputs.
        for p in out:
            assert p is three_collinear_poses[p.rank]

    def test_without_replacement(self, three_collinear_poses) -> None:
        rng = np.random.default_rng(seed=0)
        out = resample(three_collinear_poses, n_samples=3, replace=False, rng=rng)
        assert len(out) == 3
        assert sorted(p.rank for p in out) == [0, 1, 2]

    def test_without_replacement_oversample_raises(self, three_collinear_poses) -> None:
        with pytest.raises(ValueError, match="replace=False"):
            resample(three_collinear_poses, n_samples=10, replace=False)


class TestResampleErrors:
    def test_empty_poses_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            resample([], n_samples=5)

    def test_n_samples_zero_raises(self, three_collinear_poses) -> None:
        with pytest.raises(ValueError, match="n_samples must be >= 1"):
            resample(three_collinear_poses, n_samples=0)

    def test_n_samples_negative_raises(self, three_collinear_poses) -> None:
        with pytest.raises(ValueError, match="n_samples must be >= 1"):
            resample(three_collinear_poses, n_samples=-3)

    def test_weights_wrong_length_raises(self, three_collinear_poses) -> None:
        with pytest.raises(ValueError, match=r"shape \(2,\), expected \(3,\)"):
            resample(three_collinear_poses, n_samples=5, weights=np.array([0.5, 0.5]))

    def test_weights_dont_sum_to_one_raises(self, three_collinear_poses) -> None:
        with pytest.raises(ValueError, match=r"must sum to 1\.0"):
            resample(
                three_collinear_poses,
                n_samples=5,
                weights=np.array([0.5, 0.5, 0.5]),  # sums to 1.5
            )
