"""Tests for Kabsch/Umeyama superposition."""

from __future__ import annotations

import numpy as np
import pytest

from molforge.structure import kabsch_rmsd, superpose


def _make_random_points(n: int = 50, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3)).astype(np.float64) * 10.0


class TestSuperpose:
    def test_identical_points(self) -> None:
        a = _make_random_points()
        result = superpose(a, a)
        assert result.rmsd == pytest.approx(0.0, abs=1e-6)
        np.testing.assert_allclose(result.rotation, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(result.translation, np.zeros(3), atol=1e-6)

    def test_translated_points(self) -> None:
        a = _make_random_points()
        b = a + np.array([5.0, -3.0, 7.0])
        result = superpose(a, b)
        assert result.rmsd == pytest.approx(0.0, abs=1e-6)
        np.testing.assert_allclose(result.translation, [5.0, -3.0, 7.0], atol=1e-6)

    def test_rotated_points(self) -> None:
        a = _make_random_points()
        # Rotation about z-axis by 90 degrees
        theta = np.pi / 2
        rot = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        b = (rot @ a.T).T
        result = superpose(a, b)
        assert result.rmsd == pytest.approx(0.0, abs=1e-6)
        np.testing.assert_allclose(result.rotation, rot, atol=1e-6)

    def test_noisy_alignment(self) -> None:
        a = _make_random_points(seed=1)
        rng = np.random.default_rng(2)
        noise = rng.normal(scale=0.5, size=a.shape)
        b = a + noise + np.array([10.0, 0.0, 0.0])
        result = superpose(a, b)
        # RMSD should be roughly the noise stdev (0.5 Å * sqrt(3))
        assert 0.3 < result.rmsd < 1.5

    def test_proper_rotation_guaranteed(self) -> None:
        """The returned rotation must have det = +1, never -1 (reflection)."""
        # Construct a case that would naively give a reflection.
        a = np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float64)
        b = a.copy()
        b[:, 0] *= -1  # mirror along x
        result = superpose(a, b)
        det = np.linalg.det(result.rotation)
        assert det == pytest.approx(1.0, abs=1e-6)

    def test_weights(self) -> None:
        # With one outlier, weighted superposition should down-weight it.
        a = _make_random_points(n=10, seed=3)
        b = a.copy()
        b[0] += 100.0  # outlier
        weights = np.ones(10)
        weights[0] = 0.0  # ignore the outlier completely
        result = superpose(a, b, weights=weights)
        # Without the outlier, points are identical -> RMSD ~ 0
        assert result.rmsd == pytest.approx(0.0, abs=1e-6)


class TestErrors:
    def test_shape_mismatch_raises(self) -> None:
        a = _make_random_points(5)
        b = _make_random_points(6)
        with pytest.raises(ValueError, match="shape mismatch"):
            superpose(a, b)

    def test_wrong_shape_raises(self) -> None:
        a = np.zeros((5, 2))
        b = np.zeros((5, 2))
        with pytest.raises(ValueError, match=r"\(n, 3\)"):
            superpose(a, b)

    def test_too_few_points_raises(self) -> None:
        a = np.zeros((2, 3))
        b = np.zeros((2, 3))
        with pytest.raises(ValueError, match="at least 3"):
            superpose(a, b)

    def test_bad_weights_shape_raises(self) -> None:
        a = _make_random_points(5)
        b = _make_random_points(5)
        with pytest.raises(ValueError, match="weights shape"):
            superpose(a, b, weights=np.ones(3))

    def test_negative_weights_raises(self) -> None:
        a = _make_random_points(5)
        b = _make_random_points(5)
        with pytest.raises(ValueError, match="non-negative"):
            superpose(a, b, weights=np.array([1, -1, 1, 1, 1]))


class TestKabschRmsd:
    def test_shortcut(self) -> None:
        a = _make_random_points(seed=99)
        b = a + np.array([3.0, 4.0, 0.0])
        assert kabsch_rmsd(a, b) == pytest.approx(0.0, abs=1e-6)
