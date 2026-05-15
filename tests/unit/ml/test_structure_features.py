"""Tests for structure featurization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molforge.io import read_pdb
from molforge.ml import (
    local_environment,
    pair_distance_features,
    pair_distances,
    pair_orientations,
    per_residue_features,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "pdb"


class TestPairDistances:
    def test_shape_and_dtype(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        d = pair_distances(p, atom_choice="ca")
        assert d.shape == (15, 15)
        assert d.dtype == np.float32

    def test_diagonal_zero(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        d = pair_distances(p)
        np.testing.assert_allclose(np.diag(d), 0.0, atol=1e-5)

    def test_symmetric(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        d = pair_distances(p)
        np.testing.assert_allclose(d, d.T, atol=1e-5)


class TestPairDistanceFeatures:
    def test_shape(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        rbf = pair_distance_features(p, n_bins=16)
        assert rbf.shape == (15, 15, 16)
        assert rbf.dtype == np.float32

    def test_values_in_range(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        rbf = pair_distance_features(p, n_bins=8)
        # RBF basis values in [0, 1]
        assert rbf.min() >= 0.0
        assert rbf.max() <= 1.0 + 1e-6

    def test_default_centers_cover_typical_distances(self) -> None:
        """For a typical helix-residue pair, at least one bin should activate
        strongly (> 0.5)."""
        p = read_pdb(FIXTURES / "helix.pdb")
        rbf = pair_distance_features(p, n_bins=16, d_min=2.0, d_max=22.0)
        # Sequential CA-CA in alpha helix ~ 5.4 Å
        assert rbf[0, 1].max() > 0.5


class TestPairOrientations:
    def test_keys_present(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        orientations = pair_orientations(p)
        assert set(orientations.keys()) == {"direction", "distance", "cosine"}

    def test_direction_shape(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        orientations = pair_orientations(p)
        assert orientations["direction"].shape == (15, 15, 3)

    def test_distance_matches_pair_distances(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        orientations = pair_orientations(p)
        baseline = pair_distances(p)
        np.testing.assert_allclose(orientations["distance"], baseline, atol=1e-4)

    def test_cosine_in_range(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        orientations = pair_orientations(p)
        c = orientations["cosine"]
        # cosines must be in [-1, 1] (numerical slop allowed)
        assert c.min() >= -1.001
        assert c.max() <= 1.001

    def test_direction_unit_length(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        orientations = pair_orientations(p)
        d = orientations["direction"]
        # Off-diagonal direction vectors should be ~unit length
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                if i != j:
                    assert np.linalg.norm(d[i, j]) == pytest.approx(1.0, abs=1e-4)


class TestLocalEnvironment:
    def test_shape(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        env = local_environment(p, radius=10.0)
        assert env.shape == (15, 5)
        assert env.dtype == np.float32

    def test_all_counts_nonnegative(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        env = local_environment(p, radius=10.0)
        assert (env >= 0).all()

    def test_larger_radius_increases_counts(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        tight = local_environment(p, radius=3.0)
        loose = local_environment(p, radius=15.0)
        assert loose.sum() >= tight.sum()


class TestPerResidueFeatures:
    def test_default_shape(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        feats = per_residue_features(p)
        # 21 (one-hot) + 5 (env) + 3 (DSSP) = 29
        assert feats.shape == (15, 29)
        assert feats.dtype == np.float32

    def test_no_environment(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        feats = per_residue_features(p, include_environment=False)
        # 21 + 3 = 24
        assert feats.shape == (15, 24)

    def test_no_dssp(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        feats = per_residue_features(p, include_dssp=False)
        # 21 + 5 = 26
        assert feats.shape == (15, 26)

    def test_only_sequence(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        feats = per_residue_features(p, include_environment=False, include_dssp=False)
        # 21 only
        assert feats.shape == (15, 21)
