"""Tests for molforge.ensembles.density."""

from __future__ import annotations

import numpy as np
import pytest

from molforge.ensembles import binding_site_density, boltzmann_weights
from molforge.ensembles.density import DensityGrid


class TestDensityGridStructure:
    def test_returns_density_grid(self, three_collinear_poses) -> None:
        g = binding_site_density(three_collinear_poses)
        assert isinstance(g, DensityGrid)

    def test_density_shape_matches_metadata(self, three_collinear_poses) -> None:
        g = binding_site_density(three_collinear_poses)
        assert g.density.shape == g.shape

    def test_origin_is_3d(self, three_collinear_poses) -> None:
        g = binding_site_density(three_collinear_poses)
        assert g.origin.shape == (3,)

    def test_default_spacing_is_one_angstrom(self, three_collinear_poses) -> None:
        g = binding_site_density(three_collinear_poses)
        assert g.spacing == 1.0


class TestDensityAutoBox:
    def test_box_covers_all_atoms(self, three_collinear_poses) -> None:
        """Every atom's coordinate must fall inside the bounding box."""
        g = binding_site_density(three_collinear_poses, padding=4.0)
        # Collect all coords across all poses.
        all_coords = np.concatenate(
            [p.ligand.atom_array.coords for p in three_collinear_poses], axis=0
        )
        # Box bounds:
        lo = g.origin
        hi = g.origin + g.spacing * np.array(g.shape, dtype=np.float32)
        assert (all_coords >= lo).all()
        assert (all_coords <= hi).all()

    def test_padding_applied(self, three_collinear_poses) -> None:
        """With padding=4, the box should extend at least 4 Å past the atoms."""
        g = binding_site_density(three_collinear_poses, padding=4.0)
        all_coords = np.concatenate(
            [p.ligand.atom_array.coords for p in three_collinear_poses], axis=0
        )
        lo_actual = g.origin
        lo_atoms = all_coords.min(axis=0)
        # Box origin should be ≥ 4 Å below the lowest atom (each axis).
        np.testing.assert_array_less(lo_actual, lo_atoms - 3.99)


class TestDensityValues:
    def test_density_is_nonneg(self, three_collinear_poses) -> None:
        g = binding_site_density(three_collinear_poses)
        assert (g.density >= 0).all()

    def test_uniform_weights_total_weight_equals_atom_count_over_n(
        self, three_collinear_poses
    ) -> None:
        """3 poses × 3 atoms each, default uniform weights (1/3 each).

        Each atom contributes 1/3 to whichever cell it lands in, so total_weight
        should equal sum(w * n_atoms_per_pose) = 3 * (1/3 * 3) = 3.0.
        """
        g = binding_site_density(three_collinear_poses)
        assert g.total_weight == pytest.approx(3.0, abs=1e-6)

    def test_weighted_total_weight_respects_input_weights(self, three_collinear_poses) -> None:
        """If pose 0 has weight 1.0 and others 0.0, only its 3 atoms count → total=3.0."""
        w = np.array([1.0, 0.0, 0.0])
        g = binding_site_density(three_collinear_poses, weights=w)
        assert g.total_weight == pytest.approx(3.0)

    def test_density_sums_to_total_weight(self, three_collinear_poses) -> None:
        """Sum over the whole grid should equal reported total_weight."""
        g = binding_site_density(three_collinear_poses)
        assert g.density.sum() == pytest.approx(g.total_weight)

    def test_hot_spot_along_overlap_region(self, three_collinear_poses) -> None:
        """The three poses are shifted by 0, 1, 2 Å along x.

        Pose 0 has atoms at x=0, 1.5, 3.0.
        Pose 1 has atoms at x=1, 2.5, 4.0.
        Pose 2 has atoms at x=2, 3.5, 5.0.

        With 1 Å spacing, the cell at x ∈ [1, 2) receives an atom from
        pose 0 (the 1.5 atom) AND pose 1 (the 1 atom), each contributing
        1/3, so that cell sums to 2/3.
        """
        g = binding_site_density(three_collinear_poses, spacing=1.0, padding=4.0)
        # The grid cell at x=1 (i.e. covering [1, 2) along x).
        # Find its index from the origin.
        target_x = 1.5  # somewhere in [1, 2)
        i_x = int(np.floor((target_x - g.origin[0]) / g.spacing))
        i_y = int(np.floor((0.0 - g.origin[1]) / g.spacing))
        i_z = int(np.floor((0.0 - g.origin[2]) / g.spacing))
        # That cell received atoms from pose 0 (x=1.5) and pose 1 (x=1.0).
        assert g.density[i_x, i_y, i_z] == pytest.approx(2 / 3)


class TestDensityWeights:
    def test_boltzmann_weighted_hot_spot_dominates(self, two_clusters_poses) -> None:
        """With Boltzmann weights, the best-score cluster should dominate the density."""
        w = boltzmann_weights(two_clusters_poses, temperature=0.5)
        g = binding_site_density(two_clusters_poses, weights=w, spacing=1.0)
        # The single most-occupied cell should be in cluster A (poses 0/2/4),
        # since pose 0 has the best score and pulls weight there.
        max_idx = np.unravel_index(g.density.argmax(), g.shape)
        hot_coord = g.coordinate_of(max_idx)
        # Cluster A is near x=0; cluster B is near x=10.
        assert hot_coord[0] < 5.0

    def test_wrong_weight_shape_raises(self, three_collinear_poses) -> None:
        with pytest.raises(ValueError, match=r"shape \(2,\), expected \(3,\)"):
            binding_site_density(three_collinear_poses, weights=np.array([0.5, 0.5]))


class TestDensityFixedGrid:
    def test_fixed_origin_and_shape(self, three_collinear_poses) -> None:
        """User provides explicit origin and shape; auto-sizing is bypassed."""
        origin = np.array([-10.0, -10.0, -10.0], dtype=np.float32)
        shape = (20, 20, 20)
        g = binding_site_density(three_collinear_poses, origin=origin, shape=shape, spacing=1.0)
        assert g.shape == shape
        np.testing.assert_array_equal(g.origin, origin)

    def test_only_origin_raises(self, three_collinear_poses) -> None:
        with pytest.raises(ValueError, match="both be provided together"):
            binding_site_density(three_collinear_poses, origin=np.array([0.0, 0.0, 0.0]))

    def test_only_shape_raises(self, three_collinear_poses) -> None:
        with pytest.raises(ValueError, match="both be provided together"):
            binding_site_density(three_collinear_poses, shape=(10, 10, 10))

    def test_atoms_outside_box_dropped_silently(self, three_collinear_poses) -> None:
        """Box that excludes all atoms → total_weight = 0."""
        origin = np.array([1000.0, 1000.0, 1000.0], dtype=np.float32)
        g = binding_site_density(three_collinear_poses, origin=origin, shape=(5, 5, 5), spacing=1.0)
        assert g.total_weight == 0.0
        assert g.density.sum() == 0.0


class TestDensityErrors:
    def test_empty_poses_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            binding_site_density([])

    def test_zero_spacing_raises(self, three_collinear_poses) -> None:
        with pytest.raises(ValueError, match="spacing must be > 0"):
            binding_site_density(three_collinear_poses, spacing=0.0)

    def test_negative_spacing_raises(self, three_collinear_poses) -> None:
        with pytest.raises(ValueError, match="spacing must be > 0"):
            binding_site_density(three_collinear_poses, spacing=-1.0)


class TestCoordinateOf:
    def test_returns_3d_array(self, three_collinear_poses) -> None:
        g = binding_site_density(three_collinear_poses)
        c = g.coordinate_of((0, 0, 0))
        assert c.shape == (3,)

    def test_center_of_origin_cell_is_origin_plus_half_spacing(self) -> None:
        """Cell (0,0,0) should be at origin + spacing/2."""
        # Build a trivial grid and check.
        from molforge.ensembles.density import DensityGrid

        g = DensityGrid(
            density=np.zeros((5, 5, 5)),
            origin=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            spacing=1.0,
            shape=(5, 5, 5),
            total_weight=0.0,
        )
        np.testing.assert_allclose(g.coordinate_of((0, 0, 0)), [0.5, 0.5, 0.5])
        np.testing.assert_allclose(g.coordinate_of((2, 3, 4)), [2.5, 3.5, 4.5])
