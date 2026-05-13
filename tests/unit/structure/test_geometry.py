"""Tests for geometric utilities."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from molforge.io import read_pdb
from molforge.structure import (
    bounding_box,
    center_at_origin,
    center_of_mass,
    centroid,
    radius_of_gyration,
    rotate,
    translate,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "pdb"


def test_centroid_zero_for_centered() -> None:
    p = read_pdb(FIXTURES / "dipeptide.pdb")
    center_at_origin(p)
    np.testing.assert_allclose(centroid(p), 0.0, atol=1e-5)


def test_center_of_mass_differs_from_geometric() -> None:
    # If the protein has elements with different masses, mass-weighted
    # centroid will differ from the geometric one.
    p = read_pdb(FIXTURES / "dipeptide.pdb")
    com = center_of_mass(p)
    geom = centroid(p, mass_weighted=False)
    # They differ along z (since CB is offset and CB is lighter than O)
    assert not np.allclose(com, geom)


def test_radius_of_gyration_positive() -> None:
    p = read_pdb(FIXTURES / "dipeptide.pdb")
    rg = radius_of_gyration(p)
    assert rg > 0.0


def test_radius_of_gyration_zero_for_single_atom() -> None:
    from molforge.core import AtomArray, Protein

    p = Protein(AtomArray(1))
    p.atom_array.coords[0] = [1, 1, 1]
    p.atom_array.element[0] = "C"
    assert radius_of_gyration(p) == pytest.approx(0.0)


def test_bounding_box() -> None:
    p = read_pdb(FIXTURES / "dipeptide.pdb")
    lo, hi = bounding_box(p)
    assert lo.shape == (3,)
    assert hi.shape == (3,)
    assert np.all(hi >= lo)


def test_translate_in_place() -> None:
    p = read_pdb(FIXTURES / "dipeptide.pdb")
    before = p.atom_array.coords.copy()
    translate(p, np.array([1.0, 2.0, 3.0]))
    diff = p.atom_array.coords - before
    # Every atom should have the same translation applied
    np.testing.assert_allclose(diff, np.tile([1.0, 2.0, 3.0], (diff.shape[0], 1)), atol=1e-5)


def test_translate_bad_shape_raises() -> None:
    p = read_pdb(FIXTURES / "dipeptide.pdb")
    with pytest.raises(ValueError, match=r"\(3,\)"):
        translate(p, np.array([1.0, 2.0]))


def test_rotate_in_place() -> None:
    p = read_pdb(FIXTURES / "dipeptide.pdb")
    # 90° rotation about z
    theta = np.pi / 2
    r = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    before = p.atom_array.coords.copy()
    rotate(p, r)
    # The first atom's position should be rotated.
    expected_first = (r @ before[0]).astype(np.float32)
    np.testing.assert_allclose(p.atom_array.coords[0], expected_first, atol=1e-4)


def test_rotate_bad_shape_raises() -> None:
    p = read_pdb(FIXTURES / "dipeptide.pdb")
    with pytest.raises(ValueError, match=r"\(3, 3\)"):
        rotate(p, np.eye(2))


def test_center_at_origin_mass_weighted_option() -> None:
    p = read_pdb(FIXTURES / "dipeptide.pdb")
    p_copy = deepcopy(p)
    center_at_origin(p)
    center_at_origin(p_copy, mass_weighted=True)
    # Both should now have their (respective) centers at origin
    np.testing.assert_allclose(centroid(p), 0.0, atol=1e-5)
    np.testing.assert_allclose(center_of_mass(p_copy), 0.0, atol=1e-5)
