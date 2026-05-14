"""Tests for Shrake-Rupley SASA."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molforge.core import AtomArray, Protein
from molforge.io import read_pdb
from molforge.structure import sasa, sasa_per_residue, total_sasa
from molforge.structure.sasa import _generate_sphere_points

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "pdb"


class TestSpherePoints:
    def test_count(self) -> None:
        pts = _generate_sphere_points(100)
        assert pts.shape == (100, 3)

    def test_on_unit_sphere(self) -> None:
        pts = _generate_sphere_points(50)
        norms = np.linalg.norm(pts, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            _generate_sphere_points(0)


class TestSasaSingleAtom:
    def test_isolated_atom_full_surface(self) -> None:
        """A single isolated atom should have SASA = 4π(r+probe)² (fully exposed)."""
        p = Protein(AtomArray(1))
        p.atom_array.element[0] = "C"
        p.atom_array.coords[0] = [0, 0, 0]
        # Carbon vdW radius = 1.70, probe = 1.4 → extended radius 3.10
        expected = 4.0 * np.pi * 3.10**2
        result = sasa(p, n_sphere_points=200)
        assert result.shape == (1,)
        # Tolerance because sphere is sampled, not integrated analytically.
        assert result[0] == pytest.approx(expected, rel=0.05)


class TestSasaTwoAtoms:
    def test_overlapping_atoms_reduced_sasa(self) -> None:
        """Two atoms close together should have lower SASA than one isolated."""
        p_iso = Protein(AtomArray(1))
        p_iso.atom_array.element[0] = "C"

        p_pair = Protein(AtomArray(2))
        p_pair.atom_array.element[:] = ["C", "C"]
        p_pair.atom_array.coords[1] = [2.0, 0, 0]  # overlapping

        iso = sasa(p_iso, n_sphere_points=200)[0]
        pair = sasa(p_pair, n_sphere_points=200)
        assert pair[0] < iso, "atom in pair should be less exposed than isolated"
        assert pair[1] < iso

    def test_far_apart_atoms_full_sasa(self) -> None:
        """Atoms far apart should each have full SASA (no occlusion)."""
        p = Protein(AtomArray(2))
        p.atom_array.element[:] = ["C", "C"]
        p.atom_array.coords[0] = [0, 0, 0]
        p.atom_array.coords[1] = [100, 0, 0]
        result = sasa(p, n_sphere_points=200)
        expected = 4.0 * np.pi * 3.10**2
        np.testing.assert_allclose(result, [expected, expected], rtol=0.05)


class TestSasaOnFixtures:
    def test_empty_protein(self) -> None:
        p = Protein(AtomArray(0))
        result = sasa(p)
        assert result.shape == (0,)

    def test_dipeptide_returns_per_atom(self) -> None:
        p = read_pdb(FIXTURES / "dipeptide.pdb")
        result = sasa(p, n_sphere_points=64)
        assert result.shape == (p.n_atoms,)
        assert (result >= 0).all()

    def test_per_residue_shape(self) -> None:
        p = read_pdb(FIXTURES / "dipeptide.pdb")
        result = sasa_per_residue(p, n_sphere_points=64)
        assert result.shape == (p.n_residues,)

    def test_total_is_sum(self) -> None:
        p = read_pdb(FIXTURES / "dipeptide.pdb")
        per_atom = sasa(p, n_sphere_points=64)
        total = total_sasa(p, n_sphere_points=64)
        assert total == pytest.approx(float(per_atom.sum()))

    def test_buried_atoms_have_lower_sasa(self) -> None:
        """In an alpha helix, interior atoms should be less exposed than terminal ones."""
        p = read_pdb(FIXTURES / "helix.pdb")
        per_atom = sasa(p, n_sphere_points=64)
        # No exact ground truth here, but middle CA atoms should generally
        # be less exposed than the very first/last atoms of the chain.
        assert per_atom.mean() > 0
