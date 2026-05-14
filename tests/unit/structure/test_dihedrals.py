"""Tests for backbone dihedrals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molforge.core import AtomArray, Protein
from molforge.io import read_pdb
from molforge.structure import (
    dihedral,
    dihedrals_batch,
    omega,
    phi,
    phi_psi_omega,
    psi,
    ramachandran,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "pdb"


class TestDihedralScalar:
    def test_180_degrees(self) -> None:
        """Four coplanar points in a planar zigzag → ±180°."""
        p1 = np.array([0, 0, 0], dtype=np.float64)
        p2 = np.array([1, 0, 0], dtype=np.float64)
        p3 = np.array([1, 1, 0], dtype=np.float64)
        p4 = np.array([2, 1, 0], dtype=np.float64)
        assert abs(abs(dihedral(p1, p2, p3, p4)) - 180.0) < 1e-6

    def test_zero_degrees(self) -> None:
        """Four coplanar points eclipsed → 0°."""
        p1 = np.array([0, 1, 0], dtype=np.float64)
        p2 = np.array([1, 0, 0], dtype=np.float64)
        p3 = np.array([2, 0, 0], dtype=np.float64)
        p4 = np.array([3, 1, 0], dtype=np.float64)
        assert abs(dihedral(p1, p2, p3, p4)) < 1e-6

    def test_90_degrees(self) -> None:
        """Plane 1 in xy, plane 2 in xz → 90°."""
        p1 = np.array([0, 1, 0], dtype=np.float64)
        p2 = np.array([0, 0, 0], dtype=np.float64)
        p3 = np.array([1, 0, 0], dtype=np.float64)
        p4 = np.array([1, 0, 1], dtype=np.float64)
        assert abs(abs(dihedral(p1, p2, p3, p4)) - 90.0) < 1e-6

    def test_degenerate_returns_nan(self) -> None:
        """When p2 == p3 the dihedral is undefined."""
        p1 = np.array([0, 0, 0], dtype=np.float64)
        p2 = np.array([1, 0, 0], dtype=np.float64)
        p3 = np.array([1, 0, 0], dtype=np.float64)  # same as p2
        p4 = np.array([2, 1, 0], dtype=np.float64)
        result = dihedral(p1, p2, p3, p4)
        assert np.isnan(result)


class TestDihedralsBatch:
    def test_matches_scalar(self) -> None:
        rng = np.random.default_rng(42)
        quartets = rng.normal(size=(10, 4, 3))
        batch = dihedrals_batch(quartets)
        individual = np.array([dihedral(q[0], q[1], q[2], q[3]) for q in quartets])
        np.testing.assert_allclose(batch, individual, atol=1e-10)

    def test_bad_shape_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\(N, 4, 3\)"):
            dihedrals_batch(np.zeros((5, 3)))


class TestProteinDihedrals:
    def test_phi_psi_shapes(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        phi_vals, psi_vals, omega_vals = phi_psi_omega(p)
        assert phi_vals.shape == (p.n_residues,)
        assert psi_vals.shape == (p.n_residues,)
        assert omega_vals.shape == (p.n_residues,)

    def test_chain_termini_are_nan(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        phi_vals, psi_vals, omega_vals = phi_psi_omega(p)
        # First residue: no phi or omega (no previous C/CA)
        assert np.isnan(phi_vals[0])
        assert np.isnan(omega_vals[0])
        # Last residue: no psi
        assert np.isnan(psi_vals[-1])

    def test_helix_phi_psi_in_helical_region(self) -> None:
        """The idealized helix fixture has |phi| ~ 60, |psi| ~ 45 — sign depends
        on the handedness chosen when generating the fixture. We just check
        the magnitudes here."""
        p = read_pdb(FIXTURES / "helix.pdb")
        phi_vals, psi_vals, _ = phi_psi_omega(p)
        valid_phi = phi_vals[~np.isnan(phi_vals)]
        valid_psi = psi_vals[~np.isnan(psi_vals)]
        assert np.median(np.abs(valid_phi)) == pytest.approx(60.0, abs=5.0)
        assert np.median(np.abs(valid_psi)) == pytest.approx(45.0, abs=5.0)

    def test_omega_near_180(self) -> None:
        """Omega for trans peptide bonds (~all of them) should be ~180°."""
        p = read_pdb(FIXTURES / "helix.pdb")
        _, _, omega_vals = phi_psi_omega(p)
        valid = omega_vals[~np.isnan(omega_vals)]
        # |omega| should be near 180
        np.testing.assert_allclose(np.abs(valid), 180.0, atol=5.0)

    def test_phi_psi_omega_individually(self) -> None:
        """phi/psi/omega convenience wrappers match phi_psi_omega outputs."""
        p = read_pdb(FIXTURES / "helix.pdb")
        all_phi, all_psi, all_omega = phi_psi_omega(p)
        np.testing.assert_array_equal(phi(p), all_phi, strict=False)
        np.testing.assert_array_equal(psi(p), all_psi, strict=False)
        np.testing.assert_array_equal(omega(p), all_omega, strict=False)

    def test_ramachandran(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        rama = ramachandran(p)
        assert rama.shape == (p.n_residues, 2)

    def test_empty_protein(self) -> None:
        p = Protein(AtomArray(0))
        phi_vals, psi_vals, omega_vals = phi_psi_omega(p)
        assert phi_vals.shape == (0,)
        assert psi_vals.shape == (0,)
        assert omega_vals.shape == (0,)
