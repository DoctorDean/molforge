"""Tests for contact maps and residue-pair contacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from molforge.io import read_pdb
from molforge.structure import (
    contact_map,
    distance_map,
    residue_contacts,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "pdb"


class TestDistanceMap:
    def test_shape(self) -> None:
        p = read_pdb(FIXTURES / "dipeptide.pdb")
        d = distance_map(p, atom_choice="ca")
        # 3 residues total (ALA, GLY, HOH)
        assert d.shape == (3, 3)

    def test_diagonal_is_zero(self) -> None:
        p = read_pdb(FIXTURES / "dipeptide.pdb")
        d = distance_map(p, atom_choice="ca")
        np.testing.assert_allclose(np.diag(d), 0.0, atol=1e-5)

    def test_symmetric(self) -> None:
        p = read_pdb(FIXTURES / "dipeptide.pdb")
        d = distance_map(p)
        np.testing.assert_allclose(d, d.T, atol=1e-5)


class TestContactMap:
    def test_shape_and_dtype(self) -> None:
        p = read_pdb(FIXTURES / "dipeptide.pdb")
        cmap = contact_map(p, cutoff=5.0, atom_choice="ca")
        assert cmap.shape == (3, 3)
        assert cmap.dtype == np.bool_

    def test_diagonal_is_false(self) -> None:
        p = read_pdb(FIXTURES / "dipeptide.pdb")
        cmap = contact_map(p, cutoff=8.0, atom_choice="ca")
        assert not cmap.diagonal().any()

    def test_close_residues_in_contact(self) -> None:
        p = read_pdb(FIXTURES / "dipeptide.pdb")
        # ALA-GLY CAs are ~3.7 Å apart in a peptide bond
        cmap = contact_map(p, cutoff=5.0, atom_choice="ca")
        assert cmap[0, 1] is np.True_ or cmap[0, 1]
        assert cmap[1, 0] is np.True_ or cmap[1, 0]

    def test_exclude_neighbors(self) -> None:
        p = read_pdb(FIXTURES / "dipeptide.pdb")
        # Exclude immediate sequential neighbors
        cmap = contact_map(p, cutoff=8.0, atom_choice="ca", exclude_neighbors=1)
        # All off-diagonal-1 entries should be False
        n = cmap.shape[0]
        for i in range(n - 1):
            assert not cmap[i, i + 1]
            assert not cmap[i + 1, i]


class TestResidueContacts:
    def test_returns_sorted_tuples(self) -> None:
        p = read_pdb(FIXTURES / "dipeptide.pdb")
        contacts = residue_contacts(p, cutoff=8.0)
        assert all(isinstance(c, tuple) and len(c) == 3 for c in contacts)
        # Sorted by distance ascending
        distances = [c[2] for c in contacts]
        assert distances == sorted(distances)

    def test_cutoff_respected(self) -> None:
        p = read_pdb(FIXTURES / "dipeptide.pdb")
        loose = residue_contacts(p, cutoff=10.0)
        tight = residue_contacts(p, cutoff=2.0)
        assert len(tight) <= len(loose)

    def test_chain_filter(self) -> None:
        p = read_pdb(FIXTURES / "dipeptide.pdb")
        # Chain A has 2 residues; chain W has 1.
        within_a = residue_contacts(p, cutoff=20.0, chain_a="A", chain_b="A")
        # Within A: only one pair (residue 1, residue 2)
        assert len(within_a) == 1
        # Across chains: A's 2 residues vs W's 1 = 2 pairs
        cross = residue_contacts(p, cutoff=20.0, chain_a="A", chain_b="W")
        assert len(cross) == 2
