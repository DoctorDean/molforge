"""Tests for RMSD computations on Protein objects."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molforge.core import Protein
from molforge.io import read_pdb
from molforge.structure import rmsd, rmsd_per_residue, rmsd_raw

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "pdb"


class TestRmsdRaw:
    def test_zero_for_identical(self) -> None:
        a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        assert rmsd_raw(a, a) == 0.0

    def test_shape_mismatch_raises(self) -> None:
        a = np.zeros((3, 3))
        b = np.zeros((4, 3))
        with pytest.raises(ValueError, match="shape mismatch"):
            rmsd_raw(a, b)


class TestRmsdProtein:
    @pytest.fixture
    def protein(self) -> Protein:
        return read_pdb(FIXTURES / "tripeptide.pdb")

    def test_same_protein_is_zero(self, protein: Protein) -> None:
        assert rmsd(protein, protein, subset="ca") == pytest.approx(0.0, abs=1e-5)

    def test_translation_eliminated_by_alignment(self, protein: Protein) -> None:
        from copy import deepcopy

        moved = deepcopy(protein)
        # The + here is numpy broadcasting, not list concatenation.
        moved.atom_array.coords[:] = moved.atom_array.coords + np.array([100.0, 0, 0])
        assert rmsd(protein, moved, align=True) == pytest.approx(0.0, abs=1e-5)

    def test_translation_visible_without_alignment(self, protein: Protein) -> None:
        from copy import deepcopy

        moved = deepcopy(protein)
        moved.atom_array.coords[:] = moved.atom_array.coords + np.array([10.0, 0, 0])
        # Without alignment, RMSD = 10 (since every CA is offset by 10 A in x)
        assert rmsd(protein, moved, align=False) == pytest.approx(10.0, abs=1e-3)

    def test_atom_count_mismatch_raises(self, protein: Protein) -> None:
        # Build a structure with a different number of CAs than `protein`.
        from copy import deepcopy

        sub = deepcopy(protein)
        sub.atom_array.coords = sub.atom_array.coords[:5]
        for f in (
            "element",
            "atom_name",
            "residue_name",
            "residue_id",
            "insertion_code",
            "chain_id",
            "b_factor",
            "occupancy",
            "charge",
            "serial",
            "record_type",
            "entity_type",
            "altloc",
            "model_id",
        ):
            setattr(sub.atom_array, f, getattr(sub.atom_array, f)[:5])
        sub.atom_array._invalidate_cache()
        with pytest.raises(ValueError, match="aren't comparable"):
            rmsd(protein, sub, subset="ca")

    def test_subset_backbone(self, protein: Protein) -> None:
        # backbone = N + CA + C = 6 atoms in dipeptide (3 per residue, 2 residues)
        assert rmsd(protein, protein, subset="backbone") == pytest.approx(0.0, abs=1e-5)

    def test_unknown_subset_raises(self, protein: Protein) -> None:
        with pytest.raises(ValueError, match="unknown atom subset"):
            rmsd(protein, protein, subset="banana")  # type: ignore[arg-type]


class TestRmsdPerResidue:
    def test_zero_for_same_protein(self) -> None:
        p = read_pdb(FIXTURES / "tripeptide.pdb")
        per_res = rmsd_per_residue(p, p, subset="ca")
        assert per_res.shape == (3,)
        np.testing.assert_allclose(per_res, [0.0, 0.0, 0.0], atol=1e-5)

    def test_localizes_motion(self) -> None:
        from copy import deepcopy

        p = read_pdb(FIXTURES / "tripeptide.pdb")
        moved = deepcopy(p)
        # Shift only the middle (GLY) CA in z; alignment will spread the error.
        ca_indices = np.where(moved.atom_array.atom_name == "CA")[0]
        moved.atom_array.coords[ca_indices[1]] += [0, 0, 2.0]
        per_res = rmsd_per_residue(p, moved, subset="ca", align=True)
        assert per_res.shape == (3,)
        # The middle residue should have the largest per-residue RMSD
        assert per_res[1] == max(per_res)
