"""Tests for the simplified Ramachandran classifier.

Two layers, mirroring the clash-detection tests:

- **Pure classifier** (:func:`ramachandran_type`) checked on canonical
  (φ, ψ) values — this is where classification *correctness* lives, and
  it needs no backbone geometry.
- **Protein plumbing** (:func:`classify_ramachandran` and friends)
  checked on real fixtures: termini are skipped, residue categories are
  detected, and the aggregates line up. The fixtures use a mirror-image
  (left-handed) NeRF convention, which is exactly why they make good
  *outlier* material.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molforge.io import read_pdb
from molforge.structure import (
    RamachandranResult,
    classify_ramachandran,
    phi_psi_omega,
    ramachandran_favored_fraction,
    ramachandran_outliers,
    ramachandran_type,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "pdb"


class TestRamachandranType:
    @pytest.mark.parametrize(
        ("phi", "psi", "expected"),
        [
            (-63.0, -43.0, "Favored"),  # right-handed α-helix
            (-49.0, -26.0, "Favored"),  # 3-10 helix
            (-135.0, 135.0, "Favored"),  # β-sheet
            (-75.0, 145.0, "Favored"),  # polyproline-II
            (60.0, 45.0, "Allowed"),  # left-handed α (rare but real)
            (120.0, -120.0, "Outlier"),  # mirror-β, disallowed for L
            (0.0, 0.0, "Outlier"),  # eclipsed, disallowed
        ],
    )
    def test_general_regions(self, phi: float, psi: float, expected: str) -> None:
        assert ramachandran_type(phi, psi) == expected

    def test_glycine_is_symmetric(self) -> None:
        # Glycine's map is point-symmetric: (φ,ψ) and (−φ,−ψ) match.
        for phi, psi in [(-63.0, -43.0), (-135.0, 135.0), (60.0, 45.0)]:
            assert ramachandran_type(phi, psi, category="Glycine") == ramachandran_type(
                -phi, -psi, category="Glycine"
            )

    def test_glycine_allows_left_handed_as_favored(self) -> None:
        # α-L is only Allowed for a general residue but Favored for Gly.
        assert ramachandran_type(60.0, 45.0, category="General") == "Allowed"
        assert ramachandran_type(60.0, 45.0, category="Glycine") == "Favored"

    def test_proline_restricts_phi(self) -> None:
        assert ramachandran_type(-63.0, -30.0, category="Proline") == "Favored"
        assert ramachandran_type(-63.0, 150.0, category="Proline") == "Favored"
        # Positive φ is impossible for the proline ring.
        assert ramachandran_type(60.0, 45.0, category="Proline") == "Outlier"

    def test_nonfinite_raises(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            ramachandran_type(float("nan"), 0.0)
        with pytest.raises(ValueError, match="finite"):
            ramachandran_type(0.0, float("inf"))


class TestClassifyProtein:
    def test_skips_undefined_termini(self) -> None:
        p = read_pdb(FIXTURES / "real_small_protein.pdb")
        phi, psi, _ = phi_psi_omega(p)
        n_defined = int(np.sum(np.isfinite(phi) & np.isfinite(psi)))
        results = classify_ramachandran(p)
        assert len(results) == n_defined
        # Every result has finite angles.
        assert all(np.isfinite(r.phi) and np.isfinite(r.psi) for r in results)

    def test_category_detection(self) -> None:
        p = read_pdb(FIXTURES / "real_small_protein.pdb")
        by_resname = {r.residue[2]: r.category for r in classify_ramachandran(p)}
        assert by_resname["GLY"] == "Glycine"
        assert by_resname["PRO"] == "Proline"
        assert by_resname["LEU"] == "General"

    def test_result_fields_and_consistency(self) -> None:
        p = read_pdb(FIXTURES / "real_small_protein.pdb")
        results = classify_ramachandran(p)
        assert results and all(isinstance(r, RamachandranResult) for r in results)
        for r in results:
            chain, resid, resname = r.residue
            assert isinstance(chain, str) and isinstance(resid, int) and isinstance(resname, str)
            # The stored classification agrees with the pure classifier.
            assert r.classification == ramachandran_type(r.phi, r.psi, category=r.category)

    def test_empty_protein(self) -> None:
        p = read_pdb(FIXTURES / "tripeptide.pdb").select(chain_id="Z")  # no atoms
        assert classify_ramachandran(p) == []


class TestAggregates:
    def test_outliers_are_a_subset(self) -> None:
        p = read_pdb(FIXTURES / "real_small_protein.pdb")
        results = classify_ramachandran(p)
        outliers = ramachandran_outliers(p)
        assert all(r.classification == "Outlier" for r in outliers)
        assert len(outliers) == sum(1 for r in results if r.classification == "Outlier")
        assert 0 < len(outliers) < len(results)

    def test_mirror_beta_fixture_all_outliers(self) -> None:
        # mini_beta_sheet.pdb sits at the mirror-β region (+120, −120),
        # disallowed for L-amino acids → every classifiable residue is an
        # outlier and the favored fraction is zero.
        p = read_pdb(FIXTURES / "mini_beta_sheet.pdb")
        results = classify_ramachandran(p)
        assert results
        assert all(r.classification == "Outlier" for r in results)
        assert ramachandran_favored_fraction(p) == 0.0

    def test_favored_fraction_math(self) -> None:
        p = read_pdb(FIXTURES / "real_small_protein.pdb")
        results = classify_ramachandran(p)
        favored = sum(1 for r in results if r.classification == "Favored")
        assert ramachandran_favored_fraction(p) == pytest.approx(favored / len(results))
        assert 0.0 <= ramachandran_favored_fraction(p) <= 1.0

    def test_favored_fraction_empty_is_one(self) -> None:
        # Nothing to fault → 1.0.
        empty = read_pdb(FIXTURES / "tripeptide.pdb").select(chain_id="Z")
        assert ramachandran_favored_fraction(empty) == 1.0
