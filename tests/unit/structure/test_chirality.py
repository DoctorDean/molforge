"""Tests for Cα chirality checks.

Synthetic residues built with a known handedness pin the L/D/Planar
classification and the plumbing (glycine skipped, CB required, global
Cα index, non-protein filter). The mirror-NeRF fixtures
(ala_tripeptide_heavy, real_small_protein) are genuine D-peptides and
serve as the real-data net.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molforge.core import AtomArray, Protein
from molforge.io import read_pdb
from molforge.structure import (
    ChiralityResult,
    ca_chirality,
    chirality_outliers,
    classify_chirality,
    has_chirality_outliers,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "pdb"

# A Cα centre with N, C, CB arranged so the signed volume is positive
# (the L / S configuration). Cα sits at the origin.
_CA = (0.0, 0.0, 0.0)
_N = (0.0, 1.45, 0.5)
_C = (-1.25, -0.72, 0.5)
_CB = (1.25, -0.72, 0.5)


def _residue(
    atoms: dict[str, tuple[float, float, float]],
    *,
    resid: int = 1,
    chain: str = "A",
    resname: str = "ALA",
    entity: str = "protein",
) -> Protein:
    names = list(atoms)
    coords = [atoms[n] for n in names]
    n = len(names)
    arr = AtomArray(n)
    arr.coords[:] = np.array(coords, dtype=np.float32).reshape(n, 3)
    arr.atom_name[:] = names
    arr.residue_id[:] = [resid] * n
    arr.chain_id[:] = [chain] * n
    arr.residue_name[:] = [resname] * n
    arr.element[:] = [nm[0] for nm in names]
    arr.entity_type[:] = [entity] * n
    return Protein(arr, name="synthetic")


def _l_residue(**kw: object) -> Protein:
    return _residue({"N": _N, "CA": _CA, "C": _C, "CB": _CB}, **kw)  # type: ignore[arg-type]


def _d_residue(**kw: object) -> Protein:
    # Swapping C and CB reflects the centre → D.
    return _residue({"N": _N, "CA": _CA, "C": _CB, "CB": _C}, **kw)  # type: ignore[arg-type]


class TestCaChirality:
    def test_l_d_planar(self) -> None:
        n, ca, c, cb = (np.array(x, dtype=np.float64) for x in (_N, _CA, _C, _CB))
        assert ca_chirality(n, ca, c, cb) == "L"
        # Reflection through the CA plane inverts handedness.
        assert ca_chirality(n * [1, 1, -1], ca, c * [1, 1, -1], cb * [1, 1, -1]) == "D"
        # Four coplanar points → degenerate.
        planar = (
            np.array([1.0, 0, 0]),
            np.array([0.0, 0, 0]),
            np.array([-1.0, 0, 0]),
            np.array([0.0, 1.0, 0]),
        )
        assert ca_chirality(*planar) == "Planar"

    def test_planar_tolerance(self) -> None:
        # A shallow but non-zero volume: Planar at a loose tolerance,
        # resolved at zero tolerance.
        n = np.array([1.0, 0.0, 0.0])
        ca = np.array([0.0, 0.0, 0.0])
        c = np.array([0.0, 1.0, 0.0])
        cb = np.array([0.0, 0.0, 0.05])  # tiny out-of-plane
        assert ca_chirality(n, ca, c, cb, planar_tolerance=0.1) == "Planar"
        assert ca_chirality(n, ca, c, cb, planar_tolerance=0.0) in {"L", "D"}


class TestClassifyProtein:
    def test_l_and_d_residues(self) -> None:
        (lr,) = classify_chirality(_l_residue())
        assert lr.configuration == "L" and lr.volume > 0
        (dr,) = classify_chirality(_d_residue())
        assert dr.configuration == "D" and dr.volume < 0

    def test_result_fields(self) -> None:
        p = _l_residue(resid=5, chain="B", resname="LEU")
        (r,) = classify_chirality(p)
        assert isinstance(r, ChiralityResult)
        assert r.residue == ("B", 5, "LEU")
        assert r.configuration == "L"

    def test_ca_index_is_global(self) -> None:
        # Two residues; the second's CA index must be offset correctly.
        arr = AtomArray(8)
        coords = [_N, _CA, _C, _CB, _N, _CA, _C, _CB]
        arr.coords[:] = np.array(coords, dtype=np.float32)
        arr.atom_name[:] = ["N", "CA", "C", "CB", "N", "CA", "C", "CB"]
        arr.residue_id[:] = [1, 1, 1, 1, 2, 2, 2, 2]
        arr.chain_id[:] = ["A"] * 8
        arr.residue_name[:] = ["ALA"] * 8
        arr.element[:] = ["N", "C", "C", "C", "N", "C", "C", "C"]
        arr.entity_type[:] = ["protein"] * 8
        results = classify_chirality(Protein(arr, name="two"))
        assert [r.ca_index for r in results] == [1, 5]

    def test_glycine_and_missing_cb_skipped(self) -> None:
        gly = _residue({"N": _N, "CA": _CA, "C": _C}, resname="GLY")
        assert classify_chirality(gly) == []

    def test_non_protein_skipped(self) -> None:
        lig = _l_residue(entity="ligand", resname="LIG")
        assert classify_chirality(lig) == []


class TestOutliers:
    def test_l_has_no_outliers(self) -> None:
        p = _l_residue()
        assert chirality_outliers(p) == []
        assert has_chirality_outliers(p) is False

    def test_d_is_outlier(self) -> None:
        p = _d_residue()
        outliers = chirality_outliers(p)
        assert len(outliers) == 1
        assert outliers[0].configuration == "D"
        assert has_chirality_outliers(p) is True

    def test_volume_sign_matches_config(self) -> None:
        for p in (_l_residue(), _d_residue()):
            for r in classify_chirality(p):
                assert (r.volume > 0) == (r.configuration == "L")


class TestRealFixtures:
    @pytest.mark.parametrize("name", ["ala_tripeptide_heavy.pdb", "real_small_protein.pdb"])
    def test_mirror_fixtures_are_d(self, name: str) -> None:
        # These fixtures are built with an inverted (mirror) convention,
        # so every classifiable Cα is D.
        p = read_pdb(FIXTURES / name)
        results = classify_chirality(p)
        assert results
        assert all(r.configuration == "D" for r in results)
        assert len(chirality_outliers(p)) == len(results)
