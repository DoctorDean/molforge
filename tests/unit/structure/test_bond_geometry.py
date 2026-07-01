"""Tests for backbone bond-length validation.

Synthetic residues with atoms placed at exact distances pin the z-score
maths, the threshold, the CA-CB and peptide-bond handling, and the
entity filter. Real fixtures act as a coarse net: ideal-geometry
fixtures score zero, geometrically-loose ones flag outliers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molforge.core import AtomArray, Protein
from molforge.io import read_pdb
from molforge.structure import (
    IDEAL_BOND_LENGTHS,
    BondLengthOutlier,
    bond_length_rmsd,
    check_bond_lengths,
    has_bond_length_outliers,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "pdb"


def _protein(residues: list[dict]) -> Protein:
    """Build a protein from residue specs.

    Each residue is a dict with keys ``resid``, optionally ``chain``
    (default "A") and ``resname`` (default "ALA"), and ``atoms`` mapping
    atom name -> (x, y, z).
    """
    names: list[str] = []
    coords: list[tuple[float, float, float]] = []
    resids: list[int] = []
    chains: list[str] = []
    resnames: list[str] = []
    for r in residues:
        for name, xyz in r["atoms"].items():
            names.append(name)
            coords.append(xyz)
            resids.append(r["resid"])
            chains.append(r.get("chain", "A"))
            resnames.append(r.get("resname", "ALA"))
    n = len(names)
    arr = AtomArray(n)
    arr.coords[:] = np.array(coords, dtype=np.float32).reshape(n, 3)
    arr.atom_name[:] = names
    arr.residue_id[:] = resids
    arr.chain_id[:] = chains
    arr.residue_name[:] = resnames
    arr.element[:] = [nm[0] for nm in names]
    arr.entity_type[:] = "protein"
    return Protein(arr, name="synthetic")


def _bond_residue(bond: str, length: float, **kw: object) -> Protein:
    """One residue holding just the two atoms of ``bond`` at ``length``."""
    a, b = bond.split("-")
    return _protein([{"resid": 1, "atoms": {a: (0.0, 0.0, 0.0), b: (length, 0.0, 0.0)}, **kw}])


class TestZScore:
    def test_ideal_length_is_clean(self) -> None:
        ideal, _ = IDEAL_BOND_LENGTHS["N-CA"]
        assert check_bond_lengths(_bond_residue("N-CA", ideal)) == []

    def test_five_sigma_is_outlier(self) -> None:
        ideal, sigma = IDEAL_BOND_LENGTHS["CA-C"]
        p = _bond_residue("CA-C", ideal + 5 * sigma)
        (o,) = check_bond_lengths(p)
        assert o.bond == "CA-C"
        assert o.z_score == pytest.approx(5.0, abs=1e-4)
        assert o.deviation == pytest.approx(5 * sigma, abs=1e-4)
        assert o.ideal == ideal and o.sigma == sigma
        assert o.length == pytest.approx(ideal + 5 * sigma, abs=1e-4)

    def test_negative_deviation_reported(self) -> None:
        ideal, sigma = IDEAL_BOND_LENGTHS["C-O"]
        (o,) = check_bond_lengths(_bond_residue("C-O", ideal - 6 * sigma))
        assert o.z_score == pytest.approx(-6.0, abs=1e-4)
        assert o.deviation < 0

    def test_threshold(self) -> None:
        ideal, sigma = IDEAL_BOND_LENGTHS["N-CA"]
        p = _bond_residue("N-CA", ideal + 4.5 * sigma)
        assert len(check_bond_lengths(p, max_z=4.0)) == 1
        assert check_bond_lengths(p, max_z=5.0) == []

    def test_atom_indices_and_labels(self) -> None:
        ideal, sigma = IDEAL_BOND_LENGTHS["N-CA"]
        p = _protein(
            [{"resid": 7, "chain": "B", "resname": "LEU",
              "atoms": {"N": (0.0, 0.0, 0.0), "CA": (ideal + 8 * sigma, 0.0, 0.0)}}]
        )
        (o,) = check_bond_lengths(p)
        assert isinstance(o, BondLengthOutlier)
        assert (o.atom_i, o.name_i) == (0, "N")
        assert (o.atom_j, o.name_j) == (1, "CA")
        assert o.residue_i == ("B", 7, "LEU") == o.residue_j


class TestCbAndPeptide:
    def test_cb_toggle(self) -> None:
        ideal, sigma = IDEAL_BOND_LENGTHS["CA-CB"]
        p = _protein(
            [{"resid": 1, "atoms": {"CA": (0.0, 0.0, 0.0), "CB": (ideal + 6 * sigma, 0.0, 0.0)}}]
        )
        assert len(check_bond_lengths(p, include_cb=True)) == 1
        assert check_bond_lengths(p, include_cb=False) == []

    def test_peptide_bond_consecutive(self) -> None:
        ideal, sigma = IDEAL_BOND_LENGTHS["C-N"]
        bad = ideal + 7 * sigma
        p = _protein(
            [
                {"resid": 1, "atoms": {"C": (0.0, 0.0, 0.0)}},
                {"resid": 2, "atoms": {"N": (bad, 0.0, 0.0)}},
            ]
        )
        (o,) = check_bond_lengths(p)
        assert o.bond == "C-N"
        assert o.residue_i[1] == 1 and o.residue_j[1] == 2

    def test_peptide_bond_skipped_across_chain(self) -> None:
        ideal, sigma = IDEAL_BOND_LENGTHS["C-N"]
        bad = ideal + 7 * sigma
        p = _protein(
            [
                {"resid": 1, "chain": "A", "atoms": {"C": (0.0, 0.0, 0.0)}},
                {"resid": 2, "chain": "B", "atoms": {"N": (bad, 0.0, 0.0)}},
            ]
        )
        assert check_bond_lengths(p) == []

    def test_peptide_bond_skipped_across_gap(self) -> None:
        # Non-consecutive residue numbers = a gap, not a bond.
        ideal, sigma = IDEAL_BOND_LENGTHS["C-N"]
        bad = ideal + 7 * sigma
        p = _protein(
            [
                {"resid": 10, "atoms": {"C": (0.0, 0.0, 0.0)}},
                {"resid": 25, "atoms": {"N": (bad, 0.0, 0.0)}},
            ]
        )
        assert check_bond_lengths(p) == []


class TestAggregatesAndFilters:
    def test_sorted_worst_first(self) -> None:
        ni, ns = IDEAL_BOND_LENGTHS["N-CA"]
        ci, cs = IDEAL_BOND_LENGTHS["CA-C"]
        p = _protein(
            [
                {"resid": 1, "atoms": {
                    "N": (0.0, 0.0, 0.0),
                    "CA": (ni + 5 * ns, 0.0, 0.0),
                    "C": (ni + 5 * ns, ci + 9 * cs, 0.0),
                }}
            ]
        )
        outs = check_bond_lengths(p)
        assert len(outs) == 2
        assert abs(outs[0].z_score) >= abs(outs[1].z_score)

    def test_rmsd_zero_for_ideal(self) -> None:
        ni, _ = IDEAL_BOND_LENGTHS["N-CA"]
        assert bond_length_rmsd(_bond_residue("N-CA", ni)) == pytest.approx(0.0, abs=1e-6)

    def test_rmsd_empty(self) -> None:
        empty = _protein([])
        assert bond_length_rmsd(empty) == 0.0

    def test_has_bond_length_outliers(self) -> None:
        ideal, sigma = IDEAL_BOND_LENGTHS["N-CA"]
        assert has_bond_length_outliers(_bond_residue("N-CA", ideal + 9 * sigma)) is True
        assert has_bond_length_outliers(_bond_residue("N-CA", ideal)) is False

    def test_non_protein_atoms_ignored(self) -> None:
        # A ligand with N/CA-named atoms at a wild distance must not be
        # read as backbone.
        ideal, sigma = IDEAL_BOND_LENGTHS["N-CA"]
        arr = AtomArray(2)
        arr.coords[:] = np.array([[0, 0, 0], [ideal + 20 * sigma, 0, 0]], dtype=np.float32)
        arr.atom_name[:] = ["N", "CA"]
        arr.residue_id[:] = [1, 1]
        arr.chain_id[:] = ["X", "X"]
        arr.residue_name[:] = ["LIG", "LIG"]
        arr.element[:] = ["N", "C"]
        arr.entity_type[:] = "ligand"
        assert check_bond_lengths(Protein(arr, name="lig")) == []


class TestRealFixtures:
    @pytest.mark.parametrize("name", ["helix.pdb", "real_small_protein.pdb"])
    def test_ideal_geometry_fixtures_clean(self, name: str) -> None:
        # These fixtures were built with standard bond lengths — clean,
        # even though real_small_protein is a Ramachandran mess.
        p = read_pdb(FIXTURES / name)
        assert check_bond_lengths(p) == []
        assert bond_length_rmsd(p) < 0.01

    def test_loose_fixture_has_outliers(self) -> None:
        p = read_pdb(FIXTURES / "tripeptide.pdb")
        assert has_bond_length_outliers(p)
        assert bond_length_rmsd(p) > 0.01
