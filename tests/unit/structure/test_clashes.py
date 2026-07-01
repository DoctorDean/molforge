"""Tests for steric clash detection.

Two layers:

- **Synthetic** proteins built atom-by-atom, so exact distances drive
  exact expectations (clash / no-clash boundaries, bonded exclusion,
  the bonded-separation knob, hydrogens, tolerance, the Clash fields).
- **Real fixtures** as a coarse net: geometrically clean fixtures score
  zero, a handcrafted-imperfect one does not, and water contacts can be
  filtered out with ``remove_water()``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molforge.core import AtomArray, Protein
from molforge.io import read_pdb
from molforge.structure import (
    VDW_RADII,
    Clash,
    clash_score,
    find_clashes,
    has_clashes,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "pdb"


def _protein(
    coords: list[list[float]],
    elements: list[str],
    atom_names: list[str],
    residue_ids: list[int],
    *,
    chain_ids: list[str] | None = None,
    residue_names: list[str] | None = None,
) -> Protein:
    n = len(coords)
    arr = AtomArray(n)
    arr.coords[:] = np.array(coords, dtype=np.float32).reshape(n, 3)
    arr.element[:] = elements
    arr.atom_name[:] = atom_names
    arr.residue_id[:] = residue_ids
    arr.chain_id[:] = chain_ids or ["A"] * n
    arr.residue_name[:] = residue_names or ["ALA"] * n
    return Protein(arr, name="synthetic")


# ---------------------------------------------------------------------
# Core detection: clash vs no-clash
# ---------------------------------------------------------------------


class TestDetection:
    def test_nonbonded_pair_clashes(self) -> None:
        # Two carbons 2.5 Å apart in different residues: too far to be
        # a bond (>~2.0 Å), close enough to overlap (3.4 - 2.5 = 0.9).
        p = _protein([[0, 0, 0], [2.5, 0, 0]], ["C", "C"], ["CA", "CA"], [1, 5])
        clashes = find_clashes(p)
        assert len(clashes) == 1
        assert clashes[0].overlap == pytest.approx(0.9, abs=1e-6)

    def test_far_pair_does_not_clash(self) -> None:
        # 3.1 Å → overlap 0.3 < 0.4 tolerance.
        p = _protein([[0, 0, 0], [3.1, 0, 0]], ["C", "C"], ["CA", "CA"], [1, 5])
        assert find_clashes(p) == []

    def test_tolerance_controls_sensitivity(self) -> None:
        # 3.1 Å → overlap ~0.3: not a clash at 0.4, a clash at 0.2.
        p = _protein([[0, 0, 0], [3.1, 0, 0]], ["C", "C"], ["CA", "CA"], [1, 5])
        assert find_clashes(p, tolerance=0.4) == []
        assert len(find_clashes(p, tolerance=0.2)) == 1

    def test_element_radii_matter(self) -> None:
        # Same 2.8 Å separation: O-O (vdw 3.04) doesn't clash (overlap
        # 0.24) but S-S (vdw 3.60) does (overlap 0.80).
        oo = _protein([[0, 0, 0], [2.8, 0, 0]], ["O", "O"], ["O1", "O2"], [1, 5])
        ss = _protein([[0, 0, 0], [2.8, 0, 0]], ["S", "S"], ["S1", "S2"], [1, 5])
        assert find_clashes(oo) == []
        assert len(find_clashes(ss)) == 1

    def test_unknown_element_uses_fallback(self) -> None:
        # "XX" is absent from VDW_RADII → DEFAULT_VDW_RADIUS (1.70),
        # same as carbon, so 2.5 Å pair clashes.
        assert "XX" not in VDW_RADII
        p = _protein([[0, 0, 0], [2.5, 0, 0]], ["XX", "XX"], ["A", "B"], [1, 5])
        assert len(find_clashes(p)) == 1


# ---------------------------------------------------------------------
# Bonded exclusion (geometry-inferred)
# ---------------------------------------------------------------------


class TestBondedExclusion:
    def test_bonded_pair_excluded(self) -> None:
        # Two carbons 1.5 Å apart look like a covalent bond and are
        # excluded, despite a large raw overlap.
        p = _protein([[0, 0, 0], [1.5, 0, 0]], ["C", "C"], ["CA", "CB"], [1, 5])
        assert find_clashes(p) == []
        # Disabling exclusion surfaces the raw overlap.
        assert len(find_clashes(p, bonded_separation=0)) == 1

    def test_one_three_neighbour_excluded(self) -> None:
        # A-B-C chain: A-B and B-C bonded (1.5 Å), A-C ~2.45 Å apart.
        # A-C is a 1-3 neighbour, excluded at the default separation.
        chain = _protein(
            [[0, 0, 0], [1.5, 0, 0], [2.0, 1.414, 0]],
            ["C", "C", "C"],
            ["A", "B", "C"],
            [1, 1, 1],
        )
        assert find_clashes(chain) == []
        # separation=1 only excludes direct bonds, so the 1-3 A-C shows.
        assert len(find_clashes(chain, bonded_separation=1)) == 1
        # separation=0 reports every overlapping pair (A-B, B-C, A-C).
        assert len(find_clashes(chain, bonded_separation=0)) == 3

    def test_peptide_and_backbone_neighbours_not_flagged(self) -> None:
        # A clean tripeptide's backbone 1-2/1-3/1-4 contacts must not
        # register as clashes.
        p = read_pdb(FIXTURES / "tripeptide.pdb")
        assert find_clashes(p) == []


# ---------------------------------------------------------------------
# Hydrogens
# ---------------------------------------------------------------------


class TestHydrogens:
    def test_hydrogens_ignored_by_default(self) -> None:
        # Two H 1.5 Å apart (not bonded: H-H bond cut ~1.07 Å), overlap
        # 2.4 - 1.5 = 0.9. Ignored by default, seen with the flag.
        p = _protein([[0, 0, 0], [1.5, 0, 0]], ["H", "H"], ["H1", "H2"], [1, 5])
        assert find_clashes(p) == []
        assert len(find_clashes(p, include_hydrogens=True)) == 1

    def test_score_denominator_tracks_hydrogen_flag(self) -> None:
        # One heavy + many H: heavy-only denominator differs from all-atom.
        coords = [[0, 0, 0]] + [[10 + i, 0, 0] for i in range(9)]
        elements = ["C"] + ["H"] * 9
        names = ["CA"] + [f"H{i}" for i in range(9)]
        p = _protein(coords, elements, names, [1] * 10)
        # No clashes either way, but the call must not divide by zero /
        # must count atoms consistently.
        assert clash_score(p) == 0.0
        assert clash_score(p, include_hydrogens=True) == 0.0


# ---------------------------------------------------------------------
# Clash object + ordering
# ---------------------------------------------------------------------


class TestClashObject:
    def test_fields_populated(self) -> None:
        p = _protein(
            [[0, 0, 0], [2.5, 0, 0]],
            ["C", "O"],
            ["CA", "OXT"],
            [3, 7],
            residue_names=["LEU", "GLY"],
        )
        (clash,) = find_clashes(p)
        assert isinstance(clash, Clash)
        assert clash.atom_i == 0 and clash.atom_j == 1
        assert clash.atom_i < clash.atom_j
        assert clash.element_i == "C" and clash.element_j == "O"
        assert clash.distance == pytest.approx(2.5, abs=1e-5)
        assert clash.vdw_sum == pytest.approx(1.70 + 1.52, abs=1e-9)
        assert clash.overlap == pytest.approx(clash.vdw_sum - clash.distance, abs=1e-6)
        assert clash.residue_i == ("A", 3, "LEU")
        assert clash.residue_j == ("A", 7, "GLY")

    def test_sorted_worst_first(self) -> None:
        # Two independent clashing pairs with different overlaps.
        p = _protein(
            [[0, 0, 0], [2.8, 0, 0], [20, 0, 0], [22.2, 0, 0]],
            ["C", "C", "C", "C"],
            ["CA", "CA", "CA", "CA"],
            [1, 5, 9, 13],
        )
        clashes = find_clashes(p)
        assert len(clashes) == 2
        # 2.2 Å pair (overlap 1.2) ranks before the 2.8 Å pair (0.6).
        assert clashes[0].overlap > clashes[1].overlap
        assert clashes[0].distance == pytest.approx(2.2, abs=1e-5)

    def test_single_pair_not_double_counted(self) -> None:
        p = _protein([[0, 0, 0], [2.5, 0, 0]], ["C", "C"], ["CA", "CA"], [1, 5])
        assert len(find_clashes(p)) == 1


# ---------------------------------------------------------------------
# Aggregates + edge cases
# ---------------------------------------------------------------------


class TestAggregates:
    def test_clash_score_per_1000_atoms(self) -> None:
        # 1 clash over 2 heavy atoms → 500 per 1000.
        p = _protein([[0, 0, 0], [2.5, 0, 0]], ["C", "C"], ["CA", "CA"], [1, 5])
        assert clash_score(p) == pytest.approx(500.0)

    def test_has_clashes(self) -> None:
        clash = _protein([[0, 0, 0], [2.5, 0, 0]], ["C", "C"], ["CA", "CA"], [1, 5])
        clean = _protein([[0, 0, 0], [4.0, 0, 0]], ["C", "C"], ["CA", "CA"], [1, 5])
        assert has_clashes(clash) is True
        assert has_clashes(clean) is False

    def test_empty_and_single_atom(self) -> None:
        assert find_clashes(_protein([], [], [], [])) == []
        one = _protein([[0, 0, 0]], ["C"], ["CA"], [1])
        assert find_clashes(one) == []
        assert clash_score(_protein([], [], [], [])) == 0.0


# ---------------------------------------------------------------------
# Real fixtures
# ---------------------------------------------------------------------


class TestRealFixtures:
    @pytest.mark.parametrize("name", ["tripeptide.pdb", "helix.pdb", "ala_tripeptide_heavy.pdb"])
    def test_clean_fixtures_score_zero(self, name: str) -> None:
        assert find_clashes(read_pdb(FIXTURES / name)) == []

    def test_handcrafted_fixture_has_clashes(self) -> None:
        # real_small_protein.pdb is explicitly handcrafted with imperfect
        # (left-handed helix) geometry — it should register clashes.
        p = read_pdb(FIXTURES / "real_small_protein.pdb")
        assert has_clashes(p)
        assert clash_score(p) > 0.0

    def test_water_contacts_filterable(self) -> None:
        # The dipeptide's only clashes are short water contacts; dropping
        # water leaves a clean structure.
        p = read_pdb(FIXTURES / "dipeptide.pdb")
        assert has_clashes(p)
        assert find_clashes(p.remove_water()) == []
