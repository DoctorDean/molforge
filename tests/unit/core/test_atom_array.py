"""Tests for AtomArray — the canonical linear representation."""

from __future__ import annotations

import numpy as np
import pytest

from molforge.core import ATOM_FIELDS, AtomArray


def _make_dipeptide() -> AtomArray:
    """Tiny Ala-Gly array: 5 atoms in residue 1 (Ala), 4 in residue 2 (Gly).

    Same chain A; coordinates are placeholders.
    """
    aa = AtomArray(9)
    aa.atom_name[:] = ["N", "CA", "C", "O", "CB", "N", "CA", "C", "O"]
    aa.element[:] = ["N", "C", "C", "O", "C", "N", "C", "C", "O"]
    aa.residue_name[:5] = "ALA"
    aa.residue_name[5:] = "GLY"
    aa.residue_id[:5] = 1
    aa.residue_id[5:] = 2
    aa.chain_id[:] = "A"
    aa.entity_type[:] = "protein"
    aa.coords[:] = np.arange(27, dtype=np.float32).reshape(9, 3)
    return aa


class TestConstruction:
    def test_empty(self) -> None:
        aa = AtomArray(0)
        assert len(aa) == 0
        assert aa.n_residues == 0
        assert aa.n_chains == 0

    def test_with_size(self) -> None:
        aa = AtomArray(10)
        assert len(aa) == 10
        assert aa.coords.shape == (10, 3)
        assert aa.coords.dtype == np.float32

    def test_negative_size_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            AtomArray(-1)

    def test_default_field_values(self) -> None:
        aa = AtomArray(3)
        assert np.all(aa.occupancy == 1.0)
        assert np.all(aa.record_type == "ATOM")
        assert np.all(aa.entity_type == "protein")

    def test_from_dict(self) -> None:
        aa = AtomArray.from_dict(
            {
                "coords": np.zeros((2, 3), dtype=np.float32),
                "element": np.array(["C", "N"]),
            }
        )
        assert len(aa) == 2
        assert list(aa.element) == ["C", "N"]

    def test_from_dict_missing_coords_raises(self) -> None:
        with pytest.raises(KeyError, match="coords"):
            AtomArray.from_dict({"element": np.array(["C"])})

    def test_from_dict_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length"):
            AtomArray.from_dict(
                {
                    "coords": np.zeros((2, 3), dtype=np.float32),
                    "element": np.array(["C", "N", "O"]),
                }
            )

    def test_from_dict_unknown_field_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown field"):
            AtomArray.from_dict(
                {
                    "coords": np.zeros((1, 3), dtype=np.float32),
                    "bogus": np.array(["x"]),
                }
            )

    def test_schema_completeness(self) -> None:
        """Every advertised field must exist on a fresh AtomArray."""
        aa = AtomArray(1)
        for field in ATOM_FIELDS:
            assert hasattr(aa, field), f"missing field: {field}"


class TestBoundaries:
    def test_chain_starts_single_chain(self) -> None:
        aa = _make_dipeptide()
        np.testing.assert_array_equal(aa.chain_starts, [0])
        assert aa.n_chains == 1

    def test_residue_starts_dipeptide(self) -> None:
        aa = _make_dipeptide()
        np.testing.assert_array_equal(aa.residue_starts, [0, 5])
        assert aa.n_residues == 2

    def test_two_chains(self) -> None:
        aa = _make_dipeptide()
        # Flip the last 4 atoms to chain B
        aa.chain_id[5:] = "B"
        aa._invalidate_cache()
        np.testing.assert_array_equal(aa.chain_starts, [0, 5])
        assert aa.n_chains == 2

    def test_insertion_code_creates_residue_boundary(self) -> None:
        aa = AtomArray(4)
        aa.residue_id[:] = 1
        aa.chain_id[:] = "A"
        aa.insertion_code[:2] = ""
        aa.insertion_code[2:] = "A"
        np.testing.assert_array_equal(aa.residue_starts, [0, 2])

    def test_cache_invalidation(self) -> None:
        aa = _make_dipeptide()
        _ = aa.residue_starts  # populate cache
        aa.chain_id[5:] = "B"
        aa._invalidate_cache()
        # After invalidation, recompute should reflect new boundaries.
        np.testing.assert_array_equal(aa.chain_starts, [0, 5])


class TestSelection:
    def test_where_simple(self) -> None:
        aa = _make_dipeptide()
        mask = aa.where(atom_name="CA")
        assert mask.sum() == 2

    def test_where_combined(self) -> None:
        aa = _make_dipeptide()
        mask = aa.where(residue_name="ALA", atom_name="CA")
        assert mask.sum() == 1

    def test_where_with_list(self) -> None:
        aa = _make_dipeptide()
        mask = aa.where(atom_name=["N", "CA"])
        assert mask.sum() == 4

    def test_where_unknown_field_raises(self) -> None:
        aa = _make_dipeptide()
        with pytest.raises(KeyError, match="Unknown field"):
            aa.where(bogus="x")

    def test_select_returns_new_array(self) -> None:
        aa = _make_dipeptide()
        sub = aa.select(aa.where(atom_name="CA"))
        assert len(sub) == 2
        assert list(sub.atom_name) == ["CA", "CA"]
        # Original unchanged
        assert len(aa) == 9

    def test_select_shape_mismatch_raises(self) -> None:
        aa = _make_dipeptide()
        with pytest.raises(ValueError, match="shape"):
            aa.select(np.array([True, False]))


class TestSlicing:
    def test_slice_returns_atomarray(self) -> None:
        aa = _make_dipeptide()
        sub = aa[0:3]
        assert isinstance(sub, AtomArray)
        assert len(sub) == 3

    def test_int_index_returns_length_one(self) -> None:
        aa = _make_dipeptide()
        sub = aa[0]
        assert len(sub) == 1
        assert sub.atom_name[0] == "N"

    def test_fancy_index(self) -> None:
        aa = _make_dipeptide()
        sub = aa[np.array([0, 4, 8])]
        assert len(sub) == 3
        assert list(sub.atom_name) == ["N", "CB", "O"]


class TestConcatenation:
    def test_append(self) -> None:
        a = _make_dipeptide()
        b = _make_dipeptide()
        c = a.append(b)
        assert len(c) == 18

    def test_append_type_error(self) -> None:
        aa = _make_dipeptide()
        with pytest.raises(TypeError):
            aa.append([1, 2, 3])  # type: ignore[arg-type]


class TestIteration:
    def test_iter_residue_slices(self) -> None:
        aa = _make_dipeptide()
        slices = list(aa.iter_residue_slices())
        assert len(slices) == 2
        assert slices[0] == slice(0, 5)
        assert slices[1] == slice(5, 9)

    def test_iter_chain_slices(self) -> None:
        aa = _make_dipeptide()
        aa.chain_id[5:] = "B"
        aa._invalidate_cache()
        slices = list(aa.iter_chain_slices())
        assert len(slices) == 2
        assert slices[0] == slice(0, 5)
        assert slices[1] == slice(5, 9)
