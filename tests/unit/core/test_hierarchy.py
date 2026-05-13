"""Tests for the hierarchical views (Atom, Residue, Chain, Protein)."""

from __future__ import annotations

import numpy as np
import pytest

from molforge.core import Atom, AtomArray, Chain, Protein, Residue


def _make_complex() -> AtomArray:
    """Two chains, A (Ala-Gly) and B (just Trp), plus 1 water."""
    aa = AtomArray(13)
    # Chain A: Ala (5 atoms) + Gly (4 atoms) — atoms 0..8
    aa.atom_name[:9] = ["N", "CA", "C", "O", "CB", "N", "CA", "C", "O"]
    aa.element[:9] = ["N", "C", "C", "O", "C", "N", "C", "C", "O"]
    aa.residue_name[:5] = "ALA"
    aa.residue_name[5:9] = "GLY"
    aa.residue_id[:5] = 1
    aa.residue_id[5:9] = 2
    aa.chain_id[:9] = "A"
    # Chain B: Trp (3 atoms for brevity) — atoms 9..11
    aa.atom_name[9:12] = ["N", "CA", "C"]
    aa.element[9:12] = ["N", "C", "C"]
    aa.residue_name[9:12] = "TRP"
    aa.residue_id[9:12] = 10
    aa.chain_id[9:12] = "B"
    # Water — atom 12
    aa.atom_name[12] = "O"
    aa.element[12] = "O"
    aa.residue_name[12] = "HOH"
    aa.residue_id[12] = 100
    aa.chain_id[12] = "W"
    aa.entity_type[:12] = "protein"
    aa.entity_type[12] = "water"
    aa.record_type[12] = "HETATM"
    aa.coords[:] = np.arange(13 * 3, dtype=np.float32).reshape(13, 3)
    return aa


class TestAtom:
    def test_basic_access(self) -> None:
        aa = _make_complex()
        atom = Atom(aa, 1)  # CA of Ala
        assert atom.name == "CA"
        assert atom.element == "C"
        assert atom.index == 1

    def test_coord_view_is_mutable(self) -> None:
        aa = _make_complex()
        atom = Atom(aa, 0)
        atom.coord = np.array([99.0, 99.0, 99.0], dtype=np.float32)
        np.testing.assert_array_equal(aa.coords[0], [99.0, 99.0, 99.0])

    def test_setter_writes_through(self) -> None:
        aa = _make_complex()
        atom = Atom(aa, 0)
        atom.b_factor = 42.0
        assert aa.b_factor[0] == pytest.approx(42.0)

    def test_is_backbone(self) -> None:
        aa = _make_complex()
        assert Atom(aa, 1).is_backbone is True  # CA
        assert Atom(aa, 4).is_backbone is False  # CB

    def test_is_hetero(self) -> None:
        aa = _make_complex()
        assert Atom(aa, 12).is_hetero is True
        assert Atom(aa, 0).is_hetero is False

    def test_out_of_bounds_raises(self) -> None:
        aa = _make_complex()
        with pytest.raises(IndexError):
            Atom(aa, 99)

    def test_equality_and_hash(self) -> None:
        aa = _make_complex()
        a1 = Atom(aa, 0)
        a2 = Atom(aa, 0)
        a3 = Atom(aa, 1)
        assert a1 == a2
        assert a1 != a3
        assert hash(a1) == hash(a2)


class TestResidue:
    def test_identity(self) -> None:
        aa = _make_complex()
        res = Residue(aa, 0, 5)
        assert res.name == "ALA"
        assert res.seq_id == 1
        assert res.chain_id == "A"
        assert len(res) == 5

    def test_atom_lookup_by_name(self) -> None:
        aa = _make_complex()
        res = Residue(aa, 0, 5)
        ca = res["CA"]
        assert isinstance(ca, Atom)
        assert ca.name == "CA"

    def test_atom_lookup_missing_raises(self) -> None:
        aa = _make_complex()
        res = Residue(aa, 0, 5)
        with pytest.raises(KeyError, match="no atom named"):
            _ = res["ZZ"]

    def test_has_atom(self) -> None:
        aa = _make_complex()
        res = Residue(aa, 0, 5)
        assert res.has_atom("CA") is True
        assert res.has_atom("ZZ") is False

    def test_iteration(self) -> None:
        aa = _make_complex()
        res = Residue(aa, 0, 5)
        names = [a.name for a in res]
        assert names == ["N", "CA", "C", "O", "CB"]

    def test_one_letter_code(self) -> None:
        aa = _make_complex()
        assert Residue(aa, 0, 5).one_letter == "A"  # Ala
        assert Residue(aa, 5, 9).one_letter == "G"  # Gly
        assert Residue(aa, 12, 13).one_letter == "X"  # Water -> unknown

    def test_standard_aa_flag(self) -> None:
        aa = _make_complex()
        assert Residue(aa, 0, 5).is_standard_amino_acid is True
        assert Residue(aa, 12, 13).is_standard_amino_acid is False

    def test_water_flag(self) -> None:
        aa = _make_complex()
        assert Residue(aa, 12, 13).is_water is True
        assert Residue(aa, 0, 5).is_water is False

    def test_coords_shape(self) -> None:
        aa = _make_complex()
        res = Residue(aa, 0, 5)
        assert res.coords.shape == (5, 3)


class TestChain:
    def test_residues_listed_in_order(self) -> None:
        aa = _make_complex()
        ch = Chain(aa, 0, 9)
        residues = ch.residues
        assert len(residues) == 2
        assert [r.name for r in residues] == ["ALA", "GLY"]

    def test_lookup_by_seq_id(self) -> None:
        aa = _make_complex()
        ch = Chain(aa, 0, 9)
        assert ch[1].name == "ALA"
        assert ch[2].name == "GLY"

    def test_lookup_with_insertion_code(self) -> None:
        aa = AtomArray(2)
        aa.residue_id[:] = 1
        aa.chain_id[:] = "A"
        aa.insertion_code[0] = ""
        aa.insertion_code[1] = "A"
        aa.residue_name[0] = "ALA"
        aa.residue_name[1] = "GLY"
        aa.entity_type[:] = "protein"
        ch = Chain(aa, 0, 2)
        assert ch[(1, "")].name == "ALA"
        assert ch[(1, "A")].name == "GLY"

    def test_lookup_missing_raises(self) -> None:
        aa = _make_complex()
        ch = Chain(aa, 0, 9)
        with pytest.raises(KeyError, match="no residue"):
            _ = ch[99]

    def test_sequence(self) -> None:
        aa = _make_complex()
        ch_a = Chain(aa, 0, 9)
        ch_b = Chain(aa, 9, 12)
        assert ch_a.sequence == "AG"
        assert ch_b.sequence == "W"


class TestProtein:
    def test_construction_empty(self) -> None:
        p = Protein(name="test")
        assert p.n_atoms == 0
        assert p.n_chains == 0
        assert p.n_residues == 0

    def test_construction_from_array(self) -> None:
        aa = _make_complex()
        p = Protein(aa, name="1XXX")
        assert p.n_atoms == 13
        assert p.n_chains == 3  # A, B, water
        assert p.n_residues == 4  # Ala, Gly, Trp, HOH

    def test_chain_lookup(self) -> None:
        p = Protein(_make_complex())
        assert p["A"].chain_id == "A"
        assert p["B"].chain_id == "B"

    def test_chain_lookup_missing_raises(self) -> None:
        p = Protein(_make_complex())
        with pytest.raises(KeyError, match="no chain"):
            _ = p["Z"]

    def test_sequence_join(self) -> None:
        p = Protein(_make_complex())
        # Water chain is skipped; protein chains joined by "/"
        assert p.sequence == "AG/W"

    def test_per_chain_sequences(self) -> None:
        p = Protein(_make_complex())
        seqs = p.sequences()
        assert seqs == {"A": "AG", "B": "W"}

    def test_select(self) -> None:
        p = Protein(_make_complex())
        sub = p.select(chain_id="A")
        assert sub.n_atoms == 9
        assert sub.n_chains == 1

    def test_remove_water(self) -> None:
        p = Protein(_make_complex())
        dry = p.remove_water()
        assert dry.n_atoms == 12
        assert "W" not in [c.chain_id for c in dry.chains]

    def test_protein_only(self) -> None:
        p = Protein(_make_complex())
        prot = p.protein_only()
        assert prot.n_atoms == 12  # drops water
        assert all(c.chain_id in {"A", "B"} for c in prot.chains)

    def test_metadata_isolation(self) -> None:
        """Mutating the dict passed in must not affect the protein."""
        meta = {"resolution": 1.5}
        p = Protein(metadata=meta)
        meta["resolution"] = 999.0
        assert p.metadata["resolution"] == 1.5


class TestConsistency:
    """Cross-cutting tests that exercise hierarchical <-> linear sync."""

    def test_atom_mutation_propagates_to_array(self) -> None:
        p = Protein(_make_complex())
        ca = p["A"][1]["CA"]
        ca.b_factor = 77.0
        assert p.atom_array.b_factor[1] == pytest.approx(77.0)

    def test_array_mutation_visible_via_hierarchy(self) -> None:
        p = Protein(_make_complex())
        p.atom_array.b_factor[1] = 33.0
        assert p["A"][1]["CA"].b_factor == pytest.approx(33.0)

    def test_chain_residue_atom_counts_match_array(self) -> None:
        p = Protein(_make_complex())
        total_atoms = sum(len(r) for c in p.chains for r in c)
        assert total_atoms == p.n_atoms
