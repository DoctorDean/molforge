"""Tests for sequence mutation utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molforge.io import read_pdb
from molforge.sequence import (
    Mutation,
    apply_mutation,
    apply_mutations,
    mutate_protein,
    parse_mutations,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "pdb"


class TestMutationParse:
    def test_simple(self) -> None:
        m = Mutation.parse("A123V")
        assert m.wild_type == "A"
        assert m.position == 123
        assert m.mutant == "V"
        assert m.chain_id is None

    def test_with_chain_prefix(self) -> None:
        m = Mutation.parse("H:K42N")
        assert m.chain_id == "H"
        assert m.wild_type == "K"
        assert m.position == 42
        assert m.mutant == "N"

    def test_str_roundtrip(self) -> None:
        assert str(Mutation.parse("A123V")) == "A123V"
        assert str(Mutation.parse("L:K42N")) == "L:K42N"

    def test_bad_format_raises(self) -> None:
        with pytest.raises(ValueError, match="could not parse"):
            Mutation.parse("not a mutation")

    def test_unknown_wt_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown wild-type"):
            Mutation.parse("Z123A")

    def test_unknown_mutant_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown mutant"):
            Mutation.parse("A123Z")


class TestParseMutations:
    def test_slash_delimited(self) -> None:
        muts = parse_mutations("A123V/T56K/L99M")
        assert len(muts) == 3

    def test_comma_delimited(self) -> None:
        muts = parse_mutations("A123V, T56K, L99M")
        assert len(muts) == 3

    def test_whitespace_delimited(self) -> None:
        muts = parse_mutations("A123V T56K L99M")
        assert len(muts) == 3

    def test_empty(self) -> None:
        assert parse_mutations("") == []


class TestApplyMutation:
    def test_basic(self) -> None:
        assert apply_mutation("MKTV", "K2A") == "MATV"

    def test_str_form(self) -> None:
        assert apply_mutation("MKTV", "K2A") == "MATV"

    def test_object_form(self) -> None:
        m = Mutation(wild_type="K", position=2, mutant="A")
        assert apply_mutation("MKTV", m) == "MATV"

    def test_position_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            apply_mutation("MKTV", "K99A")

    def test_wt_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="wild-type mismatch"):
            apply_mutation("MKTV", "L2A")


class TestApplyMutations:
    def test_multi_string(self) -> None:
        assert apply_mutations("MKTV", "M1A/K2L") == "ALTV"

    def test_multi_iterable(self) -> None:
        muts = [Mutation.parse("M1A"), Mutation.parse("K2L")]
        assert apply_mutations("MKTV", muts) == "ALTV"


class TestMutateProtein:
    def test_changes_residue_name(self) -> None:
        p = read_pdb(FIXTURES / "tripeptide.pdb")
        # Tripeptide is ALA-GLY-VAL. Mutate Ala1 -> Cys.
        mutated = mutate_protein(p, "A1C", chain_id="A")
        # Original protein must be unchanged.
        assert p["A"][1].name == "ALA"
        # New protein has Cys at position 1.
        assert mutated["A"][1].name == "CYS"

    def test_mutation_with_chain_prefix(self) -> None:
        p = read_pdb(FIXTURES / "tripeptide.pdb")
        mutated = mutate_protein(p, "A:G2L")
        assert mutated["A"][2].name == "LEU"

    def test_missing_position_raises(self) -> None:
        p = read_pdb(FIXTURES / "tripeptide.pdb")
        with pytest.raises(ValueError, match="no residue at position"):
            mutate_protein(p, "A99V", chain_id="A")

    def test_wt_mismatch_raises(self) -> None:
        p = read_pdb(FIXTURES / "tripeptide.pdb")
        # Position 1 is ALA, not LEU; should reject.
        with pytest.raises(ValueError, match="wild-type mismatch"):
            mutate_protein(p, "L1C", chain_id="A")

    def test_defaults_to_first_chain(self) -> None:
        p = read_pdb(FIXTURES / "tripeptide.pdb")
        mutated = mutate_protein(p, "A1C")  # no chain_id given
        assert mutated["A"][1].name == "CYS"

    def test_atom_coordinates_preserved(self) -> None:
        # Sequence-only mutation should not move atoms.
        p = read_pdb(FIXTURES / "tripeptide.pdb")
        mutated = mutate_protein(p, "A1C", chain_id="A")
        np.testing.assert_allclose(mutated.atom_array.coords, p.atom_array.coords, atol=1e-6)
