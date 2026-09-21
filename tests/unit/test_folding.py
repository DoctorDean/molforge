"""Tests for :mod:`molforge.folding` — Entity and ComplexSpec types.

These are pure-Python dataclass tests; no heavy dependencies. They
exercise the validation invariants that catch user mistakes before
they hit the folding engine (where errors are opaque):

- Entity rejects bad combinations (ligand with sequence, polymer
  without sequence, copies <= 0, invalid alphabet characters, etc.)
- ComplexSpec rejects empty entity tuples and duplicate chain_ids
- Convenience constructors produce the expected shape
- assigned_chain_ids() handles explicit + auto + multi-copy mixes
- The chain-ID allocator's A-Z + AA-ZZ sequence works correctly
"""

from __future__ import annotations

import pytest

from molforge.folding import ComplexSpec, Entity, _index_to_chain_id

# ---------------------------------------------------------------------
# Entity construction + validation
# ---------------------------------------------------------------------


class TestEntityProtein:
    def test_basic(self) -> None:
        e = Entity(kind="protein", sequence="MKQH")
        assert e.kind == "protein"
        assert e.sequence == "MKQH"
        assert e.smiles is None
        assert e.ccd is None
        assert e.chain_id is None
        assert e.copies == 1
        assert e.name is None
        assert e.is_polymer is True
        assert e.is_ligand is False

    def test_normalized_sequence_strips_whitespace_and_upcases(self) -> None:
        e = Entity(kind="protein", sequence="m k qh\n")
        assert e.normalized_sequence() == "MKQH"

    def test_rejects_empty_sequence(self) -> None:
        with pytest.raises(ValueError, match="requires a non-empty"):
            Entity(kind="protein", sequence="")
        with pytest.raises(ValueError, match="requires a non-empty"):
            Entity(kind="protein", sequence="   ")

    def test_rejects_missing_sequence(self) -> None:
        with pytest.raises(ValueError, match="requires a non-empty"):
            Entity(kind="protein")

    def test_rejects_invalid_amino_acids(self) -> None:
        # B, J, O, U, X, Z are not in the standard 20.
        with pytest.raises(ValueError, match="invalid characters"):
            Entity(kind="protein", sequence="MKQZ")
        with pytest.raises(ValueError, match="invalid characters"):
            Entity(kind="protein", sequence="MK*QH")

    def test_rejects_ligand_fields(self) -> None:
        with pytest.raises(ValueError, match="takes 'sequence'"):
            Entity(kind="protein", sequence="MKQ", smiles="CCO")
        with pytest.raises(ValueError, match="takes 'sequence'"):
            Entity(kind="protein", sequence="MKQ", ccd="ATP")


class TestEntityDNA:
    def test_basic(self) -> None:
        e = Entity(kind="dna", sequence="ATCG")
        assert e.is_polymer
        assert e.normalized_sequence() == "ATCG"

    def test_accepts_lowercase(self) -> None:
        e = Entity(kind="dna", sequence="atcg")
        assert e.normalized_sequence() == "ATCG"

    def test_rejects_rna_bases(self) -> None:
        # U is RNA only.
        with pytest.raises(ValueError, match="invalid characters"):
            Entity(kind="dna", sequence="ATCGU")

    def test_rejects_protein_letters(self) -> None:
        with pytest.raises(ValueError, match="invalid characters"):
            Entity(kind="dna", sequence="MKQH")


class TestEntityRNA:
    def test_basic(self) -> None:
        e = Entity(kind="rna", sequence="AUCG")
        assert e.normalized_sequence() == "AUCG"

    def test_rejects_thymine(self) -> None:
        # T is DNA only.
        with pytest.raises(ValueError, match="invalid characters"):
            Entity(kind="rna", sequence="AUCGT")


class TestEntityLigand:
    def test_smiles(self) -> None:
        e = Entity(kind="ligand", smiles="CC(=O)O")
        assert e.is_ligand
        assert e.is_polymer is False
        assert e.smiles == "CC(=O)O"

    def test_ccd(self) -> None:
        e = Entity(kind="ligand", ccd="ATP")
        assert e.ccd == "ATP"

    def test_rejects_neither_smiles_nor_ccd(self) -> None:
        with pytest.raises(ValueError, match="requires exactly one"):
            Entity(kind="ligand")

    def test_rejects_both_smiles_and_ccd(self) -> None:
        with pytest.raises(ValueError, match="requires exactly one"):
            Entity(kind="ligand", smiles="CCO", ccd="ATP")

    def test_rejects_sequence(self) -> None:
        with pytest.raises(ValueError, match="takes 'smiles' or 'ccd'"):
            Entity(kind="ligand", sequence="MKQ")

    def test_rejects_empty_smiles(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Entity(kind="ligand", smiles="")

    def test_rejects_empty_ccd(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Entity(kind="ligand", ccd="")

    def test_rejects_overlong_ccd(self) -> None:
        with pytest.raises(ValueError, match=r"1.{1,3}5 characters"):
            Entity(kind="ligand", ccd="VERYLONGCCDCODE")

    def test_normalized_sequence_raises(self) -> None:
        e = Entity(kind="ligand", smiles="CCO")
        with pytest.raises(ValueError, match="only valid for polymer"):
            e.normalized_sequence()


class TestEntityCopies:
    def test_default(self) -> None:
        e = Entity(kind="protein", sequence="MKQ")
        assert e.copies == 1

    def test_homodimer(self) -> None:
        e = Entity(kind="protein", sequence="MKQ", copies=2)
        assert e.copies == 2

    def test_rejects_zero(self) -> None:
        with pytest.raises(ValueError, match="positive int"):
            Entity(kind="protein", sequence="MKQ", copies=0)

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="positive int"):
            Entity(kind="protein", sequence="MKQ", copies=-1)


class TestEntityChainId:
    def test_default(self) -> None:
        e = Entity(kind="protein", sequence="MKQ")
        assert e.chain_id is None

    def test_explicit(self) -> None:
        e = Entity(kind="protein", sequence="MKQ", chain_id="X")
        assert e.chain_id == "X"

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Entity(kind="protein", sequence="MKQ", chain_id="")

    def test_rejects_too_long(self) -> None:
        with pytest.raises(ValueError, match="longer than 4"):
            Entity(kind="protein", sequence="MKQ", chain_id="ABCDE")


class TestEntityKind:
    def test_rejects_unknown_kind(self) -> None:
        with pytest.raises(ValueError, match="must be one of"):
            Entity(kind="lipid", sequence="MKQ")  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# ComplexSpec
# ---------------------------------------------------------------------


class TestComplexSpec:
    def test_single_entity(self) -> None:
        spec = ComplexSpec(entities=(Entity(kind="protein", sequence="MKQ"),))
        assert len(spec.entities) == 1

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="at least one entity"):
            ComplexSpec(entities=())

    def test_rejects_duplicate_chain_ids(self) -> None:
        with pytest.raises(ValueError, match="duplicate chain_id"):
            ComplexSpec(
                entities=(
                    Entity(kind="protein", sequence="MKQ", chain_id="X"),
                    Entity(kind="protein", sequence="HIS", chain_id="X"),
                )
            )

    def test_allows_auto_assigned_to_coexist_with_explicit(self) -> None:
        # No duplicates because the auto-assigned entity has chain_id=None
        # at construction time. The conflict-check applies only to
        # explicitly-set chain_ids.
        ComplexSpec(
            entities=(
                Entity(kind="protein", sequence="MKQ"),  # auto-assign
                Entity(kind="protein", sequence="HIS", chain_id="X"),
            )
        )


# ---------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------


class TestFromProtein:
    def test_basic(self) -> None:
        spec = ComplexSpec.from_protein("MKQH")
        assert len(spec.entities) == 1
        e = spec.entities[0]
        assert e.kind == "protein"
        assert e.sequence == "MKQH"
        assert e.chain_id == "A"

    def test_custom_chain_id(self) -> None:
        spec = ComplexSpec.from_protein("MKQH", chain_id="X")
        assert spec.entities[0].chain_id == "X"


class TestProteinLigand:
    def test_smiles(self) -> None:
        spec = ComplexSpec.protein_ligand(
            protein_sequence="MKQ",
            ligand_smiles="CCO",
        )
        assert len(spec.entities) == 2
        p, lig = spec.entities
        assert p.kind == "protein"
        assert p.sequence == "MKQ"
        assert p.chain_id == "A"
        assert lig.kind == "ligand"
        assert lig.smiles == "CCO"
        assert lig.chain_id == "B"

    def test_ccd(self) -> None:
        spec = ComplexSpec.protein_ligand(
            protein_sequence="MKQ",
            ligand_ccd="ATP",
        )
        assert spec.entities[1].ccd == "ATP"

    def test_custom_chain_ids(self) -> None:
        spec = ComplexSpec.protein_ligand(
            protein_sequence="MKQ",
            ligand_smiles="CCO",
            protein_chain_id="H",
            ligand_chain_id="L",
        )
        assert spec.entities[0].chain_id == "H"
        assert spec.entities[1].chain_id == "L"

    def test_rejects_both_smiles_and_ccd(self) -> None:
        # The Entity validator catches this — the convenience
        # constructor passes both through and the Entity raises.
        with pytest.raises(ValueError, match="requires exactly one"):
            ComplexSpec.protein_ligand(
                protein_sequence="MKQ",
                ligand_smiles="CCO",
                ligand_ccd="ATP",
            )


# ---------------------------------------------------------------------
# Chain-ID assignment
# ---------------------------------------------------------------------


class TestAssignedChainIds:
    def test_single_entity_auto(self) -> None:
        spec = ComplexSpec(entities=(Entity(kind="protein", sequence="MKQ"),))
        assert spec.assigned_chain_ids() == [["A"]]

    def test_single_entity_explicit(self) -> None:
        spec = ComplexSpec(entities=(Entity(kind="protein", sequence="MKQ", chain_id="X"),))
        assert spec.assigned_chain_ids() == [["X"]]

    def test_multiple_entities_auto(self) -> None:
        spec = ComplexSpec(
            entities=(
                Entity(kind="protein", sequence="MKQ"),
                Entity(kind="dna", sequence="ATCG"),
                Entity(kind="ligand", smiles="CCO"),
            )
        )
        assert spec.assigned_chain_ids() == [["A"], ["B"], ["C"]]

    def test_homodimer_expands(self) -> None:
        spec = ComplexSpec(entities=(Entity(kind="protein", sequence="MKQ", copies=2),))
        assert spec.assigned_chain_ids() == [["A", "B"]]

    def test_mixed_explicit_and_auto_skips_claimed(self) -> None:
        spec = ComplexSpec(
            entities=(
                Entity(kind="protein", sequence="MKQ", copies=2),  # auto: A, B
                Entity(kind="protein", sequence="HIS", chain_id="X"),  # explicit X
                Entity(kind="ligand", smiles="O"),  # auto: skips A, B, X -> C
            )
        )
        assert spec.assigned_chain_ids() == [["A", "B"], ["X"], ["C"]]

    def test_explicit_with_multi_copy_extends_auto(self) -> None:
        # When user pins an explicit chain_id on a multi-copy entity,
        # the first copy gets the explicit ID; remaining copies
        # auto-assign around the claimed set.
        spec = ComplexSpec(
            entities=(Entity(kind="protein", sequence="MKQ", chain_id="X", copies=3),)
        )
        # X is claimed; auto-allocator sees claimed={"X"}, so its
        # first allocation is A.
        result = spec.assigned_chain_ids()
        assert result[0][0] == "X"
        assert len(result[0]) == 3
        # The auto-allocated copies must not duplicate X.
        assert "X" not in result[0][1:]


class TestChainIdIndex:
    """Direct tests on the _index_to_chain_id helper."""

    def test_single_letters(self) -> None:
        assert _index_to_chain_id(0) == "A"
        assert _index_to_chain_id(1) == "B"
        assert _index_to_chain_id(25) == "Z"

    def test_two_letters(self) -> None:
        assert _index_to_chain_id(26) == "AA"
        assert _index_to_chain_id(27) == "AB"
        assert _index_to_chain_id(51) == "AZ"
        assert _index_to_chain_id(52) == "BA"

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError):
            _index_to_chain_id(-1)


# ---------------------------------------------------------------------
# Source-inspection regression net
# ---------------------------------------------------------------------


class TestSourceInspection:
    def test_entity_is_frozen_dataclass(self) -> None:
        """Entity must be frozen (immutable) so it's safe to share
        by reference across multiple ComplexSpecs and across the
        Provenance chain."""
        e = Entity(kind="protein", sequence="MKQ")
        with pytest.raises((AttributeError, Exception)):  # FrozenInstanceError
            e.sequence = "HIS"  # type: ignore[misc]

    def test_complexspec_is_frozen_dataclass(self) -> None:
        """ComplexSpec must be frozen for the same reason as Entity."""
        spec = ComplexSpec.from_protein("MKQ")
        with pytest.raises((AttributeError, Exception)):
            spec.entities = ()  # type: ignore[misc]


class TestTemplatePolicy:
    """The shared value type. Engines translate it; it validates itself."""

    def test_none_mode(self) -> None:
        from molforge.folding import TemplatePolicy

        policy = TemplatePolicy.none()
        assert policy.mode == "none"
        assert policy.structures == ()
        assert policy.hits_file is None

    def test_from_structures_keeps_order(self) -> None:
        from molforge.folding import TemplatePolicy

        policy = TemplatePolicy.from_structures("b.cif", "a.cif")
        assert policy.structures == ("b.cif", "a.cif")

    def test_from_structures_accepts_path_objects(self) -> None:
        from pathlib import Path

        from molforge.folding import TemplatePolicy

        policy = TemplatePolicy.from_structures(Path("a.cif"), "b.cif")
        assert policy.structures == ("a.cif", "b.cif")
        assert all(isinstance(p, str) for p in policy.structures)

    def test_from_hits(self) -> None:
        from molforge.folding import TemplatePolicy

        assert TemplatePolicy.from_hits("hits.m8").hits_file == "hits.m8"

    def test_from_server(self) -> None:
        from molforge.folding import TemplatePolicy

        assert TemplatePolicy.from_server().mode == "server"

    def test_is_frozen(self) -> None:
        """A policy recorded in provenance must not change afterwards."""
        import dataclasses

        from molforge.folding import TemplatePolicy

        with pytest.raises(dataclasses.FrozenInstanceError):
            TemplatePolicy.none().mode = "server"  # type: ignore[misc]

    def test_unknown_mode_rejected(self) -> None:
        from molforge.folding import TemplatePolicy

        with pytest.raises(ValueError, match="mode must be"):
            TemplatePolicy(mode="apo")  # type: ignore[arg-type]

    def test_structures_mode_needs_paths(self) -> None:
        from molforge.folding import TemplatePolicy

        with pytest.raises(ValueError, match="at least one path"):
            TemplatePolicy(mode="structures")

    def test_hits_mode_needs_a_file(self) -> None:
        from molforge.folding import TemplatePolicy

        with pytest.raises(ValueError, match="needs a hits_file"):
            TemplatePolicy(mode="hits")

    @pytest.mark.parametrize("mode", ["none", "hits", "server"])
    def test_structures_rejected_outside_structures_mode(self, mode: str) -> None:
        from molforge.folding import TemplatePolicy

        kwargs = {"hits_file": "h.m8"} if mode == "hits" else {}
        with pytest.raises(ValueError, match="only meaningful for mode='structures'"):
            TemplatePolicy(mode=mode, structures=("a.cif",), **kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize("mode", ["none", "server"])
    def test_hits_file_rejected_outside_hits_mode(self, mode: str) -> None:
        from molforge.folding import TemplatePolicy

        with pytest.raises(ValueError, match="only meaningful for mode='hits'"):
            TemplatePolicy(mode=mode, hits_file="h.m8")  # type: ignore[arg-type]

    def test_to_provenance_is_json_native(self) -> None:
        import json

        from molforge.folding import TemplatePolicy

        for policy in (
            TemplatePolicy.none(),
            TemplatePolicy.from_structures("a.cif"),
            TemplatePolicy.from_hits("h.m8"),
            TemplatePolicy.from_server(),
        ):
            payload = policy.to_provenance()
            assert json.loads(json.dumps(payload)) == payload

    def test_to_provenance_omits_empty_fields(self) -> None:
        from molforge.folding import TemplatePolicy

        assert TemplatePolicy.none().to_provenance() == {"mode": "none"}

    def test_distinct_policies_serialize_distinctly(self) -> None:
        """Provenance is the cache key; collisions here would share
        results between an ablation and its control."""
        from molforge.folding import TemplatePolicy

        payloads = [
            TemplatePolicy.none().to_provenance(),
            TemplatePolicy.from_structures("a.cif").to_provenance(),
            TemplatePolicy.from_structures("a.cif", "b.cif").to_provenance(),
            TemplatePolicy.from_hits("h.m8").to_provenance(),
            TemplatePolicy.from_server().to_provenance(),
        ]
        assert len({repr(sorted(p.items())) for p in payloads}) == len(payloads)


class TestCoerceTemplatePolicy:
    def test_none_stays_none(self) -> None:
        from molforge.folding import _coerce_template_policy

        assert _coerce_template_policy(None) is None

    def test_string_shorthand(self) -> None:
        from molforge.folding import TemplatePolicy, _coerce_template_policy

        assert _coerce_template_policy("none") == TemplatePolicy.none()

    def test_policy_passes_through(self) -> None:
        from molforge.folding import TemplatePolicy, _coerce_template_policy

        policy = TemplatePolicy.from_server()
        assert _coerce_template_policy(policy) is policy

    def test_other_strings_rejected_with_a_pointer(self) -> None:
        from molforge.folding import _coerce_template_policy

        with pytest.raises(ValueError, match="from_structures"):
            _coerce_template_policy("apo")
