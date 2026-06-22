"""Tests for molforge.prep.remove_heterogens.

remove_heterogens is pure-Python — no PDBFixer / OpenMM needed, so
these tests run everywhere with no dep guard.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molforge.io import load
from molforge.prep import remove_heterogens

FIXTURES = Path(__file__).parents[2] / "fixtures"
_LIGAND_FIXTURE = FIXTURES / "pdb" / "real_with_ligand_realistic.pdb"


class TestRemoveHeterogens:
    def test_drops_water_by_default(self) -> None:
        p = load(_LIGAND_FIXTURE)
        clean = remove_heterogens(p)
        residues = {str(r).strip() for r in clean.atom_array.residue_name}
        assert "HOH" not in residues

    def test_drops_ligand_by_default(self) -> None:
        p = load(_LIGAND_FIXTURE)
        clean = remove_heterogens(p)
        residues = {str(r).strip() for r in clean.atom_array.residue_name}
        # BNZ is the benzene ligand in the fixture.
        assert "BNZ" not in residues

    def test_drops_ions_by_default(self) -> None:
        p = load(_LIGAND_FIXTURE)
        clean = remove_heterogens(p)
        residues = {str(r).strip() for r in clean.atom_array.residue_name}
        assert "ZN" not in residues

    def test_keeps_canonical_amino_acids(self) -> None:
        p = load(_LIGAND_FIXTURE)
        clean = remove_heterogens(p)
        residues = {str(r).strip() for r in clean.atom_array.residue_name}
        # The fixture has these canonical residues — all should survive.
        for aa in {"ALA", "CYS", "GLU", "HIS", "LEU", "LYS", "MET", "PHE"}:
            assert aa in residues, f"{aa} was dropped"

    def test_keep_water_brings_water_back(self) -> None:
        p = load(_LIGAND_FIXTURE)
        with_water = remove_heterogens(p, keep_water=True)
        residues = {str(r).strip() for r in with_water.atom_array.residue_name}
        assert "HOH" in residues

    def test_keep_ions_brings_zinc_back(self) -> None:
        p = load(_LIGAND_FIXTURE)
        with_ions = remove_heterogens(p, keep_ions=True)
        residues = {str(r).strip() for r in with_ions.atom_array.residue_name}
        assert "ZN" in residues

    def test_keep_ligands_brings_ligand_back(self) -> None:
        p = load(_LIGAND_FIXTURE)
        with_lig = remove_heterogens(p, keep_ligands=True)
        residues = {str(r).strip() for r in with_lig.atom_array.residue_name}
        assert "BNZ" in residues

    def test_explicit_keep_overrides_default_drop(self) -> None:
        """The ``keep`` allow-list keeps a residue by name even when
        none of the category toggles cover it."""
        p = load(_LIGAND_FIXTURE)
        named = remove_heterogens(p, keep={"BNZ"})
        residues = {str(r).strip() for r in named.atom_array.residue_name}
        assert "BNZ" in residues
        # Other heterogens are still dropped.
        assert "HOH" not in residues
        assert "ZN" not in residues

    def test_explicit_keep_is_case_insensitive(self) -> None:
        p = load(_LIGAND_FIXTURE)
        named = remove_heterogens(p, keep={"bnz"})
        residues = {str(r).strip() for r in named.atom_array.residue_name}
        assert "BNZ" in residues

    def test_input_not_mutated(self) -> None:
        p = load(_LIGAND_FIXTURE)
        n_before = p.atom_array.n_atoms
        remove_heterogens(p)
        assert p.atom_array.n_atoms == n_before

    def test_empty_protein_handled(self) -> None:
        """An empty :class:`Protein` survives the filter without error."""
        from molforge.core import AtomArray, Protein

        empty = Protein(AtomArray(0))
        clean = remove_heterogens(empty)
        assert clean.atom_array.n_atoms == 0

    def test_metadata_preserved(self) -> None:
        p = load(_LIGAND_FIXTURE)
        # Stash a marker in metadata so we can check it survives.
        p.metadata = {**p.metadata, "custom_key": "marker"}
        clean = remove_heterogens(p)
        assert clean.metadata.get("custom_key") == "marker"

    @pytest.mark.parametrize(
        "water_alias",
        ["HOH", "WAT", "H2O", "SOL", "TIP3"],
    )
    def test_recognizes_multiple_water_aliases(self, water_alias: str) -> None:
        """Different programs emit different water residue names; the
        default filter must drop them all."""

        from molforge.core import AtomArray, Protein

        arr = AtomArray(2)
        arr.element[:] = ["O", "H"]
        arr.atom_name[:] = ["O", "H1"]
        arr.residue_name[:] = water_alias[:3]
        arr.residue_id[:] = [1, 1]
        arr.chain_id[:] = "W"
        arr.record_type[:] = "HETATM"
        arr.entity_type[:] = "water"
        prot = Protein(arr)
        clean = remove_heterogens(prot)
        assert clean.atom_array.n_atoms == 0
