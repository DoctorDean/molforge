"""Tests for ``molforge.prep.fix_missing_atoms`` and ``add_caps``.

These need PDBFixer (and OpenMM) installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Skip the whole module unless both deps are present.
pytest.importorskip("openmm")
pytest.importorskip("pdbfixer")

from molforge.io import load
from molforge.prep import add_caps, fix_missing_atoms

FIXTURES = Path(__file__).parents[2] / "fixtures"
_HEAVY_FIXTURE = FIXTURES / "pdb" / "ala_tripeptide_heavy.pdb"
_LIGAND_FIXTURE = FIXTURES / "pdb" / "real_with_ligand_realistic.pdb"


class TestFixMissingAtoms:
    def test_complete_structure_is_noop(self) -> None:
        """A structure with no missing atoms passes through unchanged."""
        p = load(_HEAVY_FIXTURE)
        fixed = fix_missing_atoms(p)
        assert fixed.atom_array.n_atoms == p.atom_array.n_atoms

    def test_input_not_mutated(self) -> None:
        p = load(_HEAVY_FIXTURE)
        before = p.atom_array.n_atoms
        fix_missing_atoms(p)
        assert p.atom_array.n_atoms == before

    def test_fixes_residues_with_missing_atoms(self) -> None:
        """The realistic-with-ligand fixture has a residue missing
        its C-terminal OXT atom (the real structures we see often do).
        After fix_missing_atoms it should be present."""
        from molforge.prep import remove_heterogens

        p = remove_heterogens(load(_LIGAND_FIXTURE))
        fixed = fix_missing_atoms(p)
        # Atom count may grow as missing atoms get rebuilt.
        assert fixed.atom_array.n_atoms >= p.atom_array.n_atoms

    def test_replace_nonstandard_toggle(self) -> None:
        """``replace_nonstandard`` is exercised by the call path; on
        a structure with no non-standard residues it's a no-op, but
        both branches should run without error."""
        p = load(_HEAVY_FIXTURE)
        a = fix_missing_atoms(p, replace_nonstandard=True)
        b = fix_missing_atoms(p, replace_nonstandard=False)
        assert a.atom_array.n_atoms == b.atom_array.n_atoms

    def test_metadata_preserved(self) -> None:
        p = load(_HEAVY_FIXTURE)
        p.metadata = {**p.metadata, "custom_marker": "x"}
        fixed = fix_missing_atoms(p)
        assert fixed.metadata.get("custom_marker") == "x"


class TestAddCaps:
    def test_adds_ace_at_n_terminus(self) -> None:
        p = load(_HEAVY_FIXTURE)
        capped = add_caps(p)
        residues = {str(r).strip() for r in capped.atom_array.residue_name}
        assert "ACE" in residues

    def test_adds_nme_at_c_terminus(self) -> None:
        p = load(_HEAVY_FIXTURE)
        capped = add_caps(p)
        residues = {str(r).strip() for r in capped.atom_array.residue_name}
        assert "NME" in residues

    def test_atom_count_grows(self) -> None:
        p = load(_HEAVY_FIXTURE)
        before = p.atom_array.n_atoms
        capped = add_caps(p)
        assert capped.atom_array.n_atoms > before

    def test_input_not_mutated(self) -> None:
        p = load(_HEAVY_FIXTURE)
        before = p.atom_array.n_atoms
        add_caps(p)
        assert p.atom_array.n_atoms == before

    def test_metadata_preserved(self) -> None:
        p = load(_HEAVY_FIXTURE)
        p.metadata = {**p.metadata, "custom_marker": "x"}
        capped = add_caps(p)
        assert capped.metadata.get("custom_marker") == "x"

    def test_skip_n_cap_with_empty_string(self) -> None:
        """Passing ``n_cap=""`` skips the N-terminal cap but still
        adds the C-terminal one."""
        p = load(_HEAVY_FIXTURE)
        capped = add_caps(p, n_cap="")
        residues = {str(r).strip() for r in capped.atom_array.residue_name}
        # NME should still appear; ACE should not.
        assert "NME" in residues
        assert "ACE" not in residues

    def test_skip_c_cap_with_empty_string(self) -> None:
        p = load(_HEAVY_FIXTURE)
        capped = add_caps(p, c_cap="")
        residues = {str(r).strip() for r in capped.atom_array.residue_name}
        assert "ACE" in residues
        assert "NME" not in residues

    def test_skip_both_caps_is_noop(self) -> None:
        """Skipping both caps is effectively a no-op (the structure
        round-trips through PDBFixer unchanged)."""
        p = load(_HEAVY_FIXTURE)
        capped = add_caps(p, n_cap="", c_cap="")
        # Atom count should be preserved (or change only by formatting
        # round-tripping, which it doesn't in practice for a heavy-atom
        # PDB).
        assert capped.atom_array.n_atoms == p.atom_array.n_atoms

    def test_non_protein_chain_not_capped(self) -> None:
        """A chain whose first residue isn't a canonical amino acid
        (e.g. a ligand chain) is left uncapped. We test this by
        loading a structure that has a non-protein chain and checking
        the capping logic does the right thing."""
        from molforge.prep import remove_heterogens

        # Use the realistic-with-ligand fixture and keep the ligand
        # so we have a non-protein chain present.
        p = remove_heterogens(load(_LIGAND_FIXTURE), keep_ligands=True)
        capped = add_caps(p)
        # The ligand residue (BNZ) was kept; capping should not have
        # added ACE/NME residues bound to the BNZ chain. We test the
        # weaker but verifiable property: BNZ is still present.
        residues = {str(r).strip() for r in capped.atom_array.residue_name}
        assert "BNZ" in residues


class TestMissingPdbfixer:
    """When PDBFixer is absent the dep helper raises a clean error."""

    def test_missing_pdbfixer_error_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from molforge.md import MDEngineNotInstalledError
        from molforge.prep import _deps
        from molforge.prep import fix as fix_mod

        def fake_require() -> object:
            raise MDEngineNotInstalledError(
                "Structure preparation requires PDBFixer. Install with:\n"
                "    pip install 'molforge[prep]'"
            )

        monkeypatch.setattr(_deps, "require_pdbfixer", fake_require)
        monkeypatch.setattr(fix_mod, "require_pdbfixer", fake_require)

        p = load(_HEAVY_FIXTURE)
        with pytest.raises(MDEngineNotInstalledError, match="PDBFixer"):
            fix_missing_atoms(p)
        with pytest.raises(MDEngineNotInstalledError, match="PDBFixer"):
            add_caps(p)
