"""Tests for ``molforge.prep.prepare_for_md``.

Integration tests that exercise the full clean → fix → cap →
protonate pipeline. They need PDBFixer + OpenMM installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openmm")
pytest.importorskip("pdbfixer")

from molforge.io import load
from molforge.prep import prepare_for_md

FIXTURES = Path(__file__).parents[2] / "fixtures"
_HEAVY_FIXTURE = FIXTURES / "pdb" / "ala_tripeptide_heavy.pdb"
_LIGAND_FIXTURE = FIXTURES / "pdb" / "real_with_ligand_realistic.pdb"


def _count_hydrogens(p: object) -> int:
    return sum(
        1
        for e in p.atom_array.element  # type: ignore[attr-defined]
        if str(e).strip() == "H"
    )


class TestPrepareForMd:
    def test_simple_tripeptide_round_trip(self) -> None:
        """The all-defaults pipeline takes a heavy-atom tripeptide and
        produces a fully-prepared system: capped, hydrogenated."""
        p = load(_HEAVY_FIXTURE)
        ready = prepare_for_md(p)
        # Heavy-atom input had no hydrogens; output should have some.
        assert _count_hydrogens(ready) > 0
        # Caps should be present.
        residues = {str(r).strip() for r in ready.atom_array.residue_name}
        assert "ACE" in residues
        assert "NME" in residues

    def test_input_not_mutated(self) -> None:
        p = load(_HEAVY_FIXTURE)
        before = p.atom_array.n_atoms
        prepare_for_md(p)
        assert p.atom_array.n_atoms == before
        assert _count_hydrogens(p) == 0

    def test_skip_caps(self) -> None:
        p = load(_HEAVY_FIXTURE)
        ready = prepare_for_md(p, add_caps_to_termini=False)
        residues = {str(r).strip() for r in ready.atom_array.residue_name}
        assert "ACE" not in residues
        assert "NME" not in residues
        # Hydrogens should still be added.
        assert _count_hydrogens(ready) > 0

    def test_skip_hydrogens(self) -> None:
        p = load(_HEAVY_FIXTURE)
        ready = prepare_for_md(p, add_explicit_hydrogens=False)
        # Caps still added; no hydrogens added.
        residues = {str(r).strip() for r in ready.atom_array.residue_name}
        assert "ACE" in residues
        assert _count_hydrogens(ready) == 0

    def test_skip_caps_and_hydrogens_leaves_only_clean_plus_fix(self) -> None:
        """With caps and hydrogens off, prepare_for_md is just
        remove_heterogens + fix_missing_atoms."""
        p = load(_HEAVY_FIXTURE)
        ready = prepare_for_md(
            p,
            add_caps_to_termini=False,
            add_explicit_hydrogens=False,
        )
        residues = {str(r).strip() for r in ready.atom_array.residue_name}
        # Just the original ALA residues, no caps, no extra hydrogens.
        assert residues == {"ALA"}
        assert _count_hydrogens(ready) == 0

    def test_keep_water(self) -> None:
        """``keep_water=True`` propagates through to remove_heterogens."""
        p = load(_LIGAND_FIXTURE)
        # Don't run the full pipeline — fix/cap/protonate won't like
        # the ligand. Test the heterogen-removal stage of the chain.
        ready = prepare_for_md(
            p,
            keep_water=True,
            keep_ligands=True,  # so we don't error on the ligand later
            add_caps_to_termini=False,
            add_explicit_hydrogens=False,
        )
        residues = {str(r).strip() for r in ready.atom_array.residue_name}
        assert "HOH" in residues

    def test_pH_kwarg_forwarded(self) -> None:
        """The pH kwarg reaches add_hydrogens. We just check the call
        succeeds with a non-default pH."""
        p = load(_HEAVY_FIXTURE)
        ready = prepare_for_md(p, pH=5.0)
        assert _count_hydrogens(ready) > 0

    def test_metadata_preserved(self) -> None:
        p = load(_HEAVY_FIXTURE)
        p.metadata = {**p.metadata, "from": "test_pipeline"}
        ready = prepare_for_md(p)
        assert ready.metadata.get("from") == "test_pipeline"

    def test_explicit_keep_allowlist(self) -> None:
        """An explicit residue name in ``keep`` survives even when
        category toggles are off."""

        p = load(_LIGAND_FIXTURE)
        ready = prepare_for_md(
            p,
            keep={"BNZ"},
            add_caps_to_termini=False,
            add_explicit_hydrogens=False,
        )
        residues = {str(r).strip() for r in ready.atom_array.residue_name}
        assert "BNZ" in residues
