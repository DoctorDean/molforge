"""Tests for ``molforge.prep.add_hydrogens``.

These need OpenMM installed (the underlying ``Modeller.addHydrogens``).
``pytest.importorskip`` handles the absent-OpenMM case cleanly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Skip the whole module when OpenMM isn't available.
pytest.importorskip("openmm")

from molforge.io import load
from molforge.prep import add_hydrogens

FIXTURES = Path(__file__).parents[2] / "fixtures"
# The ala_tripeptide_heavy.pdb fixture is a heavy-atom-only tripeptide
# with complete heavy atoms — the canonical test substrate for
# H-addition pipelines.
_HEAVY_FIXTURE = FIXTURES / "pdb" / "ala_tripeptide_heavy.pdb"


def _count_hydrogens(p: object) -> int:
    """Count hydrogen atoms on a Protein's AtomArray."""
    return sum(
        1
        for e in p.atom_array.element  # type: ignore[attr-defined]
        if str(e).strip() == "H"
    )


class TestAddHydrogens:
    def test_heavy_atom_input_gains_hydrogens(self) -> None:
        p = load(_HEAVY_FIXTURE)
        assert _count_hydrogens(p) == 0
        p_h = add_hydrogens(p)
        assert _count_hydrogens(p_h) > 0

    def test_atom_count_grows(self) -> None:
        p = load(_HEAVY_FIXTURE)
        before = p.atom_array.n_atoms
        p_h = add_hydrogens(p)
        assert p_h.atom_array.n_atoms > before

    def test_input_not_mutated(self) -> None:
        p = load(_HEAVY_FIXTURE)
        before = p.atom_array.n_atoms
        add_hydrogens(p)
        assert p.atom_array.n_atoms == before
        assert _count_hydrogens(p) == 0

    def test_idempotent_on_protonated_input(self) -> None:
        """Calling add_hydrogens on an already-protonated structure
        returns a structurally equivalent Protein — the atom count
        is stable across repeated calls."""
        p = load(_HEAVY_FIXTURE)
        p_h = add_hydrogens(p)
        p_hh = add_hydrogens(p_h)
        assert p_hh.atom_array.n_atoms == p_h.atom_array.n_atoms

    def test_pH_kwarg_accepted(self) -> None:
        """Both physiological pH 7.4 and a more acidic pH 5.0 yield
        valid protonated structures. (Histidine protonation flips
        around pH 6, so the geometry can differ; what we test here is
        just that the kwarg is accepted and produces a sane structure.)"""
        p = load(_HEAVY_FIXTURE)
        p_neutral = add_hydrogens(p, pH=7.4)
        p_acidic = add_hydrogens(p, pH=5.0)
        # Both should have hydrogens added; we don't assert equal counts
        # because protonation state differences across pH may change them.
        assert _count_hydrogens(p_neutral) > 0
        assert _count_hydrogens(p_acidic) > 0

    def test_force_field_alias_accepted(self) -> None:
        """The named-alias path through _FORCE_FIELD_FILES works."""
        p = load(_HEAVY_FIXTURE)
        p_h = add_hydrogens(p, force_field="amber14-all")
        assert _count_hydrogens(p_h) > 0

    def test_force_field_direct_xml_filename_accepted(self) -> None:
        """An XML filename not in the alias table is forwarded
        directly to OpenMM's ForceField — covers the "any XML"
        fallback documented in the docstring."""
        p = load(_HEAVY_FIXTURE)
        p_h = add_hydrogens(p, force_field="amber14-all.xml")
        assert _count_hydrogens(p_h) > 0

    def test_metadata_preserved(self) -> None:
        p = load(_HEAVY_FIXTURE)
        p.metadata = {**p.metadata, "custom_marker": "hello"}
        p_h = add_hydrogens(p)
        assert p_h.metadata.get("custom_marker") == "hello"


class TestMissingOpenMM:
    """When OpenMM is genuinely absent the dep helper raises a clean
    error. Tested by monkeypatching the require_openmm helper itself
    so we don't have to uninstall OpenMM for the test."""

    def test_missing_openmm_error_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from molforge.md import MDEngineNotInstalledError
        from molforge.prep import _deps

        def fake_require() -> tuple[object, object, object]:
            raise MDEngineNotInstalledError(
                "Structure preparation requires OpenMM. Install with:\n"
                "    pip install 'molforge[prep]'"
            )

        monkeypatch.setattr(_deps, "require_openmm", fake_require)
        # Also patch the protonate module's already-imported reference.
        from molforge.prep import protonate as protonate_mod

        monkeypatch.setattr(protonate_mod, "require_openmm", fake_require)

        p = load(_HEAVY_FIXTURE)
        with pytest.raises(MDEngineNotInstalledError, match="OpenMM"):
            add_hydrogens(p)
