"""Tests for the Amber ``MMPBSA.py`` results parser.

Driven by a real-shape ``FINAL_RESULTS_MMPBSA.dat`` fixture with clean,
exactly-assertable numbers, plus small inline snippets for the tricky
bits (label anchoring, missing sections/rows).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molforge.core import AtomArray, Protein
from molforge.wrappers.freeenergy import (
    build_mmpbsa_input,
    parse_mmpbsa_dat,
    selection_to_amber_mask,
)

FIXTURE = (
    Path(__file__).resolve().parents[2] / "fixtures" / "freeenergy" / "FINAL_RESULTS_MMPBSA.dat"
)


@pytest.fixture
def dat() -> str:
    return FIXTURE.read_text()


class TestGeneralizedBorn:
    def test_delta_g_and_uncertainty_from_delta_total(self, dat: str) -> None:
        r = parse_mmpbsa_dat(dat)  # gb default
        assert r.method == "MM/GBSA"
        assert r.delta_g == pytest.approx(-21.0)
        assert r.uncertainty == pytest.approx(0.7)  # Std. Err. of Mean, not Std. Dev.

    def test_components(self, dat: str) -> None:
        c = parse_mmpbsa_dat(dat).components
        assert c is not None
        assert c.vdw == pytest.approx(-45.0)
        assert c.electrostatic == pytest.approx(-30.0)
        assert c.polar_solvation == pytest.approx(60.0)  # EGB
        assert c.nonpolar_solvation == pytest.approx(-6.0)  # ESURF
        assert c.entropy is None  # entropy section not parsed yet

    def test_reads_differences_not_complex_block(self, dat: str) -> None:
        # The Complex block has VDWAALS -900; the Differences block -45.
        assert parse_mmpbsa_dat(dat).components.vdw == pytest.approx(-45.0)

    def test_enthalpy_reconstructs_delta_total(self, dat: str) -> None:
        r = parse_mmpbsa_dat(dat)
        assert r.components.enthalpy == pytest.approx(r.delta_g)

    def test_metadata(self, dat: str) -> None:
        r = parse_mmpbsa_dat(dat)
        assert r.metadata["solvent_model"] == "gb"
        assert r.metadata["n_frames"] == 50
        assert r.metadata["delta_total_std_dev"] == pytest.approx(7.0)


class TestPoissonBoltzmann:
    def test_selects_pb_section(self, dat: str) -> None:
        r = parse_mmpbsa_dat(dat, solvent_model="pb")
        assert r.method == "MM/PBSA"
        # PB DELTA TOTAL is -24, distinct from GB's -21.
        assert r.delta_g == pytest.approx(-24.0)
        assert r.uncertainty == pytest.approx(0.72)

    def test_nonpolar_sums_enpolar_and_edisper(self, dat: str) -> None:
        c = parse_mmpbsa_dat(dat, solvent_model="pb").components
        assert c is not None
        assert c.polar_solvation == pytest.approx(55.0)  # EPB
        assert c.nonpolar_solvation == pytest.approx(-4.0)  # -8 + 4
        assert c.enthalpy == pytest.approx(-24.0)

    def test_case_insensitive_model(self, dat: str) -> None:
        assert parse_mmpbsa_dat(dat, solvent_model="PB").method == "MM/PBSA"
        assert parse_mmpbsa_dat(dat, solvent_model="Gb").method == "MM/GBSA"


class TestRobustness:
    _GB_ONLY = """GENERALIZED BORN:

Differences (Complex - Receptor - Ligand):
Energy Component            Average              Std. Dev.   Std. Err. of Mean
-------------------------------------------------------------------------------
VDWAALS                     -10.0000     1.0000    0.1000
1-4 EEL                     999.0000     1.0000    0.1000
EEL                         -20.0000     2.0000    0.2000
EGB                          15.0000     1.5000    0.1500
ESURF                        -2.0000     0.2000    0.0200

DELTA TOTAL                 -17.0000     2.5000    0.2500
"""

    def test_label_anchoring_ignores_1_4_eel(self) -> None:
        # "EEL" must match the EEL row, not "1-4 EEL".
        r = parse_mmpbsa_dat(self._GB_ONLY)
        assert r.components.electrostatic == pytest.approx(-20.0)

    def test_unknown_solvent_model_raises(self, dat: str) -> None:
        with pytest.raises(ValueError, match="gb.*or.*pb|solvent_model"):
            parse_mmpbsa_dat(dat, solvent_model="implicit")

    def test_missing_section_raises(self) -> None:
        # No POISSON BOLTZMANN section in a GB-only file.
        with pytest.raises(ValueError, match="POISSON BOLTZMANN"):
            parse_mmpbsa_dat(self._GB_ONLY, solvent_model="pb")

    def test_missing_row_raises(self) -> None:
        broken = self._GB_ONLY.replace("DELTA TOTAL                 -17.0000     2.5000    0.2500", "")
        with pytest.raises(ValueError, match="DELTA TOTAL"):
            parse_mmpbsa_dat(broken)


def _complex() -> Protein:
    """A 5-residue complex: 3 protein (chain A), 1 ligand, 1 water."""
    spec = [
        (["N", "CA", "C"], "A", 1, "ALA", "protein"),
        (["N", "CA", "C"], "A", 2, "GLY", "protein"),
        (["N", "CA", "C"], "A", 3, "LEU", "protein"),
        (["C1", "C2"], "B", 1, "LIG", "ligand"),
        (["O"], "W", 1, "HOH", "water"),
    ]
    rows = [(nm, ch, rid, rn, ent) for atoms, ch, rid, rn, ent in spec for nm in atoms]
    n = len(rows)
    arr = AtomArray(n)
    arr.coords[:] = np.zeros((n, 3), dtype=np.float32)
    arr.atom_name[:] = [r[0] for r in rows]
    arr.chain_id[:] = [r[1] for r in rows]
    arr.residue_id[:] = [r[2] for r in rows]
    arr.residue_name[:] = [r[3] for r in rows]
    arr.entity_type[:] = [r[4] for r in rows]
    arr.element[:] = [r[0][0] for r in rows]
    return Protein(arr, name="cplx")


class TestSelectionMask:
    def test_protein_is_contiguous_range(self) -> None:
        assert selection_to_amber_mask(_complex(), {"entity_type": "protein"}) == ":1-3"

    def test_single_residue(self) -> None:
        assert selection_to_amber_mask(_complex(), {"entity_type": "ligand"}) == ":4"

    def test_by_residue_name(self) -> None:
        assert selection_to_amber_mask(_complex(), {"residue_name": "LIG"}) == ":4"

    def test_non_contiguous(self) -> None:
        mask = selection_to_amber_mask(_complex(), {"residue_name": ["ALA", "LEU", "HOH"]})
        assert mask == ":1,3,5"

    def test_boolean_mask_input(self) -> None:
        cplx = _complex()
        bool_mask = cplx.atom_array.entity_type == "ligand"
        assert selection_to_amber_mask(cplx, bool_mask) == ":4"

    def test_partial_residue_raises(self) -> None:
        with pytest.raises(ValueError, match="splits residue"):
            selection_to_amber_mask(_complex(), {"atom_name": "CA"})

    def test_no_atoms_raises(self) -> None:
        with pytest.raises(ValueError, match="matches no atoms"):
            selection_to_amber_mask(_complex(), {"residue_name": "ZZZ"})

    def test_wrong_shape_bool_mask_raises(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            selection_to_amber_mask(_complex(), np.array([True, False]))


class TestBuildInput:
    def test_gb_namelists(self) -> None:
        text = build_mmpbsa_input(end_frame=50, salt_conc=0.15)
        assert "molforge MM/GBSA input" in text
        assert "&general" in text and "&gb" in text and "&pb" not in text
        assert "startframe=1, endframe=50, interval=1" in text
        assert "igb=5, saltcon=0.15," in text

    def test_pb_namelists(self) -> None:
        text = build_mmpbsa_input(solvent_model="pb", end_frame=40, salt_conc=0.1)
        assert "molforge MM/PBSA input" in text
        assert "&pb" in text and "istrng=0.1," in text
        assert "&gb" not in text

    def test_frame_params_reflected(self) -> None:
        text = build_mmpbsa_input(start_frame=5, end_frame=40, interval=2)
        assert "startframe=5, endframe=40, interval=2" in text

    def test_zero_salt_formats_cleanly(self) -> None:
        assert "saltcon=0," in build_mmpbsa_input(end_frame=10)

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError, match="'gb' or 'pb'"):
            build_mmpbsa_input(solvent_model="implicit", end_frame=10)

    def test_bad_frame_range_raises(self) -> None:
        with pytest.raises(ValueError, match="end_frame"):
            build_mmpbsa_input(start_frame=20, end_frame=10)

    def test_bad_interval_raises(self) -> None:
        with pytest.raises(ValueError, match="interval"):
            build_mmpbsa_input(end_frame=10, interval=0)

    def test_bad_start_raises(self) -> None:
        with pytest.raises(ValueError, match="start_frame"):
            build_mmpbsa_input(start_frame=0, end_frame=10)
