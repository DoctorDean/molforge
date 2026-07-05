"""Tests for the gmx_MMPBSA results parser.

gmx_MMPBSA writes the same file structure as MMPBSA.py but with
Δ-prefixed delta rows and five numeric columns; these check that the
shared helpers read column 0 (ΔG) and column -1 (SEM) correctly and that
the Δ-labels don't collide (ΔVDWAALS vs Δ1-4 VDW).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molforge.wrappers.freeenergy import parse_gmx_mmpbsa_dat

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "freeenergy"
    / "gmx_FINAL_RESULTS_MMPBSA.dat"
)


@pytest.fixture
def dat() -> str:
    return FIXTURE.read_text()


class TestGeneralizedBorn:
    def test_delta_g_and_uncertainty(self, dat: str) -> None:
        r = parse_gmx_mmpbsa_dat(dat)  # gb default
        assert r.method == "MM/GBSA"
        assert r.delta_g == pytest.approx(-21.0)
        assert r.uncertainty == pytest.approx(0.7)  # last column (SEM), not SD

    def test_components(self, dat: str) -> None:
        c = parse_gmx_mmpbsa_dat(dat).components
        assert c is not None
        assert c.vdw == pytest.approx(-45.0)
        assert c.electrostatic == pytest.approx(-30.0)
        assert c.polar_solvation == pytest.approx(60.0)  # ΔEGB
        assert c.nonpolar_solvation == pytest.approx(-6.0)  # ΔESURF
        assert c.entropy is None

    def test_enthalpy_reconstructs_total(self, dat: str) -> None:
        r = parse_gmx_mmpbsa_dat(dat)
        assert r.components.enthalpy == pytest.approx(r.delta_g)

    def test_metadata(self, dat: str) -> None:
        r = parse_gmx_mmpbsa_dat(dat)
        assert r.metadata["solvent_model"] == "gb"
        assert r.metadata["n_frames"] == 16
        assert r.metadata["delta_total_std_dev"] == pytest.approx(7.0)  # sample SD column


class TestPoissonBoltzmann:
    def test_selects_pb_section(self, dat: str) -> None:
        r = parse_gmx_mmpbsa_dat(dat, solvent_model="pb")
        assert r.method == "MM/PBSA"
        assert r.delta_g == pytest.approx(-24.0)
        assert r.uncertainty == pytest.approx(0.72)

    def test_nonpolar_sums_enpolar_and_edisper(self, dat: str) -> None:
        c = parse_gmx_mmpbsa_dat(dat, solvent_model="pb").components
        assert c is not None
        assert c.polar_solvation == pytest.approx(55.0)  # ΔEPB
        assert c.nonpolar_solvation == pytest.approx(-4.0)  # -8 + 4
        assert c.enthalpy == pytest.approx(-24.0)


class TestRobustness:
    def test_delta_label_anchoring(self, dat: str) -> None:
        # ΔVDWAALS must not pick up "Δ1-4 VDW", nor ΔEEL "Δ1-4 EEL".
        c = parse_gmx_mmpbsa_dat(dat).components
        assert c.vdw == pytest.approx(-45.0)
        assert c.electrostatic == pytest.approx(-30.0)

    def test_reads_delta_not_complex_block(self, dat: str) -> None:
        # The Complex block has VDWAALS -900 (no Δ); must be ignored.
        assert parse_gmx_mmpbsa_dat(dat).components.vdw == pytest.approx(-45.0)

    def test_unknown_model_raises(self, dat: str) -> None:
        with pytest.raises(ValueError, match="'gb' or 'pb'"):
            parse_gmx_mmpbsa_dat(dat, solvent_model="rism")

    def test_missing_section_raises(self, dat: str) -> None:
        gb_only = dat.split("POISSON BOLTZMANN:")[0]
        with pytest.raises(ValueError, match="POISSON BOLTZMANN"):
            parse_gmx_mmpbsa_dat(gb_only, solvent_model="pb")

    def test_missing_row_raises(self) -> None:
        text = "GENERALIZED BORN:\n\nDelta (Complex - Receptor - Ligand):\nΔVDWAALS 1 2 3 4 5\n"
        with pytest.raises(ValueError, match="ΔEEL"):
            parse_gmx_mmpbsa_dat(text)
