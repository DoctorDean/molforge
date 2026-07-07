"""Tests for the Amber ``MMPBSA.py`` per-residue decomposition parser.

Driven by a real-shape ``FINAL_DECOMP_MMPBSA.dat`` fixture: a Complex
section (with a decoy Total block) followed by the DELTAS section whose
four residues sum to the overall −19.0 kcal/mol, then a Sidechain block
that must not leak into the Total parse.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molforge.freeenergy import Decomposition
from molforge.wrappers.freeenergy import parse_mmpbsa_decomp

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "freeenergy"
    / "FINAL_DECOMP_MMPBSA.dat"
)


@pytest.fixture
def dat() -> str:
    return FIXTURE.read_text()


class TestParseDecomp:
    def test_reads_the_delta_section(self, dat: str) -> None:
        d = parse_mmpbsa_decomp(dat)
        assert isinstance(d, Decomposition)
        # report order, and only the four DELTAS residues (not the Sidechain
        # block, not the Complex decoy)
        assert list(d) == ["LEU 40", "THR 41", "ALA 44", "LIG 241"]

    def test_total_matches_overall(self, dat: str) -> None:
        # the per-residue deltas sum to the overall binding contribution
        assert parse_mmpbsa_decomp(dat).total == pytest.approx(-19.0)

    def test_components_and_uncertainty(self, dat: str) -> None:
        leu = parse_mmpbsa_decomp(dat)["LEU 40"]
        assert leu.total == pytest.approx(-6.5)
        assert leu.uncertainty == pytest.approx(0.30)  # TOTAL's SEM
        assert leu.internal == pytest.approx(1.0)
        assert leu.vdw == pytest.approx(-6.0)
        assert leu.electrostatic == pytest.approx(-3.0)
        assert leu.polar_solvation == pytest.approx(2.0)
        assert leu.nonpolar_solvation == pytest.approx(-0.5)
        # breakdown reconstructs the total
        parts = leu.internal + leu.vdw + leu.electrostatic
        parts += leu.polar_solvation + leu.nonpolar_solvation
        assert parts == pytest.approx(leu.total)

    def test_picks_delta_not_complex_decoy(self, dat: str) -> None:
        # the Complex section lists LEU 40 with TOTAL 305.00; the default
        # must read the DELTAS value instead
        assert parse_mmpbsa_decomp(dat)["LEU 40"].total == pytest.approx(-6.5)

    def test_hotspots(self, dat: str) -> None:
        d = parse_mmpbsa_decomp(dat)
        assert [c.residue for c in d.hotspots(2)] == ["LIG 241", "LEU 40"]
        assert d.hotspots(1, favorable=False)[0].residue == "ALA 44"

    def test_complex_section(self, dat: str) -> None:
        c = parse_mmpbsa_decomp(dat, section="complex")
        assert list(c) == ["LEU 40"]
        assert c["LEU 40"].total == pytest.approx(305.0)

    def test_unknown_section_raises(self, dat: str) -> None:
        with pytest.raises(ValueError, match="unknown section"):
            parse_mmpbsa_decomp(dat, section="bogus")

    def test_missing_section_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            parse_mmpbsa_decomp("no decomposition here", section="delta")

    def test_sidechain_block_excluded(self, dat: str) -> None:
        # the Sidechain block also lists LEU 40 (total -3.20); the Total
        # parse must stop before it, so LEU 40 appears once with the Total
        # value
        d = parse_mmpbsa_decomp(dat)
        assert d["LEU 40"].total == pytest.approx(-6.5)
        assert len(d) == 4
