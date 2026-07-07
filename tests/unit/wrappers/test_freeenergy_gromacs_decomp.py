"""Tests for the gmx_MMPBSA per-residue decomposition parser.

Driven by a real-shape whitespace ``gmx_FINAL_DECOMP_MMPBSA.dat`` fixture.
gmx's delta rows carry a ``Location`` column (``LEU 40 R LEU 40``) absent
from the Complex/Receptor/Ligand sections, so this exercises stripping it
back to the complex-numbering ``resname resnum`` label.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molforge.freeenergy import Decomposition
from molforge.wrappers.freeenergy import parse_gmx_mmpbsa_decomp

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "freeenergy"
    / "gmx_FINAL_DECOMP_MMPBSA.dat"
)


@pytest.fixture
def dat() -> str:
    return FIXTURE.read_text()


class TestParseGmxDecomp:
    def test_reads_delta_section(self, dat: str) -> None:
        d = parse_gmx_mmpbsa_decomp(dat)
        assert isinstance(d, Decomposition)
        # Location column stripped -> clean complex-numbering labels,
        # in report order, no Sidechain leakage
        assert list(d) == ["LEU 40", "THR 41", "ALA 44", "RAL 241"]

    def test_location_column_stripped(self, dat: str) -> None:
        # the raw row is "LEU  40 R LEU  40  <18 nums>"; the label must be
        # just the residue, not "LEU 40 R LEU 40"
        d = parse_gmx_mmpbsa_decomp(dat)
        assert "LEU 40 R LEU 40" not in d
        assert "LEU 40" in d

    def test_components_and_uncertainty(self, dat: str) -> None:
        leu = parse_gmx_mmpbsa_decomp(dat)["LEU 40"]
        assert leu.total == pytest.approx(-2.02)
        assert leu.uncertainty == pytest.approx(0.06)
        assert leu.internal == pytest.approx(0.0)
        assert leu.vdw == pytest.approx(-2.46)
        assert leu.electrostatic == pytest.approx(-0.45)
        assert leu.polar_solvation == pytest.approx(1.11)
        assert leu.nonpolar_solvation == pytest.approx(-0.22)

    def test_ligand_residue_keeps_complex_numbering(self, dat: str) -> None:
        # RAL is residue 241 in the complex (Location says L RAL 1)
        ral = parse_gmx_mmpbsa_decomp(dat)["RAL 241"]
        assert ral.total == pytest.approx(-28.0)
        assert ral.vdw == pytest.approx(-29.49)

    def test_total(self, dat: str) -> None:
        assert parse_gmx_mmpbsa_decomp(dat).total == pytest.approx(-33.72)

    def test_hotspots(self, dat: str) -> None:
        d = parse_gmx_mmpbsa_decomp(dat)
        assert [c.residue for c in d.hotspots(2)] == ["RAL 241", "LEU 40"]

    def test_picks_delta_not_complex_decoy(self, dat: str) -> None:
        # the Complex section lists LEU 40 with TOTAL -18.86
        assert parse_gmx_mmpbsa_decomp(dat)["LEU 40"].total == pytest.approx(-2.02)

    def test_complex_section_no_location(self, dat: str) -> None:
        c = parse_gmx_mmpbsa_decomp(dat, section="complex")
        assert list(c) == ["LEU 40"]
        assert c["LEU 40"].total == pytest.approx(-18.86)

    def test_unknown_and_missing_section_raise(self, dat: str) -> None:
        with pytest.raises(ValueError, match="unknown section"):
            parse_gmx_mmpbsa_decomp(dat, section="bogus")
        with pytest.raises(ValueError, match="not found"):
            parse_gmx_mmpbsa_decomp("nothing here")
