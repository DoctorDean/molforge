"""Tests for the PDBQT (AutoDock / Vina) reader and writer.

PDBQT is a thin extension of PDB:
  - Columns 1-66 are PDB-compatible.
  - Columns 71-76 hold the per-atom partial charge.
  - Columns 78-79 hold the AutoDock atom type.
  - ROOT / BRANCH / TORSDOF lines describe the rotatable-bond tree.

The reader reuses the standard PDB parser for the leading columns and
post-processes each atom line to pick up charges and atom types. The
writer is the symmetric operation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molforge.core import Protein
from molforge.io.pdbqt import (
    _parse_autodock_type,
    _parse_charge,
    read_pdbqt,
    read_pdbqt_string,
    write_pdbqt,
)

# A small but realistic PDBQT block. Mix of regular and 2-character
# AutoDock atom types (OA = H-bond-acceptor oxygen).
_SAMPLE_PDBQT = """REMARK  Name = ligand
ROOT
ATOM      1  N   ALA A   1      27.340  24.430   2.614  1.00  0.00    -0.103 N
ATOM      2  CA  ALA A   1      26.266  25.413   2.842  1.00  0.00     0.045 C
ATOM      3  C   ALA A   1      26.913  26.639   3.531  1.00  0.00     0.330 C
ATOM      4  O   ALA A   1      27.886  26.463   4.263  1.00  0.00    -0.297 OA
ENDROOT
BRANCH   2   5
ATOM      5  CB  ALA A   1      25.112  24.880   3.649  1.00  0.00     0.011 C
ENDBRANCH   2   5
TORSDOF 1
"""


# ---------------------------------------------------------------------
# Column extractors
# ---------------------------------------------------------------------


class TestColumnExtractors:
    def test_charge_negative(self) -> None:
        line = "ATOM      1  N   ALA A   1      27.340  24.430   2.614  1.00  0.00    -0.103 N "
        assert _parse_charge(line) == pytest.approx(-0.103)

    def test_charge_positive(self) -> None:
        line = "ATOM      2  CA  ALA A   1      26.266  25.413   2.842  1.00  0.00     0.045 C "
        assert _parse_charge(line) == pytest.approx(0.045)

    def test_charge_falls_back_to_whitespace_split(self) -> None:
        """A line that's been re-spaced (missing the fixed-col padding)
        still yields the right charge via the whitespace fallback."""
        line = "ATOM 1 N ALA A 1 27.340 24.430 2.614 1.00 0.00 -0.103 N"
        assert _parse_charge(line) == pytest.approx(-0.103)

    def test_charge_zero_when_unparseable(self) -> None:
        line = "ATOM      1  N   ALA A   1      x y z 1.00 0.00 garbage N"
        assert _parse_charge(line) == pytest.approx(0.0)

    def test_autodock_type_single_letter(self) -> None:
        line = "ATOM      1  N   ALA A   1      27.340  24.430   2.614  1.00  0.00    -0.103 N "
        assert _parse_autodock_type(line) == "N"

    def test_autodock_type_two_letter(self) -> None:
        line = "ATOM      4  O   ALA A   1      27.886  26.463   4.263  1.00  0.00    -0.297 OA"
        assert _parse_autodock_type(line) == "OA"

    def test_autodock_type_missing_returns_empty(self) -> None:
        # Truncated at the charge column — no type field.
        line = "ATOM      1  N   ALA A   1      27.340  24.430   2.614  1.00  0.00    -0.103"
        assert _parse_autodock_type(line) == ""


# ---------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------


class TestReadPdbqtString:
    def test_returns_protein(self) -> None:
        prot = read_pdbqt_string(_SAMPLE_PDBQT)
        assert isinstance(prot, Protein)
        assert prot.atom_array.n_atoms == 5

    def test_coordinates_parsed(self) -> None:
        prot = read_pdbqt_string(_SAMPLE_PDBQT)
        assert tuple(prot.atom_array.coords[0]) == pytest.approx((27.340, 24.430, 2.614), abs=1e-3)

    def test_charges_populated(self) -> None:
        prot = read_pdbqt_string(_SAMPLE_PDBQT)
        np.testing.assert_allclose(
            prot.atom_array.charge,
            [-0.103, 0.045, 0.330, -0.297, 0.011],
            atol=1e-3,
        )

    def test_autodock_types_attached_to_metadata(self) -> None:
        prot = read_pdbqt_string(_SAMPLE_PDBQT)
        assert prot.metadata["autodock_types"] == ["N", "C", "C", "OA", "C"]

    def test_root_branch_torsdof_lines_ignored(self) -> None:
        """ROOT / BRANCH / TORSDOF must not affect atom count."""
        prot = read_pdbqt_string(_SAMPLE_PDBQT)
        # 5 atoms in the sample — ROOT/BRANCH/TORSDOF didn't add any.
        assert prot.atom_array.n_atoms == 5

    def test_empty_input_is_handled(self) -> None:
        prot = read_pdbqt_string("")
        assert prot.atom_array.n_atoms == 0

    def test_no_autodock_types_no_metadata_key(self) -> None:
        """A PDBQT-shaped string with no actual AutoDock-type column
        doesn't gain a metadata['autodock_types'] key."""
        # Build atom lines truncated before the type column.
        text = (
            "ATOM      1  N   ALA A   1      "
            "27.340  24.430   2.614  1.00  0.00    -0.103\n"
            "ATOM      2  CA  ALA A   1      "
            "26.266  25.413   2.842  1.00  0.00     0.045\n"
            "END\n"
        )
        prot = read_pdbqt_string(text)
        assert prot.atom_array.n_atoms == 2
        assert "autodock_types" not in prot.metadata


# ---------------------------------------------------------------------
# File reading + dispatcher
# ---------------------------------------------------------------------


class TestReadFromDisk:
    def test_read_file(self, tmp_path: Path) -> None:
        fp = tmp_path / "ligand.pdbqt"
        fp.write_text(_SAMPLE_PDBQT)
        prot = read_pdbqt(fp)
        assert prot.atom_array.n_atoms == 5
        assert tuple(prot.atom_array.charge[3:4]) == pytest.approx((-0.297,), abs=1e-3)

    def test_dispatcher_load_routes_to_pdbqt(self, tmp_path: Path) -> None:
        from molforge.io import load

        fp = tmp_path / "ligand.pdbqt"
        fp.write_text(_SAMPLE_PDBQT)
        prot = load(fp)
        assert isinstance(prot, Protein)
        assert prot.atom_array.n_atoms == 5

    def test_dispatcher_save_routes_to_pdbqt(self, tmp_path: Path) -> None:
        from molforge.io import load, save

        fp_in = tmp_path / "in.pdbqt"
        fp_in.write_text(_SAMPLE_PDBQT)
        prot = read_pdbqt(fp_in)
        fp_out = tmp_path / "out.pdbqt"
        save(prot, fp_out)
        assert fp_out.is_file()
        rt = load(fp_out)
        assert rt.atom_array.n_atoms == prot.atom_array.n_atoms


# ---------------------------------------------------------------------
# Writing and round-trip
# ---------------------------------------------------------------------


class TestWriteRoundTrip:
    def test_round_trip_preserves_atom_count(self, tmp_path: Path) -> None:
        prot = read_pdbqt_string(_SAMPLE_PDBQT)
        fp = tmp_path / "out.pdbqt"
        write_pdbqt(prot, fp)
        rt = read_pdbqt(fp)
        assert rt.atom_array.n_atoms == prot.atom_array.n_atoms

    def test_round_trip_preserves_coordinates(self, tmp_path: Path) -> None:
        prot = read_pdbqt_string(_SAMPLE_PDBQT)
        fp = tmp_path / "out.pdbqt"
        write_pdbqt(prot, fp)
        rt = read_pdbqt(fp)
        np.testing.assert_allclose(rt.atom_array.coords, prot.atom_array.coords, atol=1e-3)

    def test_round_trip_preserves_charges(self, tmp_path: Path) -> None:
        prot = read_pdbqt_string(_SAMPLE_PDBQT)
        fp = tmp_path / "out.pdbqt"
        write_pdbqt(prot, fp)
        rt = read_pdbqt(fp)
        np.testing.assert_allclose(rt.atom_array.charge, prot.atom_array.charge, atol=1e-3)

    def test_round_trip_preserves_autodock_types(self, tmp_path: Path) -> None:
        prot = read_pdbqt_string(_SAMPLE_PDBQT)
        fp = tmp_path / "out.pdbqt"
        write_pdbqt(prot, fp)
        rt = read_pdbqt(fp)
        assert rt.metadata["autodock_types"] == prot.metadata["autodock_types"]

    def test_write_falls_back_to_element_when_no_autodock_types(self, tmp_path: Path) -> None:
        """A Protein without metadata['autodock_types'] still writes a
        valid PDBQT, using the element as the type (a documented
        best-effort fallback)."""
        prot = read_pdbqt_string(_SAMPLE_PDBQT)
        # Drop the autodock_types so the writer must fall back.
        prot.metadata = {k: v for k, v in prot.metadata.items() if k != "autodock_types"}
        fp = tmp_path / "out.pdbqt"
        write_pdbqt(prot, fp)
        rt = read_pdbqt(fp)
        # Atom 0 is N → type "N"; atom 3 is O → type "O" (not "OA",
        # because we no longer have the original info).
        assert rt.metadata["autodock_types"][0] == "N"
        assert rt.metadata["autodock_types"][3] == "O"


# ---------------------------------------------------------------------
# Multi-MODEL files (Vina pose output) round-trip via the reader
# ---------------------------------------------------------------------


class TestMultiModel:
    def test_first_model_extracted_by_default(self, tmp_path: Path) -> None:
        """Multi-MODEL PDBQT (as Vina emits) is handled by the underlying
        PDB reader; without an explicit model arg we get the union, but
        with model=1 we get just the first."""
        multi = (
            "MODEL 1\n"
            + _SAMPLE_PDBQT
            + "ENDMDL\n"
            + "MODEL 2\n"
            + _SAMPLE_PDBQT.replace("27.340", "27.500")
            + "ENDMDL\n"
        )
        prot1 = read_pdbqt_string(multi, model=1)
        assert prot1.atom_array.n_atoms == 5
        assert prot1.atom_array.coords[0, 0] == pytest.approx(27.340, abs=1e-3)
        prot2 = read_pdbqt_string(multi, model=2)
        assert prot2.atom_array.coords[0, 0] == pytest.approx(27.500, abs=1e-3)
