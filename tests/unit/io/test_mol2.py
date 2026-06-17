"""Tests for the MOL2 (Tripos) reader and writer.

The reader handles single- and multi-molecule MOL2 files, extracts
coordinates, elements (from the prefix of the Tripos atom type),
atom names, partial charges, and substructure info, and tolerates
short atom lines that omit the optional trailing columns. The writer
round-trips coordinates, elements, atom names, and charges.

Tripos atom typing (e.g. ``C.3`` → ``C``, ``C.ar`` → ``C``) and
two-letter elements (``Cl``, ``Br``) are explicitly tested.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molforge.core import Protein
from molforge.io.mol2 import read_mol2, read_mol2_string, write_mol2

# A small synthetic MOL2: 9 atoms with mixed Tripos atom types,
# two-letter element (Cl), partial charges, and a BOND section.
_SAMPLE_MOL2 = """@<TRIPOS>MOLECULE
aspirin-ish
9 8 1 0 0
SMALL
USER_CHARGES


@<TRIPOS>ATOM
      1 C1         -1.5000   -0.5000    0.0000 C.ar    1 LIG    -0.1250
      2 C2          0.0000    0.0000    0.0000 C.ar    1 LIG    -0.1250
      3 O1          0.5000   -0.5000   -1.3000 O.3     1 LIG    -0.5000
      4 C3          1.0000    0.5000    1.0000 C.3     1 LIG     0.2500
      5 O2          2.2000    0.3000    1.0000 O.2     1 LIG    -0.4000
      6 H1          0.4000    1.3000    1.9000 H       1 LIG     0.0500
      7 N1          1.2000    2.0000    2.9000 N.am    1 LIG    -0.3000
      8 Cl1         0.4000    2.5000    4.1000 Cl      1 LIG    -0.1000
      9 H2          2.0000    3.2000    2.4000 H       1 LIG     0.0500
@<TRIPOS>BOND
     1    1    2 ar
     2    2    3 1
     3    2    4 1
     4    4    5 2
     5    4    6 1
     6    4    7 1
     7    1    8 1
     8    7    9 1
"""

_MULTI_MOL2 = _SAMPLE_MOL2 + _SAMPLE_MOL2.replace("aspirin-ish", "ibuprofen-ish")


# ---------------------------------------------------------------------
# Atom-section reading
# ---------------------------------------------------------------------


class TestReadAtomSection:
    def test_returns_list_even_for_single_molecule(self) -> None:
        mols = read_mol2_string(_SAMPLE_MOL2)
        assert isinstance(mols, list)
        assert len(mols) == 1

    def test_atom_count(self) -> None:
        mol = read_mol2_string(_SAMPLE_MOL2)[0]
        assert isinstance(mol, Protein)
        assert mol.atom_array.n_atoms == 9

    def test_elements_extracted_from_tripos_type(self) -> None:
        """Tripos atom types like 'C.ar', 'N.am' yield the element prefix."""
        mol = read_mol2_string(_SAMPLE_MOL2)[0]
        elements = list(mol.atom_array.element)
        # Atoms 0-1: C.ar -> C; atom 2: O.3 -> O; atom 6: N.am -> N
        assert elements[0] == "C"
        assert elements[1] == "C"
        assert elements[2] == "O"
        assert elements[6] == "N"

    def test_two_letter_element_preserved(self) -> None:
        """A bare 'Cl' atom type yields a 'Cl' element, not 'C'."""
        mol = read_mol2_string(_SAMPLE_MOL2)[0]
        assert mol.atom_array.element[7] == "Cl"

    def test_atom_names_preserved(self) -> None:
        mol = read_mol2_string(_SAMPLE_MOL2)[0]
        assert list(mol.atom_array.atom_name) == [
            "C1",
            "C2",
            "O1",
            "C3",
            "O2",
            "H1",
            "N1",
            "Cl1",
            "H2",
        ]

    def test_coordinates_extracted(self) -> None:
        mol = read_mol2_string(_SAMPLE_MOL2)[0]
        assert tuple(mol.atom_array.coords[0]) == pytest.approx((-1.5, -0.5, 0.0))
        assert tuple(mol.atom_array.coords[7]) == pytest.approx((0.4, 2.5, 4.1))

    def test_partial_charges_extracted(self) -> None:
        mol = read_mol2_string(_SAMPLE_MOL2)[0]
        assert mol.atom_array.charge[0] == pytest.approx(-0.125)
        assert mol.atom_array.charge[4] == pytest.approx(-0.4)
        assert mol.atom_array.charge[5] == pytest.approx(0.05)

    def test_marked_as_ligand(self) -> None:
        mol = read_mol2_string(_SAMPLE_MOL2)[0]
        assert all(et == "ligand" for et in mol.atom_array.entity_type)
        assert all(rt == "HETATM" for rt in mol.atom_array.record_type)

    def test_residue_info_from_substructure_columns(self) -> None:
        mol = read_mol2_string(_SAMPLE_MOL2)[0]
        assert all(rid == 1 for rid in mol.atom_array.residue_id)
        assert all(rn == "LIG" for rn in mol.atom_array.residue_name)


# ---------------------------------------------------------------------
# Title (MOLECULE section)
# ---------------------------------------------------------------------


class TestTitle:
    def test_title_attached_to_metadata(self) -> None:
        mol = read_mol2_string(_SAMPLE_MOL2)[0]
        assert mol.metadata.get("title") == "aspirin-ish"


# ---------------------------------------------------------------------
# Multi-molecule files
# ---------------------------------------------------------------------


class TestMultiMolecule:
    def test_parses_all_molecules(self) -> None:
        mols = read_mol2_string(_MULTI_MOL2)
        assert len(mols) == 2
        assert [m.metadata["title"] for m in mols] == [
            "aspirin-ish",
            "ibuprofen-ish",
        ]

    def test_each_molecule_is_independent(self) -> None:
        mols = read_mol2_string(_MULTI_MOL2)
        mols[0].atom_array.coords[0] = (99.0, 99.0, 99.0)
        assert tuple(mols[1].atom_array.coords[0]) == pytest.approx((-1.5, -0.5, 0.0))

    def test_empty_input_returns_empty_list(self) -> None:
        assert read_mol2_string("") == []
        assert read_mol2_string("\n\n") == []

    def test_input_without_molecule_tag_returns_empty_list(self) -> None:
        # Lines before the first @<TRIPOS>MOLECULE are ignored.
        assert read_mol2_string("# a stray comment\n\n") == []


# ---------------------------------------------------------------------
# Short / minimal atom lines (optional columns omitted)
# ---------------------------------------------------------------------


class TestOptionalColumns:
    def _build_minimal_mol2(self, atom_line: str) -> str:
        return (
            "@<TRIPOS>MOLECULE\n"
            "minimal\n"
            "1 0 0 0 0\n"
            "SMALL\n"
            "NO_CHARGES\n"
            "\n"
            "@<TRIPOS>ATOM\n"
            f"{atom_line}\n"
        )

    def test_no_charge_defaults_to_zero(self) -> None:
        # Only 8 cols: atom_id, name, x, y, z, type, subst_id, subst_name
        mol2 = self._build_minimal_mol2("      1 C1   1.0  2.0  3.0  C.3  1 LIG")
        mol = read_mol2_string(mol2)[0]
        assert mol.atom_array.charge[0] == pytest.approx(0.0)

    def test_no_substructure_defaults_to_lig(self) -> None:
        # Only 6 cols (the required minimum).
        mol2 = self._build_minimal_mol2("      1 C1   1.0  2.0  3.0  C.3")
        mol = read_mol2_string(mol2)[0]
        assert mol.atom_array.residue_id[0] == 1
        assert mol.atom_array.residue_name[0] == "LIG"
        assert mol.atom_array.charge[0] == pytest.approx(0.0)

    def test_unparseable_subst_id_falls_back_silently(self) -> None:
        """A non-integer subst_id (some non-conforming writers emit
        '****' or '-') is treated as residue_id=1 rather than a parse
        error — we'd rather load the coordinates."""
        mol2 = self._build_minimal_mol2("      1 C1   1.0  2.0  3.0  C.3   *** LIG  0.0")
        mol = read_mol2_string(mol2)[0]
        assert mol.atom_array.residue_id[0] == 1

    def test_unparseable_charge_falls_back_silently(self) -> None:
        """A non-numeric charge column falls back to 0.0 rather than
        crashing the whole molecule."""
        mol2 = self._build_minimal_mol2("      1 C1   1.0  2.0  3.0  C.3   1 LIG  none")
        mol = read_mol2_string(mol2)[0]
        assert mol.atom_array.charge[0] == pytest.approx(0.0)

    def test_blank_line_inside_atom_section_skipped(self) -> None:
        """A blank line in the atom section doesn't break parsing."""
        mol2 = (
            "@<TRIPOS>MOLECULE\nx\n2 0 0 0 0\nSMALL\nNO_CHARGES\n\n"
            "@<TRIPOS>ATOM\n"
            "  1 C1 0.0 0.0 0.0 C.3 1 LIG  0.0\n"
            "\n"
            "  2 C2 1.0 1.0 1.0 C.3 1 LIG  0.0\n"
        )
        mol = read_mol2_string(mol2)[0]
        assert mol.atom_array.n_atoms == 2


# ---------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------


class TestErrorPaths:
    def test_missing_atom_section_raises(self) -> None:
        bad = "@<TRIPOS>MOLECULE\nname\n0 0 0 0 0\nSMALL\nNO_CHARGES\n"
        with pytest.raises(ValueError, match="ATOM"):
            read_mol2_string(bad)

    def test_atom_count_mismatch_raises(self) -> None:
        # Header says 10 atoms, ATOM section has 9.
        bad = _SAMPLE_MOL2.replace("9 8 1 0 0", "10 8 1 0 0")
        with pytest.raises(ValueError, match="declares 10 atoms"):
            read_mol2_string(bad)

    def test_malformed_atom_line_raises(self) -> None:
        bad = (
            "@<TRIPOS>MOLECULE\nx\n1 0 0 0 0\nSMALL\nNO_CHARGES\n\n@<TRIPOS>ATOM\n  1 C1 1.0 2.0\n"
        )
        with pytest.raises(ValueError, match="malformed MOL2 atom line"):
            read_mol2_string(bad)

    def test_malformed_coordinates_raise(self) -> None:
        bad = (
            "@<TRIPOS>MOLECULE\nx\n1 0 0 0 0\nSMALL\nNO_CHARGES\n\n"
            "@<TRIPOS>ATOM\n  1 C1 NaN-ish bad coords C.3\n"
        )
        with pytest.raises(ValueError, match="malformed coordinates"):
            read_mol2_string(bad)

    def test_empty_atom_section_raises(self) -> None:
        bad = "@<TRIPOS>MOLECULE\nx\n0 0 0 0 0\nSMALL\nNO_CHARGES\n\n@<TRIPOS>ATOM\n"
        with pytest.raises(ValueError, match="empty"):
            read_mol2_string(bad)


# ---------------------------------------------------------------------
# Read from disk
# ---------------------------------------------------------------------


class TestReadFromDisk:
    def test_reads_file_path(self, tmp_path: Path) -> None:
        fp = tmp_path / "ligand.mol2"
        fp.write_text(_SAMPLE_MOL2)
        mols = read_mol2(fp)
        assert len(mols) == 1
        assert mols[0].atom_array.n_atoms == 9

    def test_reads_via_dispatcher_load(self, tmp_path: Path) -> None:
        from molforge.io import load

        fp = tmp_path / "ligand.mol2"
        fp.write_text(_SAMPLE_MOL2)
        mols = load(fp)
        assert isinstance(mols, list)
        assert len(mols) == 1


# ---------------------------------------------------------------------
# Writing and round-trip
# ---------------------------------------------------------------------


class TestWriteRoundTrip:
    def test_single_protein_round_trips(self, tmp_path: Path) -> None:
        mol = read_mol2_string(_SAMPLE_MOL2)[0]
        fp = tmp_path / "out.mol2"
        write_mol2(mol, fp)
        result = read_mol2(fp)
        assert len(result) == 1
        assert result[0].atom_array.n_atoms == mol.atom_array.n_atoms

    def test_coordinates_preserved(self, tmp_path: Path) -> None:
        mol = read_mol2_string(_SAMPLE_MOL2)[0]
        fp = tmp_path / "out.mol2"
        write_mol2(mol, fp)
        rt = read_mol2(fp)[0]
        for i in range(mol.atom_array.n_atoms):
            assert tuple(rt.atom_array.coords[i]) == pytest.approx(
                tuple(mol.atom_array.coords[i]), abs=1e-3
            )

    def test_elements_preserved(self, tmp_path: Path) -> None:
        mol = read_mol2_string(_SAMPLE_MOL2)[0]
        fp = tmp_path / "out.mol2"
        write_mol2(mol, fp)
        rt = read_mol2(fp)[0]
        assert list(rt.atom_array.element) == list(mol.atom_array.element)

    def test_charges_preserved(self, tmp_path: Path) -> None:
        mol = read_mol2_string(_SAMPLE_MOL2)[0]
        fp = tmp_path / "out.mol2"
        write_mol2(mol, fp)
        rt = read_mol2(fp)[0]
        for i in range(mol.atom_array.n_atoms):
            assert rt.atom_array.charge[i] == pytest.approx(mol.atom_array.charge[i], abs=1e-3)

    def test_atom_names_preserved(self, tmp_path: Path) -> None:
        mol = read_mol2_string(_SAMPLE_MOL2)[0]
        fp = tmp_path / "out.mol2"
        write_mol2(mol, fp)
        rt = read_mol2(fp)[0]
        assert list(rt.atom_array.atom_name) == list(mol.atom_array.atom_name)

    def test_title_preserved(self, tmp_path: Path) -> None:
        mol = read_mol2_string(_SAMPLE_MOL2)[0]
        fp = tmp_path / "out.mol2"
        write_mol2(mol, fp)
        rt = read_mol2(fp)[0]
        assert rt.metadata.get("title") == "aspirin-ish"

    def test_list_of_proteins_round_trips(self, tmp_path: Path) -> None:
        mols = read_mol2_string(_MULTI_MOL2)
        fp = tmp_path / "multi.mol2"
        write_mol2(mols, fp)
        rt = read_mol2(fp)
        assert len(rt) == len(mols)
        assert [m.metadata["title"] for m in rt] == [
            "aspirin-ish",
            "ibuprofen-ish",
        ]

    def test_writes_via_dispatcher_save(self, tmp_path: Path) -> None:
        from molforge.io import save

        mol = read_mol2_string(_SAMPLE_MOL2)[0]
        fp = tmp_path / "via_save.mol2"
        save(mol, fp)
        assert fp.is_file()
        assert read_mol2(fp)[0].atom_array.n_atoms == 9
