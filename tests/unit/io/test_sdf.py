"""Tests for the SDF (Structure Data File) reader and writer.

The reader handles single- and multi-molecule V2000 SDF files, the
property block (the ``> <Name>`` / value pairs after ``M  END``), and
``$$$$``-delimited concatenation. The writer round-trips coordinates,
elements, the title line, and the property block.

V3000 ("extended connection table") files are intentionally not
supported and should raise a clear error pointing at conversion paths.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molforge.core import Protein
from molforge.io.sdf import read_sdf, read_sdf_string, write_sdf

_SAMPLE_SDF = """ligand-name
  molforge generated

  3  2  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.9572    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.2400    0.9270    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1  3  1  0
M  END
> <Source>
Computed

> <MultiLine>
line1
line2

$$$$
"""

_MULTI_SDF = _SAMPLE_SDF + _SAMPLE_SDF.replace("ligand-name", "second-mol")


# ---------------------------------------------------------------------
# Atom-block reading
# ---------------------------------------------------------------------


class TestReadAtomBlock:
    def test_returns_list_even_for_single_molecule(self) -> None:
        # SDF is multi-molecule by nature; the reader always returns
        # list[Protein] so callers don't have to switch on count.
        mols = read_sdf_string(_SAMPLE_SDF)
        assert isinstance(mols, list)
        assert len(mols) == 1

    def test_atom_count_and_elements(self) -> None:
        mol = read_sdf_string(_SAMPLE_SDF)[0]
        assert isinstance(mol, Protein)
        assert mol.atom_array.n_atoms == 3
        assert list(mol.atom_array.element) == ["O", "H", "H"]

    def test_marked_as_ligand(self) -> None:
        mol = read_sdf_string(_SAMPLE_SDF)[0]
        assert all(et == "ligand" for et in mol.atom_array.entity_type)
        assert all(rt == "HETATM" for rt in mol.atom_array.record_type)

    def test_atom_names_are_unique_and_element_derived(self) -> None:
        mol = read_sdf_string(_SAMPLE_SDF)[0]
        assert list(mol.atom_array.atom_name) == ["O1", "H1", "H2"]

    def test_coordinates_extracted_in_angstrom(self) -> None:
        mol = read_sdf_string(_SAMPLE_SDF)[0]
        assert tuple(mol.atom_array.coords[1]) == pytest.approx((0.9572, 0.0, 0.0))


# ---------------------------------------------------------------------
# Title and property block
# ---------------------------------------------------------------------


class TestTitleAndProperties:
    def test_title_attached_to_metadata(self) -> None:
        mol = read_sdf_string(_SAMPLE_SDF)[0]
        assert mol.metadata.get("title") == "ligand-name"

    def test_property_block_parsed_into_metadata(self) -> None:
        mol = read_sdf_string(_SAMPLE_SDF)[0]
        props = mol.metadata.get("properties")
        assert props == {"Source": "Computed", "MultiLine": "line1\nline2"}

    def test_no_properties_means_no_property_key(self) -> None:
        minimal = """name
  molforge

  1  0  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
M  END
$$$$
"""
        mol = read_sdf_string(minimal)[0]
        assert "properties" not in mol.metadata


# ---------------------------------------------------------------------
# Multi-molecule files
# ---------------------------------------------------------------------


class TestMultiMolecule:
    def test_parses_all_molecules(self) -> None:
        mols = read_sdf_string(_MULTI_SDF)
        assert len(mols) == 2
        assert [m.metadata["title"] for m in mols] == ["ligand-name", "second-mol"]

    def test_each_molecule_is_independent(self) -> None:
        mols = read_sdf_string(_MULTI_SDF)
        # Mutating one must not touch the other.
        mols[0].atom_array.coords[0] = (99.0, 99.0, 99.0)
        assert tuple(mols[1].atom_array.coords[0]) == pytest.approx((0.0, 0.0, 0.0))

    def test_trailing_blank_lines_tolerated(self) -> None:
        # Many tools emit trailing whitespace after the last $$$$.
        mols = read_sdf_string(_MULTI_SDF + "\n\n  \n")
        assert len(mols) == 2

    def test_empty_input_returns_empty_list(self) -> None:
        assert read_sdf_string("") == []
        assert read_sdf_string("\n\n") == []


# ---------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------


class TestErrorPaths:
    def test_truncated_block_raises(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            read_sdf_string("only\ntwo\nlines\n$$$$\n")

    def test_truncated_atom_block_raises(self) -> None:
        bad = "name\n\n\n  5  0  0  0  0  0  0  0  0  0999 V2000\n"
        with pytest.raises(ValueError, match="truncated"):
            read_sdf_string(bad)

    def test_unparseable_atom_count_raises(self) -> None:
        bad = "name\n\n\n  XX  0  0  0  0  0  0  0  0  0999 V2000\n"
        with pytest.raises(ValueError, match="atom count"):
            read_sdf_string(bad)

    def test_malformed_atom_line_raises(self) -> None:
        bad = (
            "name\n\n\n  1  0  0  0  0  0  0  0  0  0999 V2000\n"
            "  not-a-number here       O   0  0  0\n"
        )
        with pytest.raises(ValueError, match="malformed SDF atom line"):
            read_sdf_string(bad)

    def test_v3000_detected_with_helpful_message(self) -> None:
        v3000 = (
            "name\n\n\n  0  0  0  0  0  0  0  0  0  0999 V3000\nM  V30 BEGIN CTAB\nM  END\n$$$$\n"
        )
        with pytest.raises(ValueError, match="V3000"):
            read_sdf_string(v3000)


# ---------------------------------------------------------------------
# Read from disk
# ---------------------------------------------------------------------


class TestReadFromDisk:
    def test_reads_file_path(self, tmp_path: Path) -> None:
        fp = tmp_path / "ligand.sdf"
        fp.write_text(_SAMPLE_SDF)
        mols = read_sdf(fp)
        assert len(mols) == 1
        assert mols[0].atom_array.n_atoms == 3

    def test_reads_via_dispatcher_load(self, tmp_path: Path) -> None:
        from molforge.io import load

        fp = tmp_path / "ligand.sdf"
        fp.write_text(_SAMPLE_SDF)
        mols = load(fp)
        assert isinstance(mols, list)
        assert len(mols) == 1


# ---------------------------------------------------------------------
# Writing and round-trip
# ---------------------------------------------------------------------


class TestWriteRoundTrip:
    def test_single_protein_round_trips(self, tmp_path: Path) -> None:
        mol = read_sdf_string(_SAMPLE_SDF)[0]
        fp = tmp_path / "out.sdf"
        write_sdf(mol, fp)
        result = read_sdf(fp)
        assert len(result) == 1
        assert result[0].atom_array.n_atoms == mol.atom_array.n_atoms

    def test_coordinates_preserved(self, tmp_path: Path) -> None:
        mol = read_sdf_string(_SAMPLE_SDF)[0]
        fp = tmp_path / "out.sdf"
        write_sdf(mol, fp)
        rt = read_sdf(fp)[0]
        # SDF coordinates are written with 4 decimal places, so use approx.
        for i in range(mol.atom_array.n_atoms):
            assert tuple(rt.atom_array.coords[i]) == pytest.approx(
                tuple(mol.atom_array.coords[i]), abs=1e-3
            )

    def test_elements_preserved(self, tmp_path: Path) -> None:
        mol = read_sdf_string(_SAMPLE_SDF)[0]
        fp = tmp_path / "out.sdf"
        write_sdf(mol, fp)
        rt = read_sdf(fp)[0]
        assert list(rt.atom_array.element) == list(mol.atom_array.element)

    def test_title_preserved(self, tmp_path: Path) -> None:
        mol = read_sdf_string(_SAMPLE_SDF)[0]
        fp = tmp_path / "out.sdf"
        write_sdf(mol, fp)
        rt = read_sdf(fp)[0]
        assert rt.metadata.get("title") == "ligand-name"

    def test_properties_preserved(self, tmp_path: Path) -> None:
        mol = read_sdf_string(_SAMPLE_SDF)[0]
        fp = tmp_path / "out.sdf"
        write_sdf(mol, fp)
        rt = read_sdf(fp)[0]
        assert rt.metadata.get("properties") == {
            "Source": "Computed",
            "MultiLine": "line1\nline2",
        }

    def test_list_of_proteins_round_trips(self, tmp_path: Path) -> None:
        mols = read_sdf_string(_MULTI_SDF)
        fp = tmp_path / "multi.sdf"
        write_sdf(mols, fp)
        rt = read_sdf(fp)
        assert len(rt) == len(mols)
        assert [m.metadata["title"] for m in rt] == ["ligand-name", "second-mol"]

    def test_writes_via_dispatcher_save(self, tmp_path: Path) -> None:
        from molforge.io import save

        mol = read_sdf_string(_SAMPLE_SDF)[0]
        fp = tmp_path / "via_save.sdf"
        save(mol, fp)
        # File exists and reads back.
        assert fp.is_file()
        assert read_sdf(fp)[0].atom_array.n_atoms == 3
