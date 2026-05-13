"""Tests for the PDB reader and writer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molforge.core import Protein
from molforge.io import (
    PDBParseError,
    PDBWriteError,
    read_pdb,
    read_pdb_string,
    write_pdb,
    write_pdb_string,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "pdb"


class TestReadDipeptide:
    """The dipeptide fixture covers the common case end-to-end."""

    @pytest.fixture
    def protein(self) -> Protein:
        return read_pdb(FIXTURES / "dipeptide.pdb")

    def test_returns_protein(self, protein: Protein) -> None:
        assert isinstance(protein, Protein)

    def test_atom_count(self, protein: Protein) -> None:
        # 9 protein atoms + 1 water = 10
        assert protein.n_atoms == 10

    def test_chain_count(self, protein: Protein) -> None:
        assert protein.n_chains == 2  # protein chain A + water chain W

    def test_residue_count(self, protein: Protein) -> None:
        assert protein.n_residues == 3  # ALA, GLY, HOH

    def test_chain_ids(self, protein: Protein) -> None:
        chain_ids = [c.chain_id for c in protein.chains]
        assert chain_ids == ["A", "W"]

    def test_sequence(self, protein: Protein) -> None:
        # Chain A: AG, chain W: water (skipped)
        assert protein.sequence == "AG"

    def test_atom_names_first_residue(self, protein: Protein) -> None:
        ala = protein["A"][1]
        names = [a.name for a in ala]
        assert names == ["N", "CA", "C", "O", "CB"]

    def test_elements_assigned(self, protein: Protein) -> None:
        arr = protein.atom_array
        # First atom is N of Ala
        assert str(arr.element[0]) == "N"
        # Last atom is water O
        assert str(arr.element[-1]) == "O"

    def test_coordinates(self, protein: Protein) -> None:
        ca = protein["A"][1]["CA"]
        np.testing.assert_allclose(ca.coord, [-0.001, 0.064, -0.491], atol=1e-3)

    def test_b_factors(self, protein: Protein) -> None:
        arr = protein.atom_array
        # All protein atoms have B=20.00; water has B=30.00
        assert np.all(arr.b_factor[:9] == pytest.approx(20.0))
        assert arr.b_factor[-1] == pytest.approx(30.0)

    def test_entity_type_classification(self, protein: Protein) -> None:
        arr = protein.atom_array
        # First 9 atoms are protein
        assert all(str(t) == "protein" for t in arr.entity_type[:9])
        # Last atom is water
        assert str(arr.entity_type[-1]) == "water"

    def test_hetatm_record_type(self, protein: Protein) -> None:
        arr = protein.atom_array
        assert str(arr.record_type[-1]) == "HETATM"
        assert all(str(t) == "ATOM" for t in arr.record_type[:9])

    def test_metadata_populated(self, protein: Protein) -> None:
        assert "title" in protein.metadata
        assert "ALA-GLY" in str(protein.metadata["title"]).upper()
        assert protein.metadata.get("resolution") == pytest.approx(1.0)

    def test_name_from_filename(self, protein: Protein) -> None:
        assert protein.name == "dipeptide"


class TestReadMultiModel:
    def test_loads_all_models_by_default(self) -> None:
        p = read_pdb(FIXTURES / "multi_model.pdb")
        assert p.n_atoms == 4  # 2 atoms x 2 models
        # Two distinct model_ids
        assert {int(m) for m in p.atom_array.model_id} == {1, 2}

    def test_load_specific_model(self) -> None:
        p = read_pdb(FIXTURES / "multi_model.pdb", model=2)
        assert p.n_atoms == 2
        assert np.all(p.atom_array.model_id == 2)

    def test_load_missing_model(self) -> None:
        p = read_pdb(FIXTURES / "multi_model.pdb", model=99)
        assert p.n_atoms == 0


class TestReadAltloc:
    def test_highest_occupancy_default(self) -> None:
        p = read_pdb(FIXTURES / "with_altloc.pdb")
        # 4 backbone + 1 CB + 1 OG (winning altlocs) = 6
        assert p.n_atoms == 6
        # The kept CB should have occupancy 0.60 (altloc A)
        cb = p["A"][1]["CB"]
        assert cb.occupancy == pytest.approx(0.60)

    def test_first_strategy(self) -> None:
        p = read_pdb(FIXTURES / "with_altloc.pdb", altloc="first")
        assert p.n_atoms == 6
        cb = p["A"][1]["CB"]
        # "A" altloc was first in the file
        assert cb.occupancy == pytest.approx(0.60)

    def test_keep_all(self) -> None:
        p = read_pdb(FIXTURES / "with_altloc.pdb", altloc="all")
        assert p.n_atoms == 8  # both altlocs retained

    def test_specific_altloc(self) -> None:
        p = read_pdb(FIXTURES / "with_altloc.pdb", altloc="B")
        assert p.n_atoms == 6
        cb = p["A"][1]["CB"]
        assert cb.occupancy == pytest.approx(0.40)


class TestReadInsertionCodes:
    def test_insertion_code_creates_separate_residue(self) -> None:
        p = read_pdb(FIXTURES / "with_insertion_code.pdb")
        # 3 residues: ALA 1, GLY 1A, VAL 2
        assert p.n_residues == 3
        residues = list(p["A"])
        assert residues[0].name == "ALA"
        assert residues[0].seq_id == 1
        assert residues[0].insertion_code == ""
        assert residues[1].name == "GLY"
        assert residues[1].seq_id == 1
        assert residues[1].insertion_code == "A"
        assert residues[2].name == "VAL"
        assert residues[2].seq_id == 2


class TestReadHydrogens:
    def test_include_hydrogens_default(self) -> None:
        text = (
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N  \n"
            "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 20.00           C  \n"
            "ATOM      3  H   ALA A   1       0.500   0.500   0.500  1.00 20.00           H  \n"
            "END\n"
        )
        p = read_pdb_string(text)
        assert p.n_atoms == 3

    def test_exclude_hydrogens(self) -> None:
        text = (
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N  \n"
            "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 20.00           C  \n"
            "ATOM      3  H   ALA A   1       0.500   0.500   0.500  1.00 20.00           H  \n"
            "END\n"
        )
        p = read_pdb_string(text, include_hydrogens=False)
        assert p.n_atoms == 2


class TestReadErrors:
    def test_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            read_pdb("/nonexistent/path.pdb")

    def test_bad_coordinates_raises(self) -> None:
        bad = "ATOM      1  N   ALA A   1     XXXX   0.000   0.000  1.00 20.00           N  \nEND\n"
        with pytest.raises(PDBParseError, match="coordinates"):
            read_pdb_string(bad)

    def test_empty_file_yields_empty_protein(self) -> None:
        p = read_pdb_string("")
        assert p.n_atoms == 0

    def test_only_header_yields_empty_protein(self) -> None:
        p = read_pdb_string("HEADER    EMPTY\nEND\n")
        assert p.n_atoms == 0


class TestWriteRoundTrip:
    """Read a file, write it back, read it again — content should match."""

    @pytest.mark.parametrize("fixture", ["dipeptide.pdb", "with_insertion_code.pdb"])
    def test_round_trip_preserves_atoms(self, fixture: str, tmp_path: Path) -> None:
        original = read_pdb(FIXTURES / fixture)
        out_path = tmp_path / "roundtrip.pdb"
        write_pdb(original, out_path)
        reloaded = read_pdb(out_path)
        assert reloaded.n_atoms == original.n_atoms
        np.testing.assert_allclose(
            reloaded.atom_array.coords, original.atom_array.coords, atol=1e-3
        )
        assert list(reloaded.atom_array.atom_name) == list(original.atom_array.atom_name)
        assert list(reloaded.atom_array.residue_name) == list(original.atom_array.residue_name)

    def test_round_trip_preserves_residue_ids_and_chains(self, tmp_path: Path) -> None:
        original = read_pdb(FIXTURES / "dipeptide.pdb")
        out = tmp_path / "rt.pdb"
        write_pdb(original, out)
        reloaded = read_pdb(out)
        assert list(reloaded.atom_array.chain_id) == list(original.atom_array.chain_id)
        assert list(reloaded.atom_array.residue_id) == list(original.atom_array.residue_id)


class TestWriteFormat:
    def test_writes_atom_records(self) -> None:
        original = read_pdb(FIXTURES / "dipeptide.pdb")
        text = write_pdb_string(original)
        # Must contain ATOM records
        assert "ATOM" in text
        # First ATOM line should have right columns for N of ALA
        atom_lines = [ln for ln in text.splitlines() if ln.startswith("ATOM")]
        assert atom_lines
        first = atom_lines[0]
        assert first[17:20] == "ALA"
        assert first[21] == "A"

    def test_writes_end_record(self) -> None:
        p = read_pdb(FIXTURES / "dipeptide.pdb")
        assert write_pdb_string(p).rstrip().endswith("END")

    def test_writes_no_end_if_disabled(self) -> None:
        p = read_pdb(FIXTURES / "dipeptide.pdb")
        text = write_pdb_string(p, write_end=False)
        assert not text.rstrip().endswith("END")

    def test_writes_hetatm_for_water(self) -> None:
        p = read_pdb(FIXTURES / "dipeptide.pdb")
        text = write_pdb_string(p)
        assert "HETATM" in text


class TestWriteLimits:
    def test_oversized_raises(self) -> None:
        from molforge.core import AtomArray, Protein

        big = Protein(AtomArray(100_001))
        with pytest.raises(PDBWriteError, match="99,999"):
            write_pdb_string(big)
