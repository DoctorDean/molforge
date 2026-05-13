"""Tests for the mmCIF / PDBx reader and writer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molforge.core import Protein
from molforge.io import (
    CIFParseError,
    read_cif,
    read_cif_string,
    write_cif,
    write_cif_string,
)
from molforge.io.mmcif import _tokenize

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "cif"
PDB_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "pdb"


class TestTokenizer:
    def test_simple(self) -> None:
        toks = list(_tokenize("data_foo\n_a.b 1\n"))
        assert toks == ["data_foo", "_a.b", "1"]

    def test_quoted_string(self) -> None:
        toks = list(_tokenize("_a.b 'hello world'\n"))
        assert toks == ["_a.b", "hello world"]

    def test_comments_stripped(self) -> None:
        toks = list(_tokenize("data_x  # comment goes here\n_a 1\n"))
        assert toks == ["data_x", "_a", "1"]

    def test_dot_and_question_preserved(self) -> None:
        toks = list(_tokenize("loop_\n_x.a\n_x.b\n. ?\n"))
        assert "." in toks and "?" in toks

    def test_semicolon_text_block(self) -> None:
        text = "_long\n;line one\nline two\n;\n"
        toks = list(_tokenize(text))
        # The token following the key is the joined multi-line text
        assert toks[0] == "_long"
        assert "line one" in toks[1]
        assert "line two" in toks[1]


class TestReadDipeptide:
    @pytest.fixture
    def protein(self) -> Protein:
        return read_cif(FIXTURES / "dipeptide.cif")

    def test_returns_protein(self, protein: Protein) -> None:
        assert isinstance(protein, Protein)

    def test_atom_count(self, protein: Protein) -> None:
        assert protein.n_atoms == 10  # 9 protein + 1 water

    def test_chain_count(self, protein: Protein) -> None:
        assert protein.n_chains == 2  # A + W

    def test_residue_count(self, protein: Protein) -> None:
        assert protein.n_residues == 3  # ALA, GLY, HOH

    def test_sequence(self, protein: Protein) -> None:
        assert protein.sequence == "AG"

    def test_metadata_pdb_id(self, protein: Protein) -> None:
        assert protein.metadata.get("pdb_id") == "DIPE"

    def test_metadata_title(self, protein: Protein) -> None:
        assert "Ala-Gly" in str(protein.metadata.get("title", ""))

    def test_metadata_resolution(self, protein: Protein) -> None:
        assert protein.metadata.get("resolution") == pytest.approx(1.0)

    def test_metadata_method(self, protein: Protein) -> None:
        assert protein.metadata.get("experimental_method") == "THEORETICAL MODEL"

    def test_coordinates(self, protein: Protein) -> None:
        ca = protein["A"][1]["CA"]
        np.testing.assert_allclose(ca.coord, [-0.001, 0.064, -0.491], atol=1e-3)

    def test_entity_type_classification(self, protein: Protein) -> None:
        arr = protein.atom_array
        assert all(str(t) == "protein" for t in arr.entity_type[:9])
        assert str(arr.entity_type[-1]) == "water"

    def test_hetatm_marked(self, protein: Protein) -> None:
        arr = protein.atom_array
        assert str(arr.record_type[-1]) == "HETATM"


class TestReadEdgeCases:
    def test_empty_string(self) -> None:
        p = read_cif_string("")
        assert p.n_atoms == 0

    def test_only_header_yields_empty(self) -> None:
        p = read_cif_string("data_foo\n_entry.id foo\n")
        assert p.n_atoms == 0

    def test_missing_required_columns_raises(self) -> None:
        bad = "data_x\nloop_\n_atom_site.id\n_atom_site.type_symbol\n1 C\n"
        with pytest.raises(CIFParseError, match="missing required columns"):
            read_cif_string(bad)


class TestRoundTrip:
    def test_round_trip_atoms_preserved(self, tmp_path: Path) -> None:
        original = read_cif(FIXTURES / "dipeptide.cif")
        out = tmp_path / "rt.cif"
        write_cif(original, out)
        reloaded = read_cif(out)
        assert reloaded.n_atoms == original.n_atoms
        np.testing.assert_allclose(
            reloaded.atom_array.coords, original.atom_array.coords, atol=1e-3
        )
        assert list(reloaded.atom_array.atom_name) == list(original.atom_array.atom_name)
        assert list(reloaded.atom_array.residue_name) == list(original.atom_array.residue_name)

    def test_pdb_to_cif_round_trip(self, tmp_path: Path) -> None:
        """Read PDB, write CIF, read CIF — content should survive."""
        from molforge.io import read_pdb

        pdb = read_pdb(PDB_FIXTURES / "dipeptide.pdb")
        cif_path = tmp_path / "from_pdb.cif"
        write_cif(pdb, cif_path)
        cif_back = read_cif(cif_path)
        assert cif_back.n_atoms == pdb.n_atoms
        np.testing.assert_allclose(cif_back.atom_array.coords, pdb.atom_array.coords, atol=1e-3)
        assert cif_back.sequence == pdb.sequence


class TestWriteFormat:
    def test_emits_block_header(self) -> None:
        p = read_cif(FIXTURES / "dipeptide.cif")
        text = write_cif_string(p)
        assert text.startswith("data_")

    def test_emits_atom_site_loop(self) -> None:
        p = read_cif(FIXTURES / "dipeptide.cif")
        text = write_cif_string(p)
        assert "loop_" in text
        assert "_atom_site.Cartn_x" in text

    def test_quotes_title_with_spaces(self) -> None:
        p = read_cif(FIXTURES / "dipeptide.cif")
        text = write_cif_string(p)
        # The title contains spaces and must be quoted.
        assert "'Ala-Gly" in text


class TestDispatch:
    def test_load_cif_by_extension(self) -> None:
        from molforge.io import load

        p = load(FIXTURES / "dipeptide.cif")
        assert isinstance(p, Protein)
        assert p.n_atoms == 10

    def test_save_cif_by_extension(self, tmp_path: Path) -> None:
        from molforge.io import load, save

        p = load(FIXTURES / "dipeptide.cif")
        out = tmp_path / "out.cif"
        save(p, out)
        assert out.exists()
        assert "data_" in out.read_text()
