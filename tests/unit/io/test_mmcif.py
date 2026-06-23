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


# ======================================================================
# Round-trip fidelity audit
# ======================================================================
#
# These tests codify the fixes from the pre-1.0 mmCIF writer audit.
# Each test is named after the specific gap it guards. The audit found
# five concrete issues in the original writer:
#
#   1. model_id == 0 was clobbered to 1 (write turned 0->1 because of
#      ``int(model_id) or 1``). Every single-model PDB round-tripped
#      with model_id changed.
#   2. Partial / non-integer charges were truncated by int() — a
#      charge of -0.297 (typical PDBQT/PQR value) became 0.
#   3. metadata['classification'] and metadata['deposition_date']
#      weren't emitted at all.
#   4. _entry.id and the data_<id> block name could disagree when
#      protein.name != metadata[pdb_id], silently rewriting both on
#      round-trip.
#   5. serial == 0 was clobbered to (i+1) — latent twin of #1.
#
# The block at the end (``TestFixtureSweep``) parametrizes over every
# PDB fixture in the repo so future regressions are caught by the
# whole corpus, not just the single dipeptide.cif fixture the older
# tests used.


class TestModelIdRoundTrip:
    """model_id must round-trip verbatim, including 0.

    read_pdb uses model_id=0 as the implicit value for files without
    MODEL records. Pre-audit, the CIF writer's ``or 1`` clobbered this
    to 1 on write, so every single-model PDB lost the model_id=0
    convention on round-trip.
    """

    def test_zero_model_id_preserved(self) -> None:
        from molforge.core import AtomArray, Protein

        arr = AtomArray(2)
        arr.coords[:] = [[0, 0, 0], [1, 1, 1]]
        arr.element[:] = ["N", "C"]
        arr.atom_name[:] = ["N", "CA"]
        arr.residue_name[:] = ["ALA", "ALA"]
        arr.residue_id[:] = [1, 1]
        arr.chain_id[:] = ["A", "A"]
        arr.serial[:] = [1, 2]
        arr.occupancy[:] = [1.0, 1.0]
        arr.b_factor[:] = [10.0, 10.0]
        arr.model_id[:] = [0, 0]  # the value pre-audit silently became 1
        p = Protein(arr)

        rt = read_cif_string(write_cif_string(p))
        assert list(rt.atom_array.model_id) == [0, 0]

    def test_nonzero_model_id_preserved(self) -> None:
        from molforge.core import AtomArray, Protein

        arr = AtomArray(2)
        arr.coords[:] = [[0, 0, 0], [1, 1, 1]]
        arr.element[:] = ["N", "C"]
        arr.atom_name[:] = ["N", "CA"]
        arr.residue_name[:] = ["ALA", "ALA"]
        arr.residue_id[:] = [1, 1]
        arr.chain_id[:] = ["A", "A"]
        arr.serial[:] = [1, 2]
        arr.occupancy[:] = [1.0, 1.0]
        arr.b_factor[:] = [10.0, 10.0]
        arr.model_id[:] = [3, 3]
        p = Protein(arr)

        rt = read_cif_string(write_cif_string(p))
        assert list(rt.atom_array.model_id) == [3, 3]


class TestSerialRoundTrip:
    """serial number round-trip, including the zero edge case.

    The pre-audit writer used ``int(serial) or (i+1)``, which silently
    replaced any 0 serial with a synthesized index. The fix only
    synthesizes when serial <= 0 AND the protein didn't set a real
    value — but the most useful guarantee for tests is that any
    positive serial passes through unchanged.
    """

    def test_nonconsecutive_serials_preserved(self) -> None:
        from molforge.core import AtomArray, Protein

        arr = AtomArray(4)
        arr.coords[:] = [*np.eye(3, dtype=np.float32).tolist(), [1.0, 1.0, 1.0]]
        arr.element[:] = ["N", "C", "C", "O"]
        arr.atom_name[:] = ["N", "CA", "C", "O"]
        arr.residue_name[:] = ["ALA"] * 4
        arr.residue_id[:] = [1] * 4
        arr.chain_id[:] = ["A"] * 4
        arr.serial[:] = [10, 20, 30, 40]  # not 1..N
        arr.occupancy[:] = [1.0] * 4
        arr.b_factor[:] = [10.0] * 4
        p = Protein(arr)

        rt = read_cif_string(write_cif_string(p))
        assert list(rt.atom_array.serial) == [10, 20, 30, 40]


class TestChargeRoundTrip:
    """Partial charges from PDBQT/PQR sources must survive a CIF
    round-trip.

    Pre-audit, the writer emitted ``f"{int(charge):d}"`` which
    truncated -0.297 to 0 and -1.5 to -1. The fix uses ``f"{charge:.4f}"``
    so 4 decimal places of precision survive.
    """

    def test_partial_charges_preserved(self) -> None:
        from molforge.core import AtomArray, Protein

        arr = AtomArray(4)
        arr.coords[:] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        arr.element[:] = ["N", "C", "O", "S"]
        arr.atom_name[:] = ["N", "CA", "O", "S"]
        arr.residue_name[:] = ["ALA"] * 4
        arr.residue_id[:] = [1] * 4
        arr.chain_id[:] = ["A"] * 4
        arr.serial[:] = [1, 2, 3, 4]
        arr.occupancy[:] = [1.0] * 4
        arr.b_factor[:] = [10.0] * 4
        # Typical PDBQT / PQR partial charge values.
        arr.charge[:] = [-0.297, 0.045, 0.5, -1.5]
        p = Protein(arr)

        rt = read_cif_string(write_cif_string(p))
        np.testing.assert_allclose(
            rt.atom_array.charge,
            [-0.297, 0.045, 0.5, -1.5],
            atol=1e-4,
        )

    def test_integer_charges_still_round_trip(self) -> None:
        from molforge.core import AtomArray, Protein

        arr = AtomArray(2)
        arr.coords[:] = [[0, 0, 0], [1, 1, 1]]
        arr.element[:] = ["N", "O"]
        arr.atom_name[:] = ["N1", "O1"]
        arr.residue_name[:] = ["ALA", "ALA"]
        arr.residue_id[:] = [1, 1]
        arr.chain_id[:] = ["A", "A"]
        arr.serial[:] = [1, 2]
        arr.occupancy[:] = [1.0, 1.0]
        arr.b_factor[:] = [10.0, 10.0]
        arr.charge[:] = [1.0, -2.0]
        p = Protein(arr)

        rt = read_cif_string(write_cif_string(p))
        np.testing.assert_allclose(rt.atom_array.charge, [1.0, -2.0], atol=1e-4)

    def test_zero_charge_emits_sentinel_round_trips(self) -> None:
        """A zero charge writes as ``?`` (mmCIF 'unknown') and reads
        back as 0.0. This is the "no charge information" path."""
        from molforge.core import AtomArray, Protein

        arr = AtomArray(1)
        arr.coords[:] = [[0, 0, 0]]
        arr.element[:] = ["N"]
        arr.atom_name[:] = ["N"]
        arr.residue_name[:] = ["ALA"]
        arr.residue_id[:] = [1]
        arr.chain_id[:] = ["A"]
        arr.serial[:] = [1]
        arr.occupancy[:] = [1.0]
        arr.b_factor[:] = [10.0]
        arr.charge[:] = [0.0]
        p = Protein(arr)

        text = write_cif_string(p)
        # The sentinel ? appears in the atom row.
        assert " ? " in text or text.rstrip().endswith(" ?")
        rt = read_cif_string(text)
        assert float(rt.atom_array.charge[0]) == pytest.approx(0.0)


class TestMetadataRoundTrip:
    """metadata['classification'] and metadata['deposition_date'] are
    PDB HEADER fields that previously didn't survive a PDB -> CIF ->
    in-memory round-trip — neither was emitted by the writer. The fix
    emits ``_struct_keywords.text`` for classification and
    ``_pdbx_database_status.recvd_initial_deposition_date`` for the
    date."""

    def test_classification_preserved(self) -> None:
        from molforge.core import AtomArray, Protein
        from molforge.core import metadata_keys as mk

        arr = AtomArray(1)
        arr.coords[:] = [[0, 0, 0]]
        arr.element[:] = ["N"]
        arr.atom_name[:] = ["N"]
        arr.residue_name[:] = ["ALA"]
        arr.residue_id[:] = [1]
        arr.chain_id[:] = ["A"]
        arr.serial[:] = [1]
        arr.occupancy[:] = [1.0]
        arr.b_factor[:] = [10.0]
        p = Protein(arr)
        p.metadata = {mk.PDB_ID: "1ABC", mk.CLASSIFICATION: "OXIDOREDUCTASE"}

        rt = read_cif_string(write_cif_string(p))
        assert rt.metadata.get(mk.CLASSIFICATION) == "OXIDOREDUCTASE"

    def test_deposition_date_preserved(self) -> None:
        from molforge.core import AtomArray, Protein
        from molforge.core import metadata_keys as mk

        arr = AtomArray(1)
        arr.coords[:] = [[0, 0, 0]]
        arr.element[:] = ["N"]
        arr.atom_name[:] = ["N"]
        arr.residue_name[:] = ["ALA"]
        arr.residue_id[:] = [1]
        arr.chain_id[:] = ["A"]
        arr.serial[:] = [1]
        arr.occupancy[:] = [1.0]
        arr.b_factor[:] = [10.0]
        p = Protein(arr)
        p.metadata = {mk.PDB_ID: "1ABC", mk.DEPOSITION_DATE: "2024-01-15"}

        rt = read_cif_string(write_cif_string(p))
        assert rt.metadata.get(mk.DEPOSITION_DATE) == "2024-01-15"

    def test_title_with_spaces_preserved(self) -> None:
        from molforge.core import AtomArray, Protein
        from molforge.core import metadata_keys as mk

        arr = AtomArray(1)
        arr.coords[:] = [[0, 0, 0]]
        arr.element[:] = ["N"]
        arr.atom_name[:] = ["N"]
        arr.residue_name[:] = ["ALA"]
        arr.residue_id[:] = [1]
        arr.chain_id[:] = ["A"]
        arr.serial[:] = [1]
        arr.occupancy[:] = [1.0]
        arr.b_factor[:] = [10.0]
        p = Protein(arr)
        p.metadata = {
            mk.PDB_ID: "1ABC",
            mk.TITLE: "STRUCTURE OF MYOGLOBIN AT 1.5 A RESOLUTION",
        }

        rt = read_cif_string(write_cif_string(p))
        assert rt.metadata.get(mk.TITLE) == "STRUCTURE OF MYOGLOBIN AT 1.5 A RESOLUTION"

    def test_resolution_preserved(self) -> None:
        from molforge.core import AtomArray, Protein
        from molforge.core import metadata_keys as mk

        arr = AtomArray(1)
        arr.coords[:] = [[0, 0, 0]]
        arr.element[:] = ["N"]
        arr.atom_name[:] = ["N"]
        arr.residue_name[:] = ["ALA"]
        arr.residue_id[:] = [1]
        arr.chain_id[:] = ["A"]
        arr.serial[:] = [1]
        arr.occupancy[:] = [1.0]
        arr.b_factor[:] = [10.0]
        p = Protein(arr)
        p.metadata = {mk.PDB_ID: "1ABC", mk.RESOLUTION: 1.85}

        rt = read_cif_string(write_cif_string(p))
        assert float(rt.metadata[mk.RESOLUTION]) == pytest.approx(1.85, abs=0.01)


class TestEntryIdAndBlockNameConsistency:
    """The writer must use the SAME identifier for both ``data_<id>``
    (block name) and ``_entry.id``. Pre-audit, these could come from
    different sources (``protein.name`` for the block, ``metadata[pdb_id]``
    for ``_entry.id``) and disagree — silently rewriting Protein.name
    and metadata[pdb_id] on read.
    """

    def test_pdb_id_with_whitespace_preserved(self) -> None:
        """An identifier with embedded whitespace (which read_pdb
        tolerates from malformed HEADER lines) must survive the
        round-trip via the quoted _entry.id, even though the block
        name has to substitute underscores."""
        from molforge.core import AtomArray, Protein
        from molforge.core import metadata_keys as mk

        arr = AtomArray(1)
        arr.coords[:] = [[0, 0, 0]]
        arr.element[:] = ["N"]
        arr.atom_name[:] = ["N"]
        arr.residue_name[:] = ["ALA"]
        arr.residue_id[:] = [1]
        arr.chain_id[:] = ["A"]
        arr.serial[:] = [1]
        arr.occupancy[:] = [1.0]
        arr.b_factor[:] = [10.0]
        p = Protein(arr)
        p.metadata = {mk.PDB_ID: "Q TE"}

        text = write_cif_string(p)
        # Block name has underscore; _entry.id is quoted with the space.
        assert "data_Q_TE" in text
        assert "_entry.id  'Q TE'" in text

        rt = read_cif_string(text)
        # Whitespace is preserved in the metadata after round-trip.
        assert rt.metadata[mk.PDB_ID] == "Q TE"

    def test_no_pdb_id_does_not_manufacture_one(self) -> None:
        """When the input has no pdb_id, the round-trip must not
        manufacture one from the block name. The writer emits the
        ``_entry.id .`` sentinel; the reader keeps pdb_id absent."""
        from molforge.core import AtomArray, Protein
        from molforge.core import metadata_keys as mk

        arr = AtomArray(1)
        arr.coords[:] = [[0, 0, 0]]
        arr.element[:] = ["N"]
        arr.atom_name[:] = ["N"]
        arr.residue_name[:] = ["ALA"]
        arr.residue_id[:] = [1]
        arr.chain_id[:] = ["A"]
        arr.serial[:] = [1]
        arr.occupancy[:] = [1.0]
        arr.b_factor[:] = [10.0]
        p = Protein(arr, name="my_local_structure")  # name only, no pdb_id

        text = write_cif_string(p)
        # The sentinel signals "no real _entry.id".
        assert "_entry.id  ." in text

        rt = read_cif_string(text)
        # pdb_id is NOT present in the round-tripped metadata.
        assert mk.PDB_ID not in rt.metadata

    def test_pdb_id_present_used_for_both(self) -> None:
        """When pdb_id is set, both block name and _entry.id use it
        (not protein.name, which is a fallback for the no-pdb_id case)."""
        from molforge.core import AtomArray, Protein
        from molforge.core import metadata_keys as mk

        arr = AtomArray(1)
        arr.coords[:] = [[0, 0, 0]]
        arr.element[:] = ["N"]
        arr.atom_name[:] = ["N"]
        arr.residue_name[:] = ["ALA"]
        arr.residue_id[:] = [1]
        arr.chain_id[:] = ["A"]
        arr.serial[:] = [1]
        arr.occupancy[:] = [1.0]
        arr.b_factor[:] = [10.0]
        # The user has set name and pdb_id to different values; pdb_id wins.
        p = Protein(arr, name="local_id")
        p.metadata = {mk.PDB_ID: "1ABC"}

        text = write_cif_string(p)
        assert "data_1ABC" in text
        assert "_entry.id  1ABC" in text

        rt = read_cif_string(text)
        assert rt.metadata[mk.PDB_ID] == "1ABC"

    def test_documented_limitation_name_recovered_from_pdb_id(self) -> None:
        """mmCIF carries only one identifier slot (_entry.id), so a
        Protein with ``name != metadata[pdb_id]`` loses ``name`` on
        round-trip — ``name`` is recovered from ``pdb_id``. This test
        codifies that documented behaviour so callers know what to
        expect."""
        from molforge.core import AtomArray, Protein
        from molforge.core import metadata_keys as mk

        arr = AtomArray(1)
        arr.coords[:] = [[0, 0, 0]]
        arr.element[:] = ["N"]
        arr.atom_name[:] = ["N"]
        arr.residue_name[:] = ["ALA"]
        arr.residue_id[:] = [1]
        arr.chain_id[:] = ["A"]
        arr.serial[:] = [1]
        arr.occupancy[:] = [1.0]
        arr.b_factor[:] = [10.0]
        p = Protein(arr, name="custom_local_name")
        p.metadata = {mk.PDB_ID: "1ABC"}

        rt = read_cif_string(write_cif_string(p))
        # pdb_id survives; name is overwritten with pdb_id (documented).
        assert rt.metadata[mk.PDB_ID] == "1ABC"
        assert rt.name == "1ABC"


class TestInsertionCodeRoundTrip:
    """Insertion codes (the single-letter suffix on residue ids in
    antibody numbering, e.g. residue 100A) must survive round-trip."""

    def test_insertion_code_preserved(self) -> None:
        from molforge.core import AtomArray, Protein

        arr = AtomArray(3)
        arr.coords[:] = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
        arr.element[:] = ["N", "C", "O"]
        arr.atom_name[:] = ["N", "CA", "O"]
        arr.residue_name[:] = ["ALA", "ALA", "ALA"]
        arr.residue_id[:] = [100, 100, 101]
        arr.insertion_code[:] = ["", "A", ""]  # second atom is residue 100A
        arr.chain_id[:] = ["A"] * 3
        arr.serial[:] = [1, 2, 3]
        arr.occupancy[:] = [1.0] * 3
        arr.b_factor[:] = [10.0] * 3
        p = Protein(arr)

        rt = read_cif_string(write_cif_string(p))
        assert [str(c).strip() for c in rt.atom_array.insertion_code] == [
            "",
            "A",
            "",
        ]


class TestAltlocRoundTrip:
    """Altloc behavior — documented limitation.

    By default, ``read_cif_string`` calls ``_resolve_altlocs`` which
    picks one altloc per atom and drops the label. To round-trip
    altloc labels, callers must pass ``altloc="all"``.
    """

    def test_altloc_preserved_with_all_strategy(self) -> None:
        from molforge.core import AtomArray, Protein

        arr = AtomArray(3)
        arr.coords[:] = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
        arr.element[:] = ["N", "C", "C"]
        arr.atom_name[:] = ["N", "CA", "CA"]
        arr.residue_name[:] = ["SER", "SER", "SER"]
        arr.residue_id[:] = [1, 1, 1]
        arr.altloc[:] = ["", "A", "B"]  # alternate CA conformations
        arr.chain_id[:] = ["A"] * 3
        arr.serial[:] = [1, 2, 3]
        arr.occupancy[:] = [1.0, 0.6, 0.4]
        arr.b_factor[:] = [10.0] * 3
        p = Protein(arr)

        rt = read_cif_string(write_cif_string(p), altloc="all")
        altlocs = [str(a).strip() for a in rt.atom_array.altloc]
        assert altlocs == ["", "A", "B"]

    def test_altloc_collapsed_by_default(self) -> None:
        """Default altloc='highest_occupancy' collapses alternates to
        the highest-occupancy one. This is the documented behaviour;
        users wanting full altloc round-trip must pass altloc='all'."""
        from molforge.core import AtomArray, Protein

        arr = AtomArray(3)
        arr.coords[:] = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
        arr.element[:] = ["N", "C", "C"]
        arr.atom_name[:] = ["N", "CA", "CA"]
        arr.residue_name[:] = ["SER", "SER", "SER"]
        arr.residue_id[:] = [1, 1, 1]
        arr.altloc[:] = ["", "A", "B"]
        arr.chain_id[:] = ["A"] * 3
        arr.serial[:] = [1, 2, 3]
        arr.occupancy[:] = [1.0, 0.6, 0.4]
        arr.b_factor[:] = [10.0] * 3
        p = Protein(arr)

        rt = read_cif_string(write_cif_string(p))  # default
        # The B conformer (0.4 occ) is dropped; one CA remains.
        n_ca = sum(1 for a in rt.atom_array.atom_name if str(a).strip() == "CA")
        assert n_ca == 1


class TestFixtureSweep:
    """PDB -> CIF -> in-memory round-trip over every PDB fixture.

    For each fixture, the critical atom-array fields and the metadata
    keys we promise to preserve must survive the trip. This is the
    regression net: a future change that breaks fidelity on any
    fixture will fail here.

    Documented exclusions (the same ones the audit codified above):
      - ``Protein.name``: only carried by ``_entry.id``, which round-
        trips as ``metadata[pdb_id]``. See
        TestEntryIdAndBlockNameConsistency.
      - ``altloc``: collapsed by default-strategy
        ``_resolve_altlocs``. To preserve, callers pass
        ``altloc='all'``. See TestAltlocRoundTrip.
      - Empty-string metadata values: dropped (semantically
        equivalent to missing).
      - ``entity_type``: re-derived from residue name + atom count
        by the reader, not transported through CIF. Equivalent
        residues will get equivalent entity_type by construction.
    """

    @pytest.mark.parametrize(
        "pdb_name",
        sorted(p.name for p in PDB_FIXTURES.glob("*.pdb")),
    )
    def test_critical_fields_round_trip(self, pdb_name: str) -> None:
        from molforge.io import read_pdb

        original = read_pdb(PDB_FIXTURES / pdb_name)
        rt = read_cif_string(write_cif_string(original), altloc="all")

        # 1. Atom count.
        assert rt.atom_array.n_atoms == original.atom_array.n_atoms, (
            f"{pdb_name}: atom count changed"
        )

        # 2. Coordinates (3-decimal precision matches PDB).
        np.testing.assert_allclose(
            rt.atom_array.coords,
            original.atom_array.coords,
            atol=1e-3,
            err_msg=f"{pdb_name}: coordinates drifted",
        )

        # 3. String-typed fields with full equality (after .strip()).
        for field_name in ("element", "atom_name", "residue_name", "chain_id"):
            orig = [str(x).strip() for x in getattr(original.atom_array, field_name)]
            after = [str(x).strip() for x in getattr(rt.atom_array, field_name)]
            assert orig == after, f"{pdb_name}: {field_name} differs"

        # 4. Integer-typed fields.
        for field_name in ("residue_id", "model_id", "serial"):
            assert list(getattr(original.atom_array, field_name)) == list(
                getattr(rt.atom_array, field_name)
            ), f"{pdb_name}: {field_name} differs"

        # 5. Float-typed fields with the writer's documented
        #    precision (2 decimals for occupancy / b_factor).
        for field_name, atol in (("occupancy", 0.01), ("b_factor", 0.01)):
            np.testing.assert_allclose(
                getattr(rt.atom_array, field_name),
                getattr(original.atom_array, field_name),
                atol=atol,
                err_msg=f"{pdb_name}: {field_name} drifted",
            )

        # 6. insertion_code and altloc — both preserved when we read
        #    with altloc='all'.
        for field_name in ("insertion_code", "altloc"):
            orig = [str(x).strip() for x in getattr(original.atom_array, field_name)]
            after = [str(x).strip() for x in getattr(rt.atom_array, field_name)]
            assert orig == after, f"{pdb_name}: {field_name} differs"

        # 7. record_type round-trips (ATOM / HETATM).
        orig_rt = [str(x).strip() for x in original.atom_array.record_type]
        rt_rt = [str(x).strip() for x in rt.atom_array.record_type]
        assert orig_rt == rt_rt, f"{pdb_name}: record_type differs"

        # 8. entity_type: derived by the reader, not transported. Just
        #    assert it ends up equal — the reader's residue-name
        #    classifier should produce the same labels.
        orig_et = [str(x).strip() for x in original.atom_array.entity_type]
        rt_et = [str(x).strip() for x in rt.atom_array.entity_type]
        assert orig_et == rt_et, f"{pdb_name}: entity_type differs"

        # 9. Metadata fields we promise to preserve. Empty strings on
        #    the input are equivalent to missing on the round-trip; we
        #    only check that non-empty values survive.
        from molforge.core import metadata_keys as mk

        for key in (
            mk.PDB_ID,
            mk.TITLE,
            mk.CLASSIFICATION,
            mk.DEPOSITION_DATE,
            mk.EXPERIMENTAL_METHOD,
            mk.RESOLUTION,
        ):
            orig_val = original.metadata.get(key)
            if not orig_val:  # missing or empty
                continue
            rt_val = rt.metadata.get(key)
            if key == mk.RESOLUTION:
                assert float(rt_val) == pytest.approx(float(orig_val), abs=0.01), (
                    f"{pdb_name}: metadata[{key}] drifted"
                )
            else:
                assert rt_val == orig_val, f"{pdb_name}: metadata[{key}] {orig_val!r} -> {rt_val!r}"


class TestCifToCifRoundTrip:
    """Read a CIF fixture, write it back, read again — coordinates and
    structural fields must be stable across the cycle."""

    def test_dipeptide_cif_to_cif(self) -> None:
        original = read_cif(FIXTURES / "dipeptide.cif")
        rt = read_cif_string(write_cif_string(original))
        assert rt.atom_array.n_atoms == original.atom_array.n_atoms
        np.testing.assert_allclose(rt.atom_array.coords, original.atom_array.coords, atol=1e-3)
        assert list(rt.atom_array.atom_name) == list(original.atom_array.atom_name)
