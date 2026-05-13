"""Tests for the FASTA reader and writer."""

from __future__ import annotations

from pathlib import Path

from molforge.io import (
    FastaRecord,
    read_fasta,
    read_fasta_string,
    write_fasta,
    write_fasta_string,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "fasta"


class TestReadBasic:
    def test_read_simple_file(self) -> None:
        records = read_fasta(FIXTURES / "simple.fasta")
        assert len(records) == 2

    def test_record_ids(self) -> None:
        records = read_fasta(FIXTURES / "simple.fasta")
        assert records[0].id == "sp|P01308|INS_HUMAN"
        assert records[1].id == "sp|P01009|A1AT_HUMAN"

    def test_descriptions(self) -> None:
        records = read_fasta(FIXTURES / "simple.fasta")
        assert "Insulin" in records[0].description
        assert "Alpha-1-antitrypsin" in records[1].description

    def test_sequence_reassembly(self) -> None:
        records = read_fasta(FIXTURES / "simple.fasta")
        # First record's sequence is the two lines joined, no whitespace
        assert records[0].sequence.startswith("MALWMRLLPL")
        assert records[0].sequence.endswith("ENYCN")
        assert "\n" not in records[0].sequence
        assert " " not in records[0].sequence


class TestReadEdgeCases:
    def test_strips_whitespace_and_digits(self) -> None:
        records = read_fasta(FIXTURES / "multiline_with_digits.fasta")
        assert len(records) == 1
        seq = records[0].sequence
        # No digits, no spaces
        assert all(c.isalpha() for c in seq)
        assert seq.startswith("MKTV")

    def test_blank_lines_tolerated(self) -> None:
        text = ">a\nACGT\n\n\n>b\nGGGG\n"
        records = list(read_fasta_string(text))
        assert len(records) == 2
        assert records[0].sequence == "ACGT"
        assert records[1].sequence == "GGGG"

    def test_comment_lines_skipped(self) -> None:
        text = ";this is a legacy comment\n>a\nACGT\n"
        records = list(read_fasta_string(text))
        assert len(records) == 1
        assert records[0].id == "a"

    def test_empty_input(self) -> None:
        assert list(read_fasta_string("")) == []

    def test_header_only(self) -> None:
        records = list(read_fasta_string(">a\n"))
        assert len(records) == 1
        assert records[0].sequence == ""

    def test_header_with_no_id(self) -> None:
        records = list(read_fasta_string(">\nACGT\n"))
        assert len(records) == 1
        assert records[0].id == ""

    def test_sequence_before_header_skipped(self) -> None:
        # Malformed but should not crash
        text = "ACGT\n>a\nGGGG\n"
        records = list(read_fasta_string(text))
        assert len(records) == 1
        assert records[0].sequence == "GGGG"


class TestWrite:
    def test_round_trip(self, tmp_path: Path) -> None:
        original = read_fasta(FIXTURES / "simple.fasta")
        out = tmp_path / "out.fasta"
        write_fasta(original, out)
        reloaded = read_fasta(out)
        assert len(reloaded) == len(original)
        for o, r in zip(original, reloaded, strict=True):
            assert o.id == r.id
            assert o.sequence == r.sequence

    def test_write_tuples(self, tmp_path: Path) -> None:
        records = [("a", "ACGT"), ("b", "GGGG")]
        out = tmp_path / "tuples.fasta"
        write_fasta(records, out)
        reloaded = read_fasta(out)
        assert reloaded[0].id == "a"
        assert reloaded[0].sequence == "ACGT"

    def test_line_width_default_is_80(self) -> None:
        rec = FastaRecord(id="long", sequence="A" * 200)
        text = write_fasta_string([rec])
        # Body lines (excluding the header) should be <= 80 chars
        for line in text.splitlines():
            if line.startswith(">"):
                continue
            assert len(line) <= 80

    def test_line_width_zero_emits_one_line(self) -> None:
        rec = FastaRecord(id="long", sequence="A" * 200)
        text = write_fasta_string([rec], line_width=0)
        body_lines = [ln for ln in text.splitlines() if not ln.startswith(">")]
        assert len(body_lines) == 1
        assert len(body_lines[0]) == 200

    def test_write_empty_list(self) -> None:
        assert write_fasta_string([]) == ""


class TestFastaRecord:
    def test_len(self) -> None:
        rec = FastaRecord(id="x", sequence="ACGT")
        assert len(rec) == 4

    def test_header_with_description(self) -> None:
        rec = FastaRecord(id="x", description="some desc", sequence="A")
        assert rec.header == "x some desc"

    def test_header_without_description(self) -> None:
        rec = FastaRecord(id="x", sequence="A")
        assert rec.header == "x"
