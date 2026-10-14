"""Tests for multiple sequence alignments.

The a3m insertion convention is the subtle part: lower-case letters are
insertions relative to the query and occupy no alignment column. Parse that
wrong and every row looks a different width, so most of what's checked here
is that the convention survives contact with the code.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molforge.sequence import (
    AlignedSequence,
    MultipleSequenceAlignment,
    read_a3m,
    write_a3m,
)

A3M = (
    ">query OX=9606\n"
    "MKQHKA\n"
    ">sp|P1|HIT1 OX=9606\n"
    "MKAHKA\n"
    ">sp|P2|HIT2 TaxID=10090\n"
    "MK-HKAlll\n"
    ">unlabelled\n"
    "MKRHKA\n"
)


def _msa(**kwargs: object) -> MultipleSequenceAlignment:
    return MultipleSequenceAlignment.from_a3m(A3M, **kwargs)  # type: ignore[arg-type]


class TestParsing:
    def test_query_is_the_first_record(self) -> None:
        msa = _msa()
        assert msa.query.identifier.startswith("query")
        assert msa.query.source_database == "query"
        assert msa.query_sequence == "MKQHKA"

    def test_depth_counts_the_query(self) -> None:
        assert _msa().depth == 4
        assert len(_msa()) == 4

    def test_iteration_yields_query_first(self) -> None:
        rows = list(_msa())
        assert rows[0].source_database == "query"
        assert len(rows) == 4

    def test_insertions_survive_verbatim(self) -> None:
        """Lower-case must not be normalised away — that corruption is
        exactly what a3m exists to avoid."""
        assert _msa().hits[1].sequence == "MK-HKAlll"

    def test_insertions_are_not_alignment_columns(self) -> None:
        hit = _msa().hits[1]
        assert hit.columns == "MK-HKA"
        assert len(hit.columns) == 6

    def test_blank_lines_and_wrapping_are_tolerated(self) -> None:
        wrapped = ">query\nMKQ\nHKA\n\n>hit\nMKA\nHKA\n"
        msa = MultipleSequenceAlignment.from_a3m(wrapped)
        assert msa.query_sequence == "MKQHKA"
        assert msa.hits[0].sequence == "MKAHKA"

    def test_missing_trailing_newline_is_fine(self) -> None:
        assert MultipleSequenceAlignment.from_a3m(">q\nMKQ\n>h\nMKA").depth == 2

    def test_empty_text_rejected(self) -> None:
        for text in ("", "\n\n", "# just a comment\n"):
            with pytest.raises(ValueError, match="no sequences"):
                MultipleSequenceAlignment.from_a3m(text)

    def test_source_database_label_applies_to_hits_only(self) -> None:
        msa = _msa(source_database="uniref90")
        assert msa.query.source_database == "query"
        assert {h.source_database for h in msa.hits} == {"uniref90"}


class TestPairingKeys:
    @pytest.mark.parametrize(
        ("header", "expected"),
        [
            ("sp|P1|X OX=9606", "9606"),
            ("hit TaxID=10090", "10090"),
            ("hit taxid=559292", "559292"),
            ("hit OX:4932", "4932"),
            ("no organism here", ""),
            ("", ""),
        ],
    )
    def test_extracted_from_header(self, header: str, expected: str) -> None:
        msa = MultipleSequenceAlignment.from_a3m(f">q\nMKQ\n>{header}\nMKA\n")
        assert msa.hits[0].pairing_key == expected

    def test_pairable_selects_only_keyed_hits(self) -> None:
        """An unpairable hit is still a good monomer hit — it just can't
        take part in a complex."""
        msa = _msa()
        assert len(msa.hits) == 3
        assert len(msa.pairable()) == 2


class TestValidation:
    def test_mismatched_column_count_rejected(self) -> None:
        with pytest.raises(ValueError, match="alignment columns but the query has"):
            MultipleSequenceAlignment.from_a3m(">q\nMKQHKA\n>short\nMKQ\n")

    def test_error_names_the_offending_row(self) -> None:
        with pytest.raises(ValueError, match="badhit"):
            MultipleSequenceAlignment.from_a3m(">q\nMKQHKA\n>badhit\nMKQ\n")

    def test_error_explains_the_insertion_rule(self) -> None:
        with pytest.raises(ValueError, match="lower-case letters are insertions"):
            MultipleSequenceAlignment.from_a3m(">q\nMKQHKA\n>h\nMKQ\n")

    def test_insertions_do_not_trip_the_width_check(self) -> None:
        """A hit with insertions is longer in characters and identical in
        columns — the whole point of the format."""
        msa = MultipleSequenceAlignment.from_a3m(">q\nMKQ\n>h\nMKQaaa\n")
        assert msa.n_columns == 3

    def test_empty_query_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty query"):
            MultipleSequenceAlignment(query=AlignedSequence(sequence=""))

    def test_is_frozen(self) -> None:
        import dataclasses

        with pytest.raises(dataclasses.FrozenInstanceError):
            _msa().hits = ()  # type: ignore[misc]


class TestSubsample:
    def test_query_is_always_kept(self) -> None:
        assert _msa().subsample(1).depth == 1
        assert _msa().subsample(1).query_sequence == "MKQHKA"

    def test_depth_counts_the_query(self) -> None:
        assert _msa().subsample(3).depth == 3
        assert len(_msa().subsample(3).hits) == 2

    def test_deeper_than_available_returns_everything(self) -> None:
        msa = _msa()
        assert msa.subsample(99) is msa

    def test_unseeded_truncates_deterministically(self) -> None:
        first = _msa().subsample(2)
        second = _msa().subsample(2)
        assert first.hits[0].identifier == second.hits[0].identifier
        assert first.hits[0] == _msa().hits[0]

    def test_seeded_sampling_is_reproducible(self) -> None:
        a = _msa().subsample(3, seed=7)
        b = _msa().subsample(3, seed=7)
        assert [h.identifier for h in a.hits] == [h.identifier for h in b.hits]

    def test_different_seeds_can_differ(self) -> None:
        """Sampling rather than truncating is the point: the top of a search
        output is its most redundant part."""
        picks = {tuple(h.identifier for h in _msa().subsample(2, seed=s).hits) for s in range(12)}
        assert len(picks) > 1

    def test_sampling_preserves_the_width_invariant(self) -> None:
        sub = _msa().subsample(2, seed=3)
        assert sub.n_columns == 6

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_depth_rejected(self, bad: int) -> None:
        with pytest.raises(ValueError, match="depth must be >= 1"):
            _msa().subsample(bad)

    def test_original_is_untouched(self) -> None:
        msa = _msa()
        msa.subsample(2, seed=1)
        assert msa.depth == 4


class TestDeduplicate:
    def test_drops_rows_repeating_the_query(self) -> None:
        msa = MultipleSequenceAlignment.from_a3m(">q\nMKQ\n>same\nMKQ\n>other\nMKA\n")
        assert msa.deduplicate().depth == 2

    def test_drops_repeated_hits(self) -> None:
        msa = MultipleSequenceAlignment.from_a3m(">q\nMKQ\n>a\nMKA\n>b\nMKA\n")
        assert msa.deduplicate().depth == 2

    def test_compares_columns_not_raw_text(self) -> None:
        """Two rows differing only in insertions occupy the same columns."""
        msa = MultipleSequenceAlignment.from_a3m(">q\nMKQ\n>a\nMKAaa\n>b\nMKAbb\n")
        assert msa.deduplicate().depth == 2

    def test_keeps_the_first_occurrence(self) -> None:
        msa = MultipleSequenceAlignment.from_a3m(">q\nMKQ\n>first\nMKA\n>second\nMKA\n")
        assert msa.deduplicate().hits[0].identifier == "first"


class TestSerialisation:
    def test_a3m_round_trip(self) -> None:
        msa = _msa()
        assert MultipleSequenceAlignment.from_a3m(msa.to_a3m()).to_a3m() == msa.to_a3m()

    def test_round_trip_preserves_insertions(self) -> None:
        msa = MultipleSequenceAlignment.from_a3m(_msa().to_a3m())
        assert msa.hits[1].sequence == "MK-HKAlll"

    def test_query_is_written_first(self) -> None:
        assert _msa().to_a3m().splitlines()[0].startswith(">query")

    def test_pairing_rows_match_chai_columns(self) -> None:
        rows = _msa(source_database="uniref90").to_pairing_rows()
        assert set(rows[0]) == {
            "sequence",
            "identifier",
            "source_database",
            "pairing_key",
            "comment",
        }
        assert rows[0]["source_database"] == "query"
        assert len(rows) == 4

    def test_pairing_rows_are_json_native(self) -> None:
        import json

        rows = _msa().to_pairing_rows()
        assert json.loads(json.dumps(rows)) == rows

    def test_file_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "hits.a3m"
        write_a3m(_msa(), path)
        assert read_a3m(path).to_a3m() == _msa().to_a3m()

    def test_read_applies_the_source_label(self, tmp_path: Path) -> None:
        path = tmp_path / "hits.a3m"
        write_a3m(_msa(), path)
        assert {h.source_database for h in read_a3m(path, source_database="mgnify").hits} == {
            "mgnify"
        }


class TestWithSourceDatabase:
    def test_labels_every_hit(self) -> None:
        labelled = _msa().with_source_database("bfd_uniclust")
        assert {h.source_database for h in labelled.hits} == {"bfd_uniclust"}

    def test_leaves_the_query_alone(self) -> None:
        assert _msa().with_source_database("mgnify").query.source_database == "query"


class TestAlignedSequence:
    def test_ungapped_strips_gaps_and_casing(self) -> None:
        assert AlignedSequence(sequence="MK-Qaa.R").ungapped == "MKQAAR"

    def test_columns_excludes_insertions_and_dots(self) -> None:
        assert AlignedSequence(sequence="MK-Qaa.R").columns == "MK-QR"

    def test_defaults_are_empty_not_guessed(self) -> None:
        row = AlignedSequence(sequence="MKQ")
        assert row.identifier == ""
        assert row.pairing_key == ""
        assert row.source_database == ""
