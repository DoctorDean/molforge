"""Tests for pairwise sequence alignment."""

from __future__ import annotations

import pytest

from molforge.sequence import (
    Alignment,
    align,
    identity,
    needleman_wunsch,
    smith_waterman,
)


class TestNeedlemanWunschBasic:
    def test_identical_sequences(self) -> None:
        result = needleman_wunsch("MKTV", "MKTV")
        assert result.aligned_a == "MKTV"
        assert result.aligned_b == "MKTV"
        assert result.identity == 1.0
        assert result.length == 4

    def test_aligns_with_gaps(self) -> None:
        result = needleman_wunsch("MKTV", "MKV")
        assert "-" in result.aligned_a or "-" in result.aligned_b
        assert result.length >= 4

    def test_full_coverage(self) -> None:
        result = needleman_wunsch("MKTV", "MKTV")
        assert result.coverage_a == 1.0
        assert result.coverage_b == 1.0
        assert result.start_a == 0
        assert result.end_a == 4

    def test_no_matrix_uses_match_mismatch(self) -> None:
        result = needleman_wunsch("AAAA", "AAAA", matrix=None, match=5, mismatch=-3)
        assert result.identity == 1.0
        assert result.score == 20  # 4 matches at +5

    def test_low_identity_for_random_pair(self) -> None:
        # Two unrelated sequences should align with low identity
        result = needleman_wunsch("MKTVRQERLKSIVRILER", "AAAAAAAAAAAAAAAAA")
        assert result.identity < 0.3


class TestSmithWatermanBasic:
    def test_finds_shared_local_region(self) -> None:
        # A in middle of two different flankers
        a = "XXXXMKTVXXXX"
        b = "YYYMKTVYYYY"
        result = smith_waterman(a, b)
        assert "MKTV" in result.aligned_a
        assert "MKTV" in result.aligned_b
        assert result.identity > 0.8

    def test_local_returns_subregion(self) -> None:
        a = "ZZZZAAAAZZZZ"
        b = "WWWAAAAWWW"
        result = smith_waterman(a, b, matrix=None, match=5, mismatch=-10)
        assert result.end_a - result.start_a == 4
        assert result.aligned_a == "AAAA"
        assert result.aligned_b == "AAAA"

    def test_coverage_for_local(self) -> None:
        a = "ZZZZZAAAAZZZZZ"
        b = "AAAA"
        result = smith_waterman(a, b, matrix=None, match=5, mismatch=-10)
        # b is fully covered; only ~4/14 of a is covered
        assert result.coverage_b == 1.0
        assert result.coverage_a < 0.5


class TestAlignDispatcher:
    def test_global_default(self) -> None:
        result = align("MKTV", "MKTV")
        assert isinstance(result, Alignment)
        assert result.identity == 1.0

    def test_local_mode(self) -> None:
        result = align("AAAMKTVAAA", "MKTV", mode="local")
        assert result.identity == 1.0

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown alignment mode"):
            align("A", "A", mode="bogus")


class TestIdentityHelper:
    def test_identical(self) -> None:
        assert identity("MKTV", "MKTV") == 1.0

    def test_completely_different(self) -> None:
        # With BLOSUM62 this can still produce some alignment, but identity should be low
        result_id = identity("AAAAAAAA", "MMMMMMMM")
        assert result_id < 0.2


class TestEdgeCases:
    def test_empty_sequence_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            needleman_wunsch("", "MKTV")
        with pytest.raises(ValueError, match="empty"):
            smith_waterman("MKTV", "")

    def test_lowercase_normalized(self) -> None:
        # Case-insensitivity comes from normalizing to upper
        result = needleman_wunsch("mktv", "MKTV")
        assert result.identity == 1.0

    def test_whitespace_stripped(self) -> None:
        result = needleman_wunsch("MK TV", "MKTV")
        assert result.identity == 1.0


class TestFormat:
    def test_format_produces_three_line_blocks(self) -> None:
        result = needleman_wunsch("MKTV", "MKTV")
        formatted = result.format(width=10)
        lines = formatted.split("\n")
        # 3 lines per block (top / matches / bottom)
        assert "MKTV" in lines[0]
        assert "MKTV" in lines[2]
