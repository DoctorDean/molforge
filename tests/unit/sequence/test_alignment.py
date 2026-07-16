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


class TestSmithWatermanAffineTraceback:
    """Regression tests for the affine-gap local traceback.

    A single traceback-pointer matrix cannot distinguish gap-open from
    gap-extend, so it can reconstruct a suboptimal alignment even though
    the reported score is the true optimum. The invariant below — the
    returned alignment must score exactly what the function reports —
    catches that: it fails for the single-pointer implementation on the
    ``CAAAACCACAACCCCC`` case at the default ``gap_open=-10``.
    """

    @staticmethod
    def _affine_score(
        aligned_a: str,
        aligned_b: str,
        match: int,
        mismatch: int,
        gap_open: int,
        gap_extend: int,
    ) -> int:
        total = 0
        in_gap = False
        for x, y in zip(aligned_a, aligned_b, strict=True):
            if x == "-" or y == "-":
                total += gap_extend if in_gap else gap_open
                in_gap = True
            else:
                total += match if x == y else mismatch
                in_gap = False
        return total

    def test_multi_residue_gap_run_reconstructed(self) -> None:
        # The two MKTV blocks must align, forcing a 3-residue gap in b.
        result = smith_waterman(
            "MKTVANDMKTV",
            "MKTVMKTV",
            matrix=None,
            match=3,
            mismatch=-2,
            gap_open=-4,
            gap_extend=-1,
        )
        assert result.aligned_a == "MKTVANDMKTV"
        assert result.aligned_b == "MKTV---MKTV"
        assert result.score == 18  # 8*3 matches - (4 + 1 + 1) for the 3-gap
        assert result.identity == 1.0

    @pytest.mark.parametrize(
        "a, b, match, mismatch, gap_open, gap_extend",
        [
            ("MKTVANDMKTV", "MKTVMKTV", 3, -2, -4, -1),
            # Discovered by randomized search; the single-pointer traceback
            # reported 27 but returned an alignment scoring only 22.
            ("CAAAACCACAACCCCC", "AACAAACAAC", 5, -4, -10, -1),
            ("WWWACDEFGHIKLWWW", "ACDEFGHIKL", 2, -1, -10, -1),
        ],
    )
    def test_returned_alignment_matches_reported_score(
        self,
        a: str,
        b: str,
        match: int,
        mismatch: int,
        gap_open: int,
        gap_extend: int,
    ) -> None:
        result = smith_waterman(
            a,
            b,
            matrix=None,
            match=match,
            mismatch=mismatch,
            gap_open=gap_open,
            gap_extend=gap_extend,
        )
        recomputed = self._affine_score(
            result.aligned_a,
            result.aligned_b,
            match,
            mismatch,
            gap_open,
            gap_extend,
        )
        assert recomputed == result.score


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


# BLOSUM62, gap_open=-10, gap_extend=-1 (the aligner defaults) — golden
# scores computed offline with Biopython's Bio.Align.PairwiseAligner
# (global for NW, local for SW). molforge reproduces every one exactly.
# Biopython is NOT a test-time dependency; the goldens are hard-coded, per
# the TM-align / DSSP reference-value precedent.
#
#   from Bio.Align import PairwiseAligner, substitution_matrices
#   al = PairwiseAligner(); al.mode = "global" | "local"
#   al.substitution_matrix = substitution_matrices.load("BLOSUM62")
#   al.open_gap_score = -10; al.extend_gap_score = -1
#   al.score(a, b)
_GLOBAL_GOLDENS = [
    ("HEAGAWGHEE", "PAWHEAE", 3.0),  # Durbin et al. textbook pair
    ("MKTAYIAKQR", "MKTAYIAKQR", 49.0),
    ("ACDEFGHIKLMNPQRSTVWY", "ACDEFGHILMNPQRSTVWY", 101.0),  # one deletion
    ("WWWWACDEFGHIKLMNWWWW", "ACDEFGHIKLMN", 42.0),
]
_LOCAL_GOLDENS = [
    ("HEAGAWGHEE", "PAWHEAE", 18.0),  # Durbin et al. textbook pair
    ("MKTAYIAKQR", "MKTAYIAKQR", 49.0),
    ("ACDEFGHIKLMNPQRSTVWY", "ACDEFGHILMNPQRSTVWY", 101.0),
    ("WWWWACDEFGHIKLMNWWWW", "ACDEFGHIKLMN", 68.0),  # local island
]


class TestReferenceValue:
    """Golden alignment scores against Biopython's PairwiseAligner."""

    @pytest.mark.parametrize(("a", "b", "expected"), _GLOBAL_GOLDENS)
    def test_needleman_wunsch_matches_biopython(self, a: str, b: str, expected: float) -> None:
        assert needleman_wunsch(a, b).score == pytest.approx(expected)

    @pytest.mark.parametrize(("a", "b", "expected"), _LOCAL_GOLDENS)
    def test_smith_waterman_matches_biopython(self, a: str, b: str, expected: float) -> None:
        assert smith_waterman(a, b).score == pytest.approx(expected)

    def test_smith_waterman_traceback_extracts_local_region(self) -> None:
        # Guards the affine-gap traceback fix: the core must be located
        # inside the W-flanked sequence, not mis-bounded.
        al = smith_waterman("WWWWACDEFGHIKLMNWWWW", "ACDEFGHIKLMN")
        assert al.aligned_a == "ACDEFGHIKLMN"
        assert al.aligned_b == "ACDEFGHIKLMN"
        assert al.identity == pytest.approx(1.0)
        assert (al.start_a, al.end_a) == (4, 16)
        assert (al.start_b, al.end_b) == (0, 12)

    def test_smith_waterman_textbook_alignment(self) -> None:
        # The classic HEAGAWGHEE / PAWHEAE local alignment (Durbin et al.).
        al = smith_waterman("HEAGAWGHEE", "PAWHEAE")
        assert al.aligned_a == "AWGHE"
        assert al.aligned_b == "AW-HE"
        assert al.score == pytest.approx(18.0)

    def test_needleman_wunsch_places_the_gap(self) -> None:
        al = needleman_wunsch("ACDEFGHIKLMNPQRSTVWY", "ACDEFGHILMNPQRSTVWY")
        assert al.aligned_a == "ACDEFGHIKLMNPQRSTVWY"
        assert al.aligned_b == "ACDEFGHI-LMNPQRSTVWY"
        assert al.identity == pytest.approx(1.0)
