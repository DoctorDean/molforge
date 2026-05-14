"""Tests for substitution matrices."""

from __future__ import annotations

import pytest

from molforge.sequence import BLOSUM62, PAM250, available_matrices, get_matrix


class TestMatrices:
    def test_blosum62_shape(self) -> None:
        assert BLOSUM62.shape == (24, 24)

    def test_pam250_shape(self) -> None:
        assert PAM250.shape == (24, 24)

    def test_blosum62_symmetric(self) -> None:

        assert (BLOSUM62 == BLOSUM62.T).all()

    def test_pam250_symmetric(self) -> None:

        assert (PAM250 == PAM250.T).all()

    def test_get_matrix_known(self) -> None:
        matrix, idx = get_matrix("BLOSUM62")
        assert matrix.shape == (24, 24)
        assert idx["A"] == 0

    def test_get_matrix_case_insensitive(self) -> None:
        matrix, _ = get_matrix("blosum62")
        assert matrix.shape == (24, 24)

    def test_get_matrix_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown substitution matrix"):
            get_matrix("BOGUS9000")

    def test_available_matrices_list(self) -> None:
        names = available_matrices()
        assert "BLOSUM62" in names
        assert "PAM250" in names
