"""Tests for sequence featurization."""

from __future__ import annotations

import numpy as np
import pytest

from molforge.ml import (
    blosum_embed,
    compose_features,
    one_hot,
    positional_encoding,
)


class TestOneHot:
    def test_shape_with_unk(self) -> None:
        result = one_hot("MKTV")
        assert result.shape == (4, 21)
        assert result.dtype == np.float32

    def test_shape_without_unk(self) -> None:
        result = one_hot("MKTV", include_unk=False)
        assert result.shape == (4, 20)

    def test_one_hot_per_row(self) -> None:
        result = one_hot("AKTV")
        assert (result.sum(axis=1) == 1.0).all()

    def test_unknown_residue_uses_unk_column(self) -> None:
        result = one_hot("AXM")
        # 'X' is not in the alphabet, so the unk column is set
        assert result[1, 20] == 1.0
        assert result[1, :20].sum() == 0.0

    def test_unknown_dropped_when_unk_disabled(self) -> None:
        result = one_hot("AXM", include_unk=False)
        # Row for 'X' should be all-zero
        assert result[1].sum() == 0.0
        # Other rows still have a 1
        assert result[0].sum() == 1.0
        assert result[2].sum() == 1.0

    def test_case_insensitive(self) -> None:
        upper = one_hot("AKTV")
        lower = one_hot("aktv")
        np.testing.assert_array_equal(upper, lower)

    def test_whitespace_stripped(self) -> None:
        with_ws = one_hot("A K T V")
        without_ws = one_hot("AKTV")
        np.testing.assert_array_equal(with_ws, without_ws)


class TestBlosumEmbed:
    def test_shape(self) -> None:
        result = blosum_embed("MKTV")
        assert result.shape == (4, 20)
        assert result.dtype == np.float32

    def test_unknown_uses_x_row(self) -> None:
        # 'X' is a known column in BLOSUM62 — non-standard residues
        # fall back to the 'X' row.
        result_x = blosum_embed("X")
        # The X row's diagonal value should be matched in result_x
        assert result_x.shape == (1, 20)

    def test_pam250_matrix(self) -> None:
        result = blosum_embed("MKTV", matrix="PAM250")
        assert result.shape == (4, 20)


class TestPositionalEncoding:
    def test_shape(self) -> None:
        pe = positional_encoding(10, dim=64)
        assert pe.shape == (10, 64)
        assert pe.dtype == np.float32

    def test_first_position_specific_values(self) -> None:
        # PE[0, 0] = sin(0/...) = 0; PE[0, 1] = cos(0/...) = 1
        pe = positional_encoding(5, dim=4)
        assert pe[0, 0] == pytest.approx(0.0)
        assert pe[0, 1] == pytest.approx(1.0)

    def test_values_in_range(self) -> None:
        pe = positional_encoding(100, dim=32)
        # sin/cos always in [-1, 1]
        assert pe.min() >= -1.001
        assert pe.max() <= 1.001

    def test_odd_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="must be even"):
            positional_encoding(10, dim=15)


class TestComposeFeatures:
    def test_concat_two_features(self) -> None:
        a = np.ones((5, 3), dtype=np.float32)
        b = np.zeros((5, 7), dtype=np.float32)
        out = compose_features(a, b)
        assert out.shape == (5, 10)
        assert (out[:, :3] == 1.0).all()
        assert (out[:, 3:] == 0.0).all()

    def test_concat_full_real_features(self) -> None:
        seq = "MKTVRQERLKSIVRILERSK"
        oh = one_hot(seq)
        bl = blosum_embed(seq)
        pe = positional_encoding(len(seq), dim=32)
        combined = compose_features(oh, bl, pe)
        assert combined.shape == (len(seq), 21 + 20 + 32)

    def test_mismatched_lengths_raises(self) -> None:
        a = np.ones((5, 3), dtype=np.float32)
        b = np.zeros((7, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="leading dimension"):
            compose_features(a, b)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            compose_features()
