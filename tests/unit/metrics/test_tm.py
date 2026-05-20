"""Tests for TM-score."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from molforge.core import AtomArray, Protein
from molforge.io import read_pdb
from molforge.metrics import tm_score
from molforge.metrics.tm import _d0

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "pdb"


class TestD0:
    def test_long_protein(self) -> None:
        d0 = _d0(100)
        # Formula: 1.24 * (100 - 15)^(1/3) - 1.8 = 1.24 * 4.397 - 1.8 ≈ 3.65
        assert 3.5 < d0 < 3.7

    def test_short_protein(self) -> None:
        assert _d0(15) == pytest.approx(0.5)
        assert _d0(10) == pytest.approx(0.5)

    def test_boundary(self) -> None:
        assert _d0(21) == pytest.approx(1.24 * 6 ** (1.0 / 3.0) - 1.8, abs=1e-6)


class TestIdenticalStructures:
    def test_perfect_match_is_one(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        score = tm_score(p, p)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_translation_doesnt_matter(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        q = deepcopy(p)
        q.atom_array.coords += np.array([100, 50, -30], dtype=np.float32)
        # TM-score works after optimal superposition, so translation alone -> 1.0
        assert tm_score(q, p) == pytest.approx(1.0, abs=1e-5)

    def test_rotation_doesnt_matter(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        q = deepcopy(p)
        theta = np.radians(45)
        rot = np.array(
            [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]],
            dtype=np.float32,
        )
        q.atom_array.coords = (rot @ p.atom_array.coords.T).T
        assert tm_score(q, p) == pytest.approx(1.0, abs=1e-5)


class TestNoiseDegrades:
    def test_small_noise_high_score(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        q = deepcopy(p)
        rng = np.random.default_rng(42)
        q.atom_array.coords += rng.normal(scale=0.5, size=q.atom_array.coords.shape).astype(
            np.float32
        )
        score = tm_score(q, p)
        # For this 15-residue fixture, d0 is at its floor of 0.5 A, which
        # makes the score very sensitive to noise. Still, ~0.4 is a clear
        # signal vs. the < 0.2 we'd get from random structures.
        assert 0.2 < score < 1.0

    def test_large_noise_low_score(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        q = deepcopy(p)
        rng = np.random.default_rng(42)
        q.atom_array.coords += rng.normal(scale=10.0, size=q.atom_array.coords.shape).astype(
            np.float32
        )
        score = tm_score(q, p)
        # Heavy noise -> low score
        assert score < 0.5


class TestNormalization:
    def test_reference_normalization_default(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        score = tm_score(p, p, normalize_by="reference")
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_model_normalization(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        score = tm_score(p, p, normalize_by="model")
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_unknown_normalization_raises(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        with pytest.raises(ValueError, match="unknown normalize_by"):
            tm_score(p, p, normalize_by="bogus")


class TestErrors:
    def test_mismatched_lengths_raises(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        # Build a shorter protein
        from copy import deepcopy as _deepcopy

        short = _deepcopy(p)
        arr = short.atom_array
        arr.coords = arr.coords[:20]
        for f in (
            "element",
            "atom_name",
            "residue_name",
            "residue_id",
            "insertion_code",
            "chain_id",
            "b_factor",
            "occupancy",
            "charge",
            "serial",
            "record_type",
            "entity_type",
            "altloc",
            "model_id",
        ):
            setattr(arr, f, getattr(arr, f)[:20])
        arr._invalidate_cache()
        with pytest.raises(ValueError, match="matched residue lists"):
            tm_score(short, p)

    def test_too_small_raises(self) -> None:
        empty = Protein(AtomArray(0))
        with pytest.raises(ValueError, match="at least 3"):
            tm_score(empty, empty)
