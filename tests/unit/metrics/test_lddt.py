"""Tests for lDDT."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from molforge.io import read_pdb
from molforge.metrics import lddt, lddt_per_residue

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "pdb"


class TestIdenticalStructures:
    def test_global_lddt_is_one(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        assert lddt(p, p) == pytest.approx(1.0)

    def test_per_residue_all_ones(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        per_res = lddt_per_residue(p, p)
        # Drop NaN entries before checking
        valid = per_res[~np.isnan(per_res)]
        np.testing.assert_allclose(valid, 1.0)


class TestNoiseDegrades:
    def test_small_noise(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        q = deepcopy(p)
        rng = np.random.default_rng(42)
        q.atom_array.coords += rng.normal(scale=0.3, size=q.atom_array.coords.shape).astype(
            np.float32
        )
        score = lddt(q, p)
        # 0.3 Å noise should still give high lDDT
        assert 0.5 < score < 1.0

    def test_heavy_noise_low(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        q = deepcopy(p)
        rng = np.random.default_rng(42)
        q.atom_array.coords += rng.normal(scale=10.0, size=q.atom_array.coords.shape).astype(
            np.float32
        )
        # Heavy noise destroys local distances
        assert lddt(q, p) < 0.5


class TestAlignmentFree:
    """lDDT's key property: it doesn't need superposition."""

    def test_translation_doesnt_matter(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        q = deepcopy(p)
        q.atom_array.coords += np.array([100, 50, -30], dtype=np.float32)
        # lDDT shouldn't care — it's based on inter-atom distances
        assert lddt(q, p) == pytest.approx(1.0, abs=1e-5)

    def test_rotation_doesnt_matter(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        q = deepcopy(p)
        theta = np.radians(45)
        rot = np.array(
            [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]],
            dtype=np.float32,
        )
        q.atom_array.coords = (rot @ p.atom_array.coords.T).T
        assert lddt(q, p) == pytest.approx(1.0, abs=1e-5)


class TestParameters:
    def test_custom_inclusion_radius(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        score_default = lddt(p, p, inclusion_radius=15.0)
        score_small = lddt(p, p, inclusion_radius=5.0)
        # Identical structures should both score 1 regardless of radius
        assert score_default == pytest.approx(1.0)
        assert score_small == pytest.approx(1.0)

    def test_custom_thresholds(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        score = lddt(p, p, thresholds=(0.1, 1.0))
        assert score == pytest.approx(1.0)


class TestErrors:
    def test_mismatched_lengths_raises(self) -> None:
        from copy import deepcopy as _deepcopy

        p = read_pdb(FIXTURES / "helix.pdb")
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
            lddt(short, p)
