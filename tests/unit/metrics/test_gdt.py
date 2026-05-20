"""Tests for GDT-TS and GDT-HA."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from molforge.io import read_pdb
from molforge.metrics import gdt_ha, gdt_per_cutoff, gdt_ts

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "pdb"


class TestIdenticalStructures:
    def test_gdt_ts_is_one(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        assert gdt_ts(p, p) == pytest.approx(1.0)

    def test_gdt_ha_is_one(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        assert gdt_ha(p, p) == pytest.approx(1.0)


class TestNoiseDegrades:
    def test_small_noise(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        q = deepcopy(p)
        rng = np.random.default_rng(42)
        q.atom_array.coords += rng.normal(scale=0.3, size=q.atom_array.coords.shape).astype(
            np.float32
        )
        # 0.3 Å noise: 1 Å cutoff should pass, smaller might fail
        ts = gdt_ts(q, p)
        ha = gdt_ha(q, p)
        assert 0.5 < ts <= 1.0
        # HA is stricter, so it's <= TS for the same structures
        assert ha <= ts + 1e-6

    def test_heavy_noise_low(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        q = deepcopy(p)
        rng = np.random.default_rng(42)
        q.atom_array.coords += rng.normal(scale=8.0, size=q.atom_array.coords.shape).astype(
            np.float32
        )
        # ~8 Å noise: probably below the 8 Å cutoff but well above the others
        assert gdt_ts(q, p) < 0.7

    def test_gdt_ha_stricter_than_ts(self) -> None:
        """For any non-trivial noise, GDT-HA should be ≤ GDT-TS."""
        p = read_pdb(FIXTURES / "helix.pdb")
        q = deepcopy(p)
        rng = np.random.default_rng(42)
        q.atom_array.coords += rng.normal(scale=1.5, size=q.atom_array.coords.shape).astype(
            np.float32
        )
        assert gdt_ha(q, p) <= gdt_ts(q, p) + 1e-6


class TestPerCutoff:
    def test_returns_dict(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        result = gdt_per_cutoff(p, p)
        assert set(result.keys()) == {1.0, 2.0, 4.0, 8.0}

    def test_all_one_for_identical(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        result = gdt_per_cutoff(p, p)
        for v in result.values():
            assert v == pytest.approx(1.0)

    def test_custom_cutoffs(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        result = gdt_per_cutoff(p, p, cutoffs=(0.1, 0.5, 1.0))
        assert set(result.keys()) == {0.1, 0.5, 1.0}

    def test_monotonic(self) -> None:
        """Larger cutoffs include more residues, so fractions are
        monotonically non-decreasing."""
        p = read_pdb(FIXTURES / "helix.pdb")
        q = deepcopy(p)
        rng = np.random.default_rng(42)
        q.atom_array.coords += rng.normal(scale=2.0, size=q.atom_array.coords.shape).astype(
            np.float32
        )
        result = gdt_per_cutoff(q, p, cutoffs=(0.5, 1, 2, 4, 8))
        values = [result[c] for c in (0.5, 1, 2, 4, 8)]
        from itertools import pairwise

        for a, b in pairwise(values):
            assert a <= b + 1e-6
