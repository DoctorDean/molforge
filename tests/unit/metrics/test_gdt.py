"""Tests for GDT-TS and GDT-HA."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from molforge.core import Protein
from molforge.core.atom_array import AtomArray
from molforge.io import read_pdb
from molforge.metrics import gdt_ha, gdt_per_cutoff, gdt_ts
from molforge.structure.superposition import superpose

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "pdb"


def _ca_protein(coords: np.ndarray) -> Protein:
    """Build a CA-only protein from an (n, 3) coordinate array."""
    n = coords.shape[0]
    return Protein(
        AtomArray.from_dict(
            {
                "coords": coords.astype(np.float32),
                "atom_name": np.array(["CA"] * n, dtype="U4"),
                "residue_id": np.arange(1, n + 1, dtype="int32"),
                "chain_id": np.array(["A"] * n, dtype="U4"),
                "entity_type": np.array(["protein"] * n, dtype="U8"),
            }
        )
    )


def _helix_ca(n: int) -> np.ndarray:
    """Idealized alpha-helix CA trace with n residues."""
    i = np.arange(n)
    theta = np.radians(100.0) * i
    return np.stack([2.3 * np.cos(theta), 2.3 * np.sin(theta), 1.5 * i], axis=1).astype(np.float64)


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


class TestDomainMotionMaximization:
    """GDT is a maximum over superpositions (LGA fits the most residues
    under each cutoff), not the value at the RMSD-minimizing Kabsch fit.
    A displaced sub-domain must not drag the score down.
    """

    def test_displaced_domain_is_recovered(self) -> None:
        ref = _helix_ca(40)
        model = ref.copy()
        model[28:] += np.array([40.0, 0.0, 0.0])  # rigidly displace the last 12 residues
        ref_p, model_p = _ca_protein(ref), _ca_protein(model)

        # Kabsch-fit GDT-TS (what the old implementation returned).
        sp = superpose(model, ref)
        d = np.linalg.norm(sp.mobile_aligned - ref, axis=1)
        kabsch_gdt = float(np.mean([(d < c).sum() / 40 for c in (1.0, 2.0, 4.0, 8.0)]))

        score = gdt_ts(model_p, ref_p)
        # The 28-residue core (70%) superposes exactly, so GDT-TS ~ 0.70 ...
        assert score == pytest.approx(28 / 40, abs=0.06)
        # ... far above the Kabsch value (~0.10 here).
        assert score > kabsch_gdt + 0.1


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
