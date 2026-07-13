"""Tests for TM-score."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from molforge.core import AtomArray, Protein
from molforge.io import read_pdb
from molforge.metrics import tm_score
from molforge.metrics.tm import _ca_coords, _d0
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


class TestDomainMotionMaximization:
    """TM-score is the maximum over superpositions, not the value at the
    RMSD-minimizing Kabsch fit. When part of a structure is displaced (a
    hinge / domain motion), Kabsch compromises and underestimates TM; the
    fragment-seeded search must recover the well-superposable core.
    """

    def test_displaced_domain_is_recovered(self) -> None:
        ref = _helix_ca(40)
        model = ref.copy()
        model[28:] += np.array([40.0, 0.0, 0.0])  # rigidly displace the last 12 residues
        ref_p, model_p = _ca_protein(ref), _ca_protein(model)

        # The old implementation returned the Kabsch-fit value.
        sp = superpose(model, ref)
        d = np.linalg.norm(sp.mobile_aligned - ref, axis=1)
        d0 = _d0(40)
        kabsch_tm = float((1.0 / (1.0 + (d / d0) ** 2)).sum() / 40)

        score = tm_score(model_p, ref_p)
        # The 28-residue core (70%) superposes exactly, so TM ~ 0.70 ...
        assert score == pytest.approx(28 / 40, abs=0.05)
        # ... far above the RMSD-compromised Kabsch value (~0.05 here).
        assert score > kabsch_tm + 0.1


class TestReferenceValue:
    """Golden TM-score against TM-align (tmtools), computed offline on
    ubiquitin (1UBQ) with a 40-degree hinge at residue 45. Guards the
    superposition-maximizing search from regressing to a single Kabsch
    fit, which scores ~0.45 on this case — far outside tolerance.
    """

    def test_matches_tmalign_on_hinged_ubiquitin(self) -> None:
        ref_ca = _ca_coords(read_pdb(FIXTURES / "real_ubiquitin.pdb"))
        theta = np.radians(40.0)
        rot = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        model_ca = ref_ca.copy()
        model_ca[45:] = (rot @ (ref_ca[45:] - ref_ca[45]).T).T + ref_ca[45]
        score = tm_score(_ca_protein(model_ca), _ca_protein(ref_ca))
        # TM-align (tmtools) golden = 0.696; the old Kabsch fit gave ~0.445.
        assert score == pytest.approx(0.696, abs=0.03)


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
