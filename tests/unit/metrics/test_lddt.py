"""Tests for lDDT."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from molforge.core import AtomArray, Protein
from molforge.io import read_pdb
from molforge.metrics import lddt, lddt_per_residue

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "pdb"


def _ca_coords_independent(protein: Protein) -> np.ndarray:
    """CA coordinates, extracted independently of the production code path."""
    arr = protein.atom_array
    idx = np.where((arr.atom_name == "CA") & (arr.entity_type == "protein"))[0]
    return arr.coords[idx].astype(np.float64)


def _brute_force_lddt_per_residue(
    model: Protein,
    reference: Protein,
    *,
    inclusion_radius: float = 15.0,
    thresholds: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
) -> list[float | None]:
    """A dead-simple, obviously-correct lDDT straight from Mariani et al. 2013.

    Explicit nested loops (no NumPy broadcasting) so it shares no code shape
    with the production implementation — an independent oracle. ``None`` for
    residues with no reference neighbours inside the radius.
    """
    m = _ca_coords_independent(model)
    r = _ca_coords_independent(reference)
    n = len(r)
    out: list[float | None] = []
    for i in range(n):
        fracs: list[float] = []
        for j in range(n):
            if i == j:
                continue
            d_ref = float(np.linalg.norm(r[i] - r[j]))
            if d_ref <= 0.0 or d_ref >= inclusion_radius:
                continue
            d_mod = float(np.linalg.norm(m[i] - m[j]))
            delta = abs(d_ref - d_mod)
            fracs.append(sum(1 for t in thresholds if delta < t) / len(thresholds))
        out.append(sum(fracs) / len(fracs) if fracs else None)
    return out


def _brute_force_lddt(model: Protein, reference: Protein) -> float:
    """Global lDDT = mean of per-residue values, dropping empty residues."""
    per_res = [v for v in _brute_force_lddt_per_residue(model, reference) if v is not None]
    return sum(per_res) / len(per_res) if per_res else 0.0


def _ca_protein(coords: np.ndarray) -> Protein:
    """Build a CA-only protein from an ``(n, 3)`` coordinate array."""
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


class TestReferenceValue:
    """Verify lDDT against independent references — not just self-consistency.

    OpenStructure (the canonical lDDT implementation) is not pip-installable,
    so the oracle here is a from-scratch brute-force implementation
    (:func:`_brute_force_lddt`, explicit loops straight from Mariani et al.
    2013) plus one hand-computed exact case. Together they anchor the
    optimized production code to the literature definition.
    """

    def test_hand_computed_exact_case(self) -> None:
        # Three collinear CA atoms at x = 0, 5, 10 (all pairs within 15 Å).
        # The model moves the third atom to x = 10.6, so only the distances
        # to it change by 0.6 Å:
        #   pair (0,1): |5.0-5.0| = 0.0 -> passes 4/4 thresholds  -> 1.00
        #   pair (0,2): |10.0-10.6| = 0.6 -> passes {1,2,4}       -> 0.75
        #   pair (1,2): |5.0-5.6|  = 0.6 -> passes {1,2,4}        -> 0.75
        # Per residue: r0=(1.00+0.75)/2, r1=(1.00+0.75)/2, r2=(0.75+0.75)/2
        #   -> 0.875, 0.875, 0.750; global mean = 0.83333…
        ref = _ca_protein(np.array([[0.0, 0, 0], [5.0, 0, 0], [10.0, 0, 0]]))
        model = _ca_protein(np.array([[0.0, 0, 0], [5.0, 0, 0], [10.6, 0, 0]]))
        assert lddt(model, ref) == pytest.approx(0.833333, abs=1e-5)
        per_res = lddt_per_residue(model, ref)
        np.testing.assert_allclose(per_res, [0.875, 0.875, 0.750], atol=1e-6)

    def test_global_matches_brute_force_on_ubiquitin(self) -> None:
        ref = read_pdb(FIXTURES / "real_ubiquitin.pdb")
        model = deepcopy(ref)
        rng = np.random.default_rng(0)
        noise = rng.normal(0.0, 0.8, model.atom_array.coords.shape).astype(np.float32)
        model.atom_array.coords[:] = model.atom_array.coords + noise
        expected = _brute_force_lddt(model, ref)
        # A non-trivial score (real noise), matched by an independent impl.
        assert 0.3 < expected < 0.95
        assert lddt(model, ref) == pytest.approx(expected, abs=1e-4)

    def test_per_residue_matches_brute_force_on_ubiquitin(self) -> None:
        ref = read_pdb(FIXTURES / "real_ubiquitin.pdb")
        model = deepcopy(ref)
        rng = np.random.default_rng(1)
        noise = rng.normal(0.0, 0.6, model.atom_array.coords.shape).astype(np.float32)
        model.atom_array.coords[:] = model.atom_array.coords + noise
        expected = _brute_force_lddt_per_residue(model, ref)
        got = lddt_per_residue(model, ref)
        for i, (g, e) in enumerate(zip(got.tolist(), expected, strict=True)):
            if e is None:
                assert np.isnan(g), f"residue {i}: production {g}, oracle None"
            else:
                assert g == pytest.approx(e, abs=1e-4), f"residue {i}"


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
