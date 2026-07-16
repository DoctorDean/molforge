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


def _atoms_independent(protein: Protein, atom_set: str) -> tuple[np.ndarray, list[int]]:
    """Coords and per-atom residue ordinals, extracted independently of production.

    Returns ``(coords, atom_residue)`` where ``atom_residue[k]`` is the
    0-based residue ordinal of atom ``k``. ``atom_set="ca"`` gives one atom
    per residue; ``"heavy"`` gives every non-hydrogen atom, in residue order.
    """
    arr = protein.atom_array
    coords: list[np.ndarray] = []
    residue: list[int] = []
    r = 0
    for sl in _residue_slices(arr):
        if str(arr.entity_type[sl.start]) != "protein":
            continue
        names = arr.atom_name[sl]
        elements = arr.element[sl]
        if atom_set == "ca":
            keep = [k for k in range(len(names)) if names[k] == "CA"]
        else:  # heavy
            keep = [k for k in range(len(names)) if elements[k] != "H"]
        if not keep:
            continue
        for k in keep:
            coords.append(arr.coords[sl][k].astype(np.float64))
            residue.append(r)
        r += 1
    return np.array(coords, dtype=np.float64).reshape(-1, 3), residue


def _residue_slices(arr: object) -> list[slice]:
    return list(arr.iter_residue_slices())  # type: ignore[attr-defined]


def _brute_force_lddt_per_residue(
    model: Protein,
    reference: Protein,
    *,
    atom_set: str = "ca",
    inclusion_radius: float = 15.0,
    thresholds: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
) -> list[float | None]:
    """A dead-simple, obviously-correct lDDT straight from Mariani et al. 2013.

    Explicit nested loops (no NumPy broadcasting) so it shares no code shape
    with the production implementation — an independent oracle. Excludes
    intra-residue pairs (a no-op for ``"ca"``). ``None`` for residues with no
    reference neighbours inside the radius.
    """
    m, _ = _atoms_independent(model, atom_set)
    r, res = _atoms_independent(reference, atom_set)
    n_res = (max(res) + 1) if res else 0
    per_res_fracs: list[list[float]] = [[] for _ in range(n_res)]
    for i in range(len(r)):
        for j in range(len(r)):
            if i == j or res[i] == res[j]:
                continue
            d_ref = float(np.linalg.norm(r[i] - r[j]))
            if d_ref <= 0.0 or d_ref >= inclusion_radius:
                continue
            d_mod = float(np.linalg.norm(m[i] - m[j]))
            delta = abs(d_ref - d_mod)
            per_res_fracs[res[i]].append(sum(1 for t in thresholds if delta < t) / len(thresholds))
    return [sum(f) / len(f) if f else None for f in per_res_fracs]


def _brute_force_lddt(model: Protein, reference: Protein, *, atom_set: str = "ca") -> float:
    """Global lDDT = mean of per-residue values, dropping empty residues."""
    per_res = [
        v
        for v in _brute_force_lddt_per_residue(model, reference, atom_set=atom_set)
        if v is not None
    ]
    return sum(per_res) / len(per_res) if per_res else 0.0


def _heavy_protein(coords: np.ndarray, residue_ids: list[int]) -> Protein:
    """Build an all-heavy-atom protein from coords + a residue id per atom."""
    n = coords.shape[0]
    return Protein(
        AtomArray.from_dict(
            {
                "coords": coords.astype(np.float32),
                "atom_name": np.array([f"C{i}" for i in range(n)], dtype="U4"),
                "element": np.array(["C"] * n, dtype="U2"),
                "residue_id": np.array(residue_ids, dtype="int32"),
                "chain_id": np.array(["A"] * n, dtype="U4"),
                "entity_type": np.array(["protein"] * n, dtype="U8"),
            }
        )
    )


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


class TestAllAtom:
    """The ``atom_set="heavy"`` (all-atom) lDDT variant."""

    def test_identical_is_one(self) -> None:
        p = read_pdb(FIXTURES / "real_ubiquitin.pdb")
        assert lddt(p, p, atom_set="heavy") == pytest.approx(1.0)

    def test_hand_computed_case_excludes_intra_residue(self) -> None:
        # Two residues, two heavy atoms each:
        #   res1: A1=(0,0,0)  A2=(1,0,0)      res2: B1=(5,0,0)  B2=(6,0,0)
        # The model moves B2 to x=6.6, so only distances *to B2* change by
        # 0.6 Å. The intra-residue pair B1-B2 is EXCLUDED, so it doesn't
        # count. Inter-residue pair scores (|Δd|=0 -> 1.00, |Δd|=0.6 -> 0.75):
        #   res1 atoms' partners: A1-B1(1.00) A1-B2(0.75) A2-B1(1.00) A2-B2(0.75)
        #     -> mean 0.875
        #   res2 atoms' partners: B1-A1(1.00) B1-A2(1.00) B2-A1(0.75) B2-A2(0.75)
        #     -> mean 0.875
        # global = 0.875. (Were B1-B2 wrongly included, res2 would shift.)
        ref = _heavy_protein(
            np.array([[0.0, 0, 0], [1.0, 0, 0], [5.0, 0, 0], [6.0, 0, 0]]), [1, 1, 2, 2]
        )
        model = _heavy_protein(
            np.array([[0.0, 0, 0], [1.0, 0, 0], [5.0, 0, 0], [6.6, 0, 0]]), [1, 1, 2, 2]
        )
        assert lddt(model, ref, atom_set="heavy") == pytest.approx(0.875, abs=1e-6)
        per_res = lddt_per_residue(model, ref, atom_set="heavy")
        np.testing.assert_allclose(per_res, [0.875, 0.875], atol=1e-6)

    def test_global_matches_brute_force_on_ubiquitin(self) -> None:
        ref = read_pdb(FIXTURES / "real_ubiquitin.pdb")
        model = deepcopy(ref)
        rng = np.random.default_rng(3)
        noise = rng.normal(0.0, 0.7, model.atom_array.coords.shape).astype(np.float32)
        model.atom_array.coords[:] = model.atom_array.coords + noise
        expected = _brute_force_lddt(model, ref, atom_set="heavy")
        assert 0.3 < expected < 0.95
        assert lddt(model, ref, atom_set="heavy") == pytest.approx(expected, abs=1e-4)

    def test_per_residue_matches_brute_force_on_ubiquitin(self) -> None:
        ref = read_pdb(FIXTURES / "real_ubiquitin.pdb")
        model = deepcopy(ref)
        rng = np.random.default_rng(4)
        noise = rng.normal(0.0, 0.5, model.atom_array.coords.shape).astype(np.float32)
        model.atom_array.coords[:] = model.atom_array.coords + noise
        expected = _brute_force_lddt_per_residue(model, ref, atom_set="heavy")
        got = lddt_per_residue(model, ref, atom_set="heavy")
        assert got.shape[0] == len(expected)
        for i, (g, e) in enumerate(zip(got.tolist(), expected, strict=True)):
            if e is None:
                assert np.isnan(g), f"residue {i}"
            else:
                assert g == pytest.approx(e, abs=1e-4), f"residue {i}"

    def test_unknown_atom_set_raises(self) -> None:
        p = read_pdb(FIXTURES / "helix.pdb")
        with pytest.raises(ValueError, match="unknown atom_set"):
            lddt(p, p, atom_set="sidechain")  # type: ignore[arg-type]

    def test_mismatched_heavy_atom_counts_raise(self) -> None:
        ref = _heavy_protein(np.array([[0.0, 0, 0], [1.0, 0, 0], [5.0, 0, 0]]), [1, 1, 2])
        model = _heavy_protein(np.array([[0.0, 0, 0], [1.0, 0, 0]]), [1, 1])
        with pytest.raises(ValueError, match="matched heavy atoms"):
            lddt(model, ref, atom_set="heavy")


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
