"""Tests for molforge.ensembles.cross_engine.

Uses fake folding engines that return CA-only proteins from controllable
geometries, so the ensemble math (superposition, pairwise TM / RMSD,
medoid consensus, per-residue disagreement) can be asserted exactly with
no GPU or model weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from molforge.core import AtomArray, Protein
from molforge.ensembles import CrossEngineEnsemble, cross_engine_fold


def _helix_ca(n: int) -> np.ndarray:
    """Idealized alpha-helix CA trace with n residues."""
    i = np.arange(n)
    theta = np.radians(100.0) * i
    return np.stack([2.3 * np.cos(theta), 2.3 * np.sin(theta), 1.5 * i], axis=1).astype(np.float64)


def _rotation(deg: float, axis: int = 2) -> np.ndarray:
    """A proper rotation of ``deg`` degrees about ``axis``."""
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    r = np.eye(3)
    a = [i for i in range(3) if i != axis]
    r[a[0], a[0]] = c
    r[a[0], a[1]] = -s
    r[a[1], a[0]] = s
    r[a[1], a[1]] = c
    return r


def _make_protein(coords: np.ndarray, *, confidence: float | None = None) -> Protein:
    """Build a CA-only protein from an (n, 3) coordinate array."""
    n = coords.shape[0]
    p = Protein(
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
    if confidence is not None:
        p.metadata["mean_confidence"] = confidence
    return p


class _FakeEngine:
    """A folding engine that returns a fixed structure regardless of sequence."""

    def __init__(self, name: str, coords: np.ndarray, *, confidence: float | None = None) -> None:
        self.name = name
        self._coords = coords
        self._confidence = confidence

    def predict(self, sequence: str, **kwargs: object) -> Protein:
        return _make_protein(self._coords, confidence=self._confidence)


class _FailingEngine:
    name = "Broken"

    def predict(self, sequence: str, **kwargs: object) -> Protein:
        raise RuntimeError("engine crashed")


@pytest.fixture
def three_engines():
    """Three engines folding a 40-residue helix with increasing perturbation.

    Engine 0 is the pristine helix; 1 and 2 add small Gaussian noise. Engine
    0 is the most central, so it is the medoid / default reference.
    """
    base = _helix_ca(40)
    rng = np.random.default_rng(0)
    noisy1 = base + 0.3 * rng.standard_normal((40, 3))
    noisy2 = base + 0.7 * rng.standard_normal((40, 3))
    return [
        _FakeEngine("ESMFold", base, confidence=95.0),
        _FakeEngine("AlphaFold", noisy1, confidence=88.0),
        _FakeEngine("Boltz", noisy2, confidence=80.0),
    ]


class TestBasics:
    def test_returns_ensemble(self, three_engines) -> None:
        ens = cross_engine_fold("A" * 40, three_engines)
        assert isinstance(ens, CrossEngineEnsemble)
        assert ens.n_members == 3
        assert ens.engine_names == ["ESMFold", "AlphaFold", "Boltz"]
        assert ens.sequence == "A" * 40

    def test_matrix_shapes_and_diagonals(self, three_engines) -> None:
        ens = cross_engine_fold("A" * 40, three_engines)
        assert ens.tm_matrix.shape == (3, 3)
        assert ens.rmsd_matrix.shape == (3, 3)
        assert np.allclose(np.diag(ens.tm_matrix), 1.0)
        assert np.allclose(np.diag(ens.rmsd_matrix), 0.0)

    def test_matrices_symmetric(self, three_engines) -> None:
        ens = cross_engine_fold("A" * 40, three_engines)
        assert np.allclose(ens.tm_matrix, ens.tm_matrix.T)
        assert np.allclose(ens.rmsd_matrix, ens.rmsd_matrix.T)

    def test_rmsf_shape_matches_length(self, three_engines) -> None:
        ens = cross_engine_fold("A" * 40, three_engines)
        assert ens.per_residue_rmsf.shape == (40,)
        assert ens.per_residue_rmsf.dtype == np.float32
        # disagreement() is an alias for per_residue_rmsf.
        assert np.array_equal(ens.disagreement(), ens.per_residue_rmsf)


class TestRigidInvariance:
    def test_identical_structure_rotated_scores_perfectly(self) -> None:
        """Two engines returning the same structure, one rigidly rotated and
        translated, must agree perfectly after superposition."""
        base = _helix_ca(40)
        moved = (_rotation(57.0) @ base.T).T + np.array([10.0, -5.0, 3.0])
        e1 = _FakeEngine("A", base)
        e2 = _FakeEngine("B", moved)
        ens = cross_engine_fold("A" * 40, [e1, e2])
        assert ens.tm_matrix[0, 1] == pytest.approx(1.0, abs=1e-4)
        assert ens.rmsd_matrix[0, 1] == pytest.approx(0.0, abs=1e-4)
        # RMSF is essentially zero everywhere.
        assert float(ens.per_residue_rmsf.max()) == pytest.approx(0.0, abs=1e-3)

    def test_members_overlay_after_superposition(self) -> None:
        base = _helix_ca(40)
        moved = (_rotation(90.0) @ base.T).T + np.array([1.0, 2.0, 3.0])
        ens = cross_engine_fold("A" * 40, [_FakeEngine("A", base), _FakeEngine("B", moved)])
        c0 = ens.members[0].atom_array.coords
        c1 = ens.members[1].atom_array.coords
        assert np.linalg.norm(c0 - c1, axis=1).max() == pytest.approx(0.0, abs=1e-3)


class TestReferenceSelection:
    def test_default_reference_is_medoid(self, three_engines) -> None:
        ens = cross_engine_fold("A" * 40, three_engines)
        # The pristine helix (engine 0) is most central.
        assert ens.reference_index == 0

    def test_reference_first(self, three_engines) -> None:
        ens = cross_engine_fold("A" * 40, three_engines, reference="first")
        assert ens.reference_index == 0

    def test_reference_most_confident(self, three_engines) -> None:
        ens = cross_engine_fold("A" * 40, three_engines, reference="most_confident")
        # ESMFold has the highest mean_confidence (95).
        assert ens.engine_names[ens.reference_index] == "ESMFold"

    def test_reference_by_engine_name(self, three_engines) -> None:
        ens = cross_engine_fold("A" * 40, three_engines, reference="Boltz")
        assert ens.engine_names[ens.reference_index] == "Boltz"

    def test_most_confident_without_metadata_raises(self) -> None:
        base = _helix_ca(30)
        engines = [_FakeEngine("A", base), _FakeEngine("B", base + 0.1)]
        with pytest.raises(ValueError, match="mean_confidence"):
            cross_engine_fold("A" * 30, engines, reference="most_confident")

    def test_unknown_reference_raises(self, three_engines) -> None:
        with pytest.raises(ValueError, match="unknown reference"):
            cross_engine_fold("A" * 40, three_engines, reference="nope")


class TestConsensusAndSpread:
    def test_consensus_is_medoid_member(self, three_engines) -> None:
        ens = cross_engine_fold("A" * 40, three_engines)
        consensus = ens.consensus()
        assert consensus is ens.members[0]

    def test_consensus_carries_ensemble_metadata(self, three_engines) -> None:
        ens = cross_engine_fold("A" * 40, three_engines)
        consensus = ens.consensus()
        assert consensus.metadata["cross_engine_ensemble"]["reference"] == "ESMFold"
        assert consensus.metadata["cross_engine_ensemble"]["engines"] == ens.engine_names
        assert np.array_equal(consensus.metadata["cross_engine_rmsf"], ens.per_residue_rmsf)

    def test_spread_keys_and_ranges(self, three_engines) -> None:
        s = cross_engine_fold("A" * 40, three_engines).spread()
        assert s["n_members"] == 3.0
        assert 0.0 <= s["tm_min"] <= s["tm_mean"] <= s["tm_max"] <= 1.0
        assert 0.0 <= s["rmsd_min"] <= s["rmsd_mean"] <= s["rmsd_max"]

    def test_perfect_agreement_spread(self) -> None:
        base = _helix_ca(40)
        ens = cross_engine_fold("A" * 40, [_FakeEngine("A", base), _FakeEngine("B", base)])
        s = ens.spread()
        assert s["tm_mean"] == pytest.approx(1.0, abs=1e-4)
        assert s["rmsd_mean"] == pytest.approx(0.0, abs=1e-4)


class TestErrorHandling:
    def test_fewer_than_two_engines_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 engines"):
            cross_engine_fold("A" * 20, [_FakeEngine("A", _helix_ca(20))])

    def test_complex_input_not_implemented(self, three_engines) -> None:
        with pytest.raises(NotImplementedError, match="single-chain"):
            cross_engine_fold(["AAA", "BBB"], three_engines)

    def test_on_error_skip_drops_failing_engine(self) -> None:
        base = _helix_ca(30)
        engines = [_FakeEngine("A", base), _FakeEngine("B", base + 0.2), _FailingEngine()]
        ens = cross_engine_fold("A" * 30, engines, on_error="skip")
        assert ens.engine_names == ["A", "B"]

    def test_on_error_skip_below_two_survivors_raises(self) -> None:
        engines = [_FakeEngine("A", _helix_ca(30)), _FailingEngine()]
        with pytest.raises(ValueError, match="fewer than 2 engines produced"):
            cross_engine_fold("A" * 30, engines, on_error="skip")

    def test_on_error_raise_propagates(self) -> None:
        engines = [_FakeEngine("A", _helix_ca(30)), _FailingEngine()]
        with pytest.raises(RuntimeError, match="engine crashed"):
            cross_engine_fold("A" * 30, engines, on_error="raise")

    def test_mismatched_residue_counts_raise(self) -> None:
        engines = [_FakeEngine("A", _helix_ca(30)), _FakeEngine("B", _helix_ca(25))]
        with pytest.raises(ValueError, match="CA atoms"):
            cross_engine_fold("A" * 30, engines)

    def test_too_few_residues_raise(self) -> None:
        engines = [_FakeEngine("A", _helix_ca(2)), _FakeEngine("B", _helix_ca(2))]
        with pytest.raises(ValueError, match="at least 3 residues"):
            cross_engine_fold("AA", engines)


class TestDuplicateNames:
    def test_duplicate_engine_names_disambiguated(self) -> None:
        base = _helix_ca(30)
        engines = [_FakeEngine("ESMFold", base), _FakeEngine("ESMFold", base + 0.1)]
        ens = cross_engine_fold("A" * 30, engines)
        assert ens.engine_names == ["ESMFold", "ESMFold#2"]
