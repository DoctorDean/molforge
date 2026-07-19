"""Tests for molforge.design (the DesignLoop).

Fake designer / folder / docker engines return controllable outputs so
the loop's orchestration (design → fold → dock → score → iterate),
self-consistency scoring, objectives, ranking, and error paths can be
asserted exactly without any GPU or model weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from molforge.core import AtomArray, Protein
from molforge.design import DesignCandidate, DesignLoop, DesignTable
from molforge.docking import DockingResult, Pose
from molforge.generative import DesignedSequence


def _helix_ca(n: int, *, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    i = np.arange(n)
    theta = np.radians(100.0) * i
    coords = np.stack([2.3 * np.cos(theta), 2.3 * np.sin(theta), 1.5 * i], axis=1).astype(
        np.float64
    )
    if noise:
        coords = coords + noise * np.random.default_rng(seed).standard_normal((n, 3))
    return coords


def _protein(coords: np.ndarray, *, confidence: float | None = None) -> Protein:
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


_L = 30


class _FakeDesigner:
    """Proposes ``n`` sequences (all poly-A of the backbone length) with
    monotonically improving designer scores."""

    name = "FakeMPNN"

    def __init__(self, n: int = 5) -> None:
        self.n = n
        self.calls = 0

    def generate(self, backbone: Protein, **kwargs: object) -> list[DesignedSequence]:
        self.calls += 1
        length = backbone.n_residues
        return [
            DesignedSequence(sequence="A" * length, score=-1.0 * (i + 1)) for i in range(self.n)
        ]


class _FakeFolder:
    """Folds to a helix with a per-call noise level, so self-consistency
    varies deterministically across the candidates in a round."""

    def __init__(self, name: str, noise_schedule: list[float]) -> None:
        self.name = name
        self.parallelism = "serial"
        self.noise_schedule = noise_schedule
        self._i = 0

    def predict(self, sequence: str, **kwargs: object) -> Protein:
        noise = self.noise_schedule[self._i % len(self.noise_schedule)]
        self._i += 1
        return _protein(
            _helix_ca(len(sequence), noise=noise, seed=self._i),
            confidence=max(0.0, 100.0 - 10.0 * noise),
        )


class _FakeDocker:
    name = "FakeVina"
    parallelism = "serial"

    def __init__(self, score: float = -9.5) -> None:
        self.score = score

    def dock(self, receptor: Protein, ligand: object, **kwargs: object) -> DockingResult:
        return DockingResult(
            poses=[
                Pose(ligand=receptor, score=self.score, rank=0),
                Pose(ligand=receptor, score=self.score + 2.0, rank=1),
            ],
            receptor=receptor,
            engine="FakeVina",
        )


@pytest.fixture
def backbone() -> Protein:
    return _protein(_helix_ca(_L))


class TestSingleFolderLoop:
    def test_returns_design_table(self, backbone) -> None:
        loop = DesignLoop(designer=_FakeDesigner(3), folder=_FakeFolder("F", [0.0]), n_designs=3)
        table = loop.run(backbone)
        assert isinstance(table, DesignTable)
        assert len(table) == 3
        assert table.objective == "self_consistency"
        assert all(isinstance(c, DesignCandidate) for c in table)

    def test_self_consistency_metrics_recorded(self, backbone) -> None:
        loop = DesignLoop(designer=_FakeDesigner(3), folder=_FakeFolder("F", [0.0, 1.0, 2.0]))
        table = loop.run(backbone)
        for c in table:
            assert "sc_tm" in c.metrics
            assert "sc_rmsd" in c.metrics
            assert "plddt" in c.metrics
            assert "mpnn_score" in c.metrics

    def test_perfect_fold_scores_one(self, backbone) -> None:
        loop = DesignLoop(designer=_FakeDesigner(2), folder=_FakeFolder("F", [0.0]))
        table = loop.run(backbone)
        # Zero-noise fold reproduces the backbone exactly: scTM == 1, scRMSD == 0.
        assert table.best.metrics["sc_tm"] == pytest.approx(1.0, abs=1e-4)
        assert table.best.metrics["sc_rmsd"] == pytest.approx(0.0, abs=1e-4)
        assert table.best.score == pytest.approx(1.0, abs=1e-4)

    def test_candidates_ranked_best_first(self, backbone) -> None:
        loop = DesignLoop(designer=_FakeDesigner(4), folder=_FakeFolder("F", [0.0, 1.0, 2.0, 3.0]))
        table = loop.run(backbone)
        scores = [c.score for c in table]
        assert scores == sorted(scores, reverse=True)

    def test_n_designs_caps_candidates(self, backbone) -> None:
        loop = DesignLoop(designer=_FakeDesigner(10), folder=_FakeFolder("F", [0.0]), n_designs=3)
        table = loop.run(backbone)
        assert len(table) == 3


class TestIteration:
    def test_multiple_rounds_accumulate(self, backbone) -> None:
        loop = DesignLoop(
            designer=_FakeDesigner(5),
            folder=_FakeFolder("F", [0.0, 0.5, 1.0]),
            n_designs=5,
            n_rounds=3,
            select_top=2,
        )
        table = loop.run(backbone)
        # Round 0: 1 backbone × 5. Rounds 1,2: 2 seeds × 5 each.
        assert len(table) == 5 + 5 * 2 + 5 * 2
        assert sorted({c.round for c in table}) == [0, 1, 2]

    def test_designer_reinvoked_per_round(self, backbone) -> None:
        designer = _FakeDesigner(3)
        loop = DesignLoop(
            designer=designer, folder=_FakeFolder("F", [0.0]), n_designs=3, n_rounds=2, select_top=1
        )
        loop.run(backbone)
        # Round 0: 1 call. Round 1: 1 seed → 1 call. Total 2.
        assert designer.calls == 2


class TestCrossEngineFolder:
    def test_list_folder_records_consensus_metrics(self, backbone) -> None:
        loop = DesignLoop(
            designer=_FakeDesigner(2),
            folder=[_FakeFolder("E1", [0.0]), _FakeFolder("E2", [0.3]), _FakeFolder("E3", [0.6])],
            n_designs=2,
        )
        table = loop.run(backbone)
        for c in table:
            assert "cross_engine_tm_mean" in c.metrics
            assert "cross_engine_rmsf_mean" in c.metrics
            assert c.metrics["cross_engine_rmsf_mean"] >= 0.0
            # Consensus still yields a self-consistency score against the backbone.
            assert "sc_tm" in c.metrics


class TestObjectives:
    def test_plddt_objective(self, backbone) -> None:
        loop = DesignLoop(
            designer=_FakeDesigner(3), folder=_FakeFolder("F", [0.0]), objective="plddt"
        )
        table = loop.run(backbone)
        assert table.objective == "plddt"
        assert table.best.score == pytest.approx(100.0, abs=1e-4)

    def test_affinity_objective(self, backbone) -> None:
        loop = DesignLoop(
            designer=_FakeDesigner(3),
            folder=_FakeFolder("F", [0.0]),
            docker=_FakeDocker(score=-9.5),
            objective="affinity",
        )
        table = loop.run(backbone, receptor=_protein(_helix_ca(50)))
        # Best pose -9.5 kcal/mol; negated so higher-is-better → 9.5.
        assert table.best.metrics["affinity"] == pytest.approx(-9.5)
        assert table.best.score == pytest.approx(9.5)
        assert table.best.docking is not None

    def test_custom_callable_objective(self, backbone) -> None:
        loop = DesignLoop(
            designer=_FakeDesigner(3),
            folder=_FakeFolder("F", [0.0]),
            objective=lambda c: c.metrics.get("mpnn_score", 0.0),
        )
        table = loop.run(backbone)
        assert table.objective == "custom"
        # mpnn scores are -1, -2, -3; highest (best) is -1.
        assert table.best.score == pytest.approx(-1.0)

    def test_scorer_objective(self, backbone) -> None:
        from molforge.scoring import ConfidenceScorer

        # Varying noise → folder confidence 100/90/80 (conf = 100 - 10*noise);
        # a ConfidenceScorer objective ranks the folded structures by pLDDT.
        loop = DesignLoop(
            designer=_FakeDesigner(3),
            folder=_FakeFolder("F", [0.0, 1.0, 2.0]),
            objective=ConfidenceScorer(),
        )
        table = loop.run(backbone)
        assert table.objective == "confidence"
        scores = [c.score for c in table]
        assert scores == pytest.approx([100.0, 90.0, 80.0])


class TestToRecords:
    def test_records_are_flat_and_aligned(self, backbone) -> None:
        loop = DesignLoop(designer=_FakeDesigner(3), folder=_FakeFolder("F", [0.0, 1.0, 2.0]))
        records = loop.run(backbone).to_records()
        assert len(records) == 3
        for row in records:
            assert {"round", "score", "sequence"} <= set(row)
            assert "sc_tm" in row and "plddt" in row


class TestErrorHandling:
    def test_generator_slot_deferred(self, backbone) -> None:
        with pytest.raises(NotImplementedError, match="backbone generation"):
            DesignLoop(
                designer=_FakeDesigner(), folder=_FakeFolder("F", [0.0]), generator=_FakeDesigner()
            )

    def test_affinity_without_docker_raises(self) -> None:
        with pytest.raises(ValueError, match="requires a docker"):
            DesignLoop(
                designer=_FakeDesigner(), folder=_FakeFolder("F", [0.0]), objective="affinity"
            )

    def test_docker_without_receptor_raises(self, backbone) -> None:
        loop = DesignLoop(
            designer=_FakeDesigner(3), folder=_FakeFolder("F", [0.0]), docker=_FakeDocker()
        )
        with pytest.raises(ValueError, match="needs a receptor"):
            loop.run(backbone)

    @pytest.mark.parametrize("param", ["n_designs", "n_rounds", "select_top"])
    def test_non_positive_params_raise(self, param: str) -> None:
        with pytest.raises(ValueError, match=f"{param} must be"):
            DesignLoop(designer=_FakeDesigner(), folder=_FakeFolder("F", [0.0]), **{param: 0})

    def test_unknown_objective_raises(self) -> None:
        # Objective is resolved at construction (fail-fast), not at run().
        with pytest.raises(ValueError, match="unknown objective"):
            DesignLoop(
                designer=_FakeDesigner(3),
                folder=_FakeFolder("F", [0.0]),
                objective="bogus",  # type: ignore[arg-type]
            )
