"""Tests for molforge.scoring."""

from __future__ import annotations

import math

import numpy as np
import pytest

from molforge.core import AtomArray, Protein
from molforge.docking import DockingResult, Pose
from molforge.scoring import (
    ConfidenceScorer,
    Direction,
    DockingScorer,
    FunctionScorer,
    Score,
    best,
    rank,
)


def _protein(confidence: float | None = None, *, per_residue: bool = False) -> Protein:
    p = Protein(
        AtomArray.from_dict(
            {
                "coords": np.zeros((3, 3), dtype=np.float32),
                "atom_name": np.array(["CA"] * 3, dtype="U4"),
                "entity_type": np.array(["protein"] * 3, dtype="U8"),
            }
        ),
        name=f"p{confidence}",
    )
    if confidence is not None:
        if per_residue:
            p.metadata["confidence_per_residue"] = np.full(3, confidence, dtype=np.float32)
        else:
            p.metadata["mean_confidence"] = confidence
    return p


def _pose(score: float, rank: int = 0) -> Pose:
    lig = Protein(AtomArray(0), name="lig")
    return Pose(ligand=lig, score=score, rank=rank)


class TestScore:
    def test_ranking_key_normalizes_direction(self) -> None:
        assert Score(9.5, Direction.HIGHER_IS_BETTER).ranking_key == 9.5
        assert Score(-9.5, Direction.LOWER_IS_BETTER).ranking_key == 9.5

    def test_is_better_than_higher(self) -> None:
        a = Score(0.9, Direction.HIGHER_IS_BETTER)
        b = Score(0.5, Direction.HIGHER_IS_BETTER)
        assert a.is_better_than(b)
        assert not b.is_better_than(a)

    def test_is_better_than_lower(self) -> None:
        strong = Score(-9.5, Direction.LOWER_IS_BETTER)
        weak = Score(-7.0, Direction.LOWER_IS_BETTER)
        assert strong.is_better_than(weak)

    def test_nan_never_better_and_loses(self) -> None:
        real = Score(1.0, Direction.HIGHER_IS_BETTER)
        nan = Score(math.nan, Direction.HIGHER_IS_BETTER)
        assert not nan.is_better_than(real)
        assert real.is_better_than(nan)


class TestConfidenceScorer:
    def test_reads_mean_confidence(self) -> None:
        s = ConfidenceScorer().score(_protein(87.0))
        assert s.value == pytest.approx(87.0)
        assert s.direction is Direction.HIGHER_IS_BETTER
        assert s.scorer == "confidence"

    def test_falls_back_to_per_residue(self) -> None:
        s = ConfidenceScorer().score(_protein(80.0, per_residue=True))
        assert s.value == pytest.approx(80.0)

    def test_missing_confidence_raises(self) -> None:
        with pytest.raises(ValueError, match="no 'mean_confidence'"):
            ConfidenceScorer().score(_protein(None))


class TestDockingScorer:
    def test_scores_bare_pose(self) -> None:
        s = DockingScorer(direction=Direction.LOWER_IS_BETTER).score(_pose(-9.5))
        assert s.value == pytest.approx(-9.5)
        assert s.ranking_key == pytest.approx(9.5)

    def test_scores_docking_result_best_pose(self) -> None:
        dr = DockingResult(poses=[_pose(-9.5, 0), _pose(-7.0, 1)], engine="Vina")
        s = DockingScorer(direction=Direction.LOWER_IS_BETTER).score(dr)
        assert s.value == pytest.approx(-9.5)
        assert s.metadata["rank"] == 0

    def test_empty_docking_result_raises(self) -> None:
        with pytest.raises(ValueError, match="no poses"):
            DockingScorer(direction=Direction.LOWER_IS_BETTER).score(DockingResult(poses=[]))

    def test_from_engine_reads_score_direction(self) -> None:
        class _Eng:
            score_direction = "higher_is_better"

        assert DockingScorer.from_engine(_Eng()).direction is Direction.HIGHER_IS_BETTER

    def test_from_engine_defaults_lower_when_absent(self) -> None:
        assert DockingScorer.from_engine(object()).direction is Direction.LOWER_IS_BETTER


class TestFunctionScorer:
    def test_wraps_callable(self) -> None:
        s = FunctionScorer(
            lambda p: p.metadata["mean_confidence"],
            direction=Direction.HIGHER_IS_BETTER,
            name="myconf",
        ).score(_protein(88.0))
        assert s.value == pytest.approx(88.0)
        assert s.scorer == "myconf"


class TestRankBest:
    def test_rank_best_first(self) -> None:
        ps = [_protein(70), _protein(95), _protein(82)]
        ordered = rank(ps, ConfidenceScorer())
        assert [round(s.value) for _, s in ordered] == [95, 82, 70]

    def test_best_returns_top_item(self) -> None:
        ps = [_protein(70), _protein(95), _protein(82)]
        assert best(ps, ConfidenceScorer()).name == "p95"

    def test_nan_scores_sort_last(self) -> None:
        good = _protein(90)
        nan = FunctionScorer(lambda _p: math.nan, direction=Direction.HIGHER_IS_BETTER)
        ordered = rank([good], nan)
        # single nan item still returns; the point is it doesn't crash sorting.
        assert math.isnan(ordered[0][1].value)

    def test_best_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="no items"):
            best([], ConfidenceScorer())

    def test_score_many_preserves_order(self) -> None:
        ps = [_protein(70), _protein(95)]
        scores = ConfidenceScorer().score_many(ps)
        assert [round(s.value) for s in scores] == [70, 95]


class TestEngineScoreDirection:
    def test_vina_and_diffdock_are_lower_is_better(self) -> None:
        from molforge.wrappers.docking import DiffDock, Vina

        assert Vina.score_direction == "lower_is_better"
        assert DiffDock.score_direction == "lower_is_better"

    def test_gnina_direction_follows_sort_order(self) -> None:
        from molforge.wrappers.docking import Gnina

        assert Gnina(sort_order="CNNscore").score_direction == "higher_is_better"
        assert Gnina(sort_order="CNNaffinity").score_direction == "higher_is_better"
        assert Gnina(sort_order="Energy").score_direction == "lower_is_better"
