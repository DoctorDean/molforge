"""Tests for molforge.validation.report (rolled-up structure quality)."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from molforge.io import read_pdb
from molforge.validation import QualityCheck, QualityReport, report

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "pdb"


def _ubiquitin():
    return read_pdb(FIXTURES / "real_ubiquitin.pdb")


class TestGoodStructure:
    def test_ubiquitin_passes(self) -> None:
        q = report(_ubiquitin())
        assert isinstance(q, QualityReport)
        assert q.passed
        assert q.score == pytest.approx(1.0)

    def test_four_named_checks(self) -> None:
        q = report(_ubiquitin())
        assert [c.name for c in q] == ["clashscore", "ramachandran", "chirality", "bond_length"]
        assert len(q) == 4
        assert all(isinstance(c, QualityCheck) for c in q)

    def test_ubiquitin_ramachandran_mostly_favored(self) -> None:
        # Guards the dihedral sign fix at the report level: a real structure's
        # Ramachandran must pass (near-zero outliers), not fail.
        q = report(_ubiquitin())
        rama = q["ramachandran"]
        assert rama.passed
        assert rama.value < 0.05  # outlier fraction

    def test_getitem_and_missing_key(self) -> None:
        q = report(_ubiquitin())
        assert q["clashscore"].name == "clashscore"
        with pytest.raises(KeyError):
            _ = q["nonexistent"]


class TestBrokenStructure:
    def test_scrambled_coords_fail(self) -> None:
        p = deepcopy(_ubiquitin())
        rng = np.random.default_rng(0)
        p.atom_array.coords[:] = rng.normal(0.0, 1.0, p.atom_array.coords.shape).astype(np.float32)
        q = report(p)
        assert not q.passed
        assert q.score < 1.0
        # Chirality and bond-length collapse under scrambling.
        assert not q["chirality"].passed
        assert not q["bond_length"].passed


class TestThresholds:
    def test_override_tightens_gate(self) -> None:
        p = _ubiquitin()
        assert report(p)["clashscore"].passed  # default clashscore <= 20
        strict = report(p, thresholds={"clashscore": 5.0})
        assert not strict["clashscore"].passed
        assert not strict.passed
        # Only that one check flipped, so score is 3/4.
        assert strict.score == pytest.approx(0.75)

    def test_partial_thresholds_keep_defaults(self) -> None:
        # Overriding one key must not drop the others to zero.
        q = report(_ubiquitin(), thresholds={"chirality_outliers": 5.0})
        assert q["clashscore"].threshold == pytest.approx(20.0)


class TestReportShape:
    def test_score_is_fraction_passed(self) -> None:
        q = report(_ubiquitin())
        n_pass = sum(1 for c in q if c.passed)
        assert q.score == pytest.approx(n_pass / len(q))

    def test_summary_lists_each_check(self) -> None:
        text = report(_ubiquitin()).summary()
        assert "PASS" in text
        for name in ("clashscore", "ramachandran", "chirality", "bond_length"):
            assert name in text

    def test_check_fields(self) -> None:
        c = report(_ubiquitin())["clashscore"]
        assert c.value >= 0.0
        assert c.threshold == pytest.approx(20.0)
        assert isinstance(c.passed, bool)
        assert c.detail
