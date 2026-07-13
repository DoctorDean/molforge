"""Tests for the cinnabar network ingest.

cinnabar isn't a dependency, so these use fakes mimicking its
``get_absolute_dataframe()`` output: a ``_DF`` exposing ``.columns`` and
``.to_dict(orient="records")``, and a ``_FEMap`` exposing
``get_absolute_dataframe()``.
"""

from __future__ import annotations

import pytest

from molforge.freeenergy import FreeEnergyRanking
from molforge.wrappers.freeenergy import from_cinnabar


class _DF:
    def __init__(self, columns: list[str], rows: list[dict]) -> None:
        self.columns = columns
        self._rows = rows

    def to_dict(self, orient: str | None = None) -> list[dict]:
        assert orient == "records"
        return list(self._rows)


class _FEMap:
    def __init__(self, df: _DF) -> None:
        self._df = df

    def get_absolute_dataframe(self) -> _DF:
        return self._df


class _Quantity:
    """openff.units.Quantity stand-in: value via .magnitude."""

    def __init__(self, magnitude: float) -> None:
        self.magnitude = magnitude


def _standard_df() -> _DF:
    return _DF(
        ["label", "DG", "uncertainty", "source", "computational"],
        [
            {
                "label": "lig1",
                "DG": -9.5,
                "uncertainty": 0.3,
                "source": "calc",
                "computational": True,
            },
            {
                "label": "lig2",
                "DG": -11.2,
                "uncertainty": 0.4,
                "source": "calc",
                "computational": True,
            },
            {
                "label": "lig3",
                "DG": -8.0,
                "uncertainty": 0.2,
                "source": "exp",
                "computational": False,
            },
        ],
    )


class TestFromCinnabar:
    def test_reads_from_femap(self) -> None:
        results = from_cinnabar(_FEMap(_standard_df()))
        assert set(results) == {"lig1", "lig2"}  # experimental row dropped
        assert results["lig2"].delta_g == pytest.approx(-11.2)
        assert results["lig2"].uncertainty == pytest.approx(0.4)
        assert results["lig2"].method == "FEP (network)"
        assert results["lig2"].components is None

    def test_reads_dataframe_directly(self) -> None:
        results = from_cinnabar(_standard_df())
        assert set(results) == {"lig1", "lig2"}

    def test_metadata(self) -> None:
        r = from_cinnabar(_standard_df())["lig1"]
        assert r.metadata["source"] == "cinnabar"
        assert r.metadata["computational"] is True
        assert r.metadata["cinnabar_source"] == "calc"

    def test_computational_only_false_keeps_experimental(self) -> None:
        results = from_cinnabar(_standard_df(), computational_only=False)
        assert set(results) == {"lig1", "lig2", "lig3"}
        assert results["lig3"].metadata["computational"] is False

    def test_no_computational_column_keeps_all(self) -> None:
        df = _DF(
            ["label", "DG", "uncertainty"],
            [
                {"label": "a", "DG": -5.0, "uncertainty": 0.1},
                {"label": "b", "DG": -6.0, "uncertainty": 0.1},
            ],
        )
        assert set(from_cinnabar(df)) == {"a", "b"}

    def test_unit_suffixed_columns_and_quantity_values(self) -> None:
        df = _DF(
            ["label", "DG (kcal/mol)", "uncertainty (kcal/mol)"],
            [
                {
                    "label": "L",
                    "DG (kcal/mol)": _Quantity(-7.7),
                    "uncertainty (kcal/mol)": _Quantity(0.25),
                }
            ],
        )
        r = from_cinnabar(df)["L"]
        assert r.delta_g == pytest.approx(-7.7)
        assert r.uncertainty == pytest.approx(0.25)

    def test_extra_metadata_merged(self) -> None:
        r = from_cinnabar(_standard_df(), metadata={"campaign": "series-3"})["lig1"]
        assert r.metadata["campaign"] == "series-3"

    def test_missing_dg_column_raises(self) -> None:
        with pytest.raises(ValueError, match="DG"):
            from_cinnabar(_DF(["label", "uncertainty"], []))

    def test_ranks_the_network(self) -> None:
        ranking = FreeEnergyRanking(from_cinnabar(_standard_df()))
        assert ranking.best[0] == "lig2"
        assert [name for name, _ in ranking.ranked] == ["lig2", "lig1"]
        ddg = ranking.delta_delta_g("lig1", "lig2")
        assert ddg.value == pytest.approx(-1.7)  # network offset cancels
