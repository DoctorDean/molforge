"""Tests for the engine version helper (molforge.wrappers._versions).

The wrappers can't run their real (GPU) engines in CI, so these tests
drive the version helper directly by monkeypatching importlib.metadata,
covering: version recording, the below-minimum and above-tested_max
warnings, graceful behaviour when the distribution is absent or the
version is unparseable, and the packaging-missing fallback.
"""

from __future__ import annotations

import warnings

import pytest

from molforge.wrappers import _versions
from molforge.wrappers._versions import check_engine_version, engine_version


def _fake_version(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    """Make importlib.metadata.version return ``value`` (or raise if None)."""

    def _version(_dist: str) -> str:
        if value is None:
            from importlib.metadata import PackageNotFoundError

            raise PackageNotFoundError("not installed")
        return value

    monkeypatch.setattr(_versions.metadata, "version", _version)


class TestEngineVersion:
    def test_returns_installed_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_version(monkeypatch, "4.41.2")
        assert engine_version("transformers") == "4.41.2"

    def test_absent_distribution_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_version(monkeypatch, None)
        assert engine_version("nope") == ""


class TestCheckEngineVersion:
    def test_in_range_does_not_warn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_version(monkeypatch, "4.41.0")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            v = check_engine_version("transformers", engine="ESMFold", minimum="4.40")
        assert v == "4.41.0"

    def test_below_minimum_warns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_version(monkeypatch, "4.30.0")
        with pytest.warns(UserWarning, match="older than the minimum 4.40"):
            v = check_engine_version("transformers", engine="ESMFold", minimum="4.40")
        assert v == "4.30.0"

    def test_above_tested_max_warns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_version(monkeypatch, "1.0.0")
        with pytest.warns(UserWarning, match="newer than the last version"):
            check_engine_version("boltz", engine="Boltz", tested_max="0.4")

    def test_no_bounds_records_without_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_version(monkeypatch, "0.6.1")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            v = check_engine_version("chai_lab", engine="Chai-1")
        assert v == "0.6.1"

    def test_absent_distribution_no_warning_empty_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fake_version(monkeypatch, None)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            v = check_engine_version("transformers", engine="ESMFold", minimum="4.40")
        assert v == ""

    def test_unparseable_version_skips_range_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A VCS / dev build string isn't PEP 440 — record it, don't warn.
        _fake_version(monkeypatch, "main-abcdef")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            v = check_engine_version("transformers", engine="ESMFold", minimum="4.40")
        assert v == "main-abcdef"

    def test_packaging_missing_falls_back_gracefully(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # With packaging unavailable, the range check is skipped (no warning),
        # but the version is still recorded.
        _fake_version(monkeypatch, "1.0.0")
        monkeypatch.setattr(_versions, "_parse", lambda _v: None)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            v = check_engine_version("boltz", engine="Boltz", tested_max="0.4")
        assert v == "1.0.0"
