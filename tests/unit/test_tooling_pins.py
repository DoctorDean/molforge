"""The ruff pin must agree between pyproject and pre-commit.

``pyproject.toml`` pins an exact ruff version and says why: "pinned to the
pre-commit rev so CI and local agree". Nothing enforced it, so a bot that
bumps one file and not the other silently splits the two — CI formats with
one ruff, the hook with another, and the difference shows up as a
mysteriously dirty tree after every commit.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"
PRE_COMMIT = ROOT / ".pre-commit-config.yaml"


def _pyproject_ruff_pin() -> str:
    match = re.search(r'"ruff==([0-9][^"]*)"', PYPROJECT.read_text(encoding="utf-8"))
    assert match, "pyproject.toml no longer pins an exact ruff version"
    return match.group(1)


def _pre_commit_ruff_rev() -> str:
    text = PRE_COMMIT.read_text(encoding="utf-8")
    match = re.search(
        r"repo:\s*https://github\.com/astral-sh/ruff-pre-commit\s*\n\s*rev:\s*v?([0-9][^\s]*)",
        text,
    )
    assert match, ".pre-commit-config.yaml no longer pins the ruff-pre-commit rev"
    return match.group(1)


class TestRuffPin:
    def test_config_files_exist(self) -> None:
        assert PYPROJECT.is_file()
        assert PRE_COMMIT.is_file()

    def test_pins_agree(self) -> None:
        pyproject = _pyproject_ruff_pin()
        pre_commit = _pre_commit_ruff_rev()
        assert pyproject == pre_commit, (
            f"ruff pins have drifted: pyproject.toml has {pyproject}, "
            f".pre-commit-config.yaml has v{pre_commit}. Bump both together — "
            "CI and the hook must run the same formatter."
        )

    @pytest.mark.parametrize("pin", ["pyproject", "pre_commit"])
    def test_pins_are_exact(self, pin: str) -> None:
        """A range would reintroduce the drift this guards against."""
        value = _pyproject_ruff_pin() if pin == "pyproject" else _pre_commit_ruff_rev()
        assert re.fullmatch(r"\d+\.\d+\.\d+", value), f"{pin} ruff pin {value!r} is not exact"


class TestFormatterScope:
    def test_markdown_is_excluded_from_the_formatter(self) -> None:
        """ruff 0.16 began formatting Python blocks inside Markdown, which
        flattens the hand-aligned examples the docs rely on. The exclusion
        is deliberate; losing it would silently rewrite the docs."""
        text = PYPROJECT.read_text(encoding="utf-8")
        assert "[tool.ruff.format]" in text
        section = text.split("[tool.ruff.format]", 1)[1].split("[tool.ruff.lint]", 1)[0]
        assert '"*.md"' in section
