"""Regression guard: ``mypy --strict`` must stay clean on the whole package.

Every module under ``src/molforge/`` passes ``mypy --strict``. This
test runs mypy against the package and fails if any type error is
introduced, so a regression is caught in a normal test run rather
than only in CI.

The mypy configuration (``[tool.mypy]`` in ``pyproject.toml``)
already sets ``strict = true``, so running ``mypy`` on the source
tree is itself a strict check; this test invokes it the same way the
CI ``typecheck`` job does.

The test is skipped (not failed) if mypy isn't installed, so the
suite still runs in minimal environments. It's marked ``slow``
because invoking mypy as a subprocess takes a few seconds — run the
full suite, or ``pytest -m slow``, to include it.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# Repo root: tests/unit/test_typing.py -> up 2 levels.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src" / "molforge"


def _mypy_available() -> bool:
    """True if mypy can be invoked (as a module or on PATH)."""
    if shutil.which("mypy") is not None:
        return True
    try:
        import mypy  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.slow
@pytest.mark.skipif(not _mypy_available(), reason="mypy not installed")
def test_package_is_mypy_strict_clean() -> None:
    """``mypy`` (strict, per pyproject config) must report no errors
    anywhere under ``src/molforge/``."""
    assert _SRC.is_dir(), f"package directory not found: {_SRC}"

    result = subprocess.run(
        [sys.executable, "-m", "mypy", str(_SRC)],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        check=False,
    )

    if result.returncode != 0:
        pytest.fail(
            "mypy --strict found type errors in molforge:\n\n"
            + result.stdout
            + result.stderr
        )
