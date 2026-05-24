"""Regression guard: ``mypy --strict`` must stay clean on molforge.core.

``molforge.core`` is the package's foundation — the data model every
other subpackage builds on — and it is fully ``mypy --strict`` clean.
This test runs mypy against ``src/molforge/core/`` and fails if any
type error is introduced, so a regression is caught in the normal
test run rather than only in CI.

The test is skipped (not failed) if mypy isn't installed, so the
suite still runs in minimal environments. It's marked ``slow``
because invoking mypy as a subprocess takes a couple of seconds —
run the full suite, or ``pytest -m slow``, to include it.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# Repo root: tests/unit/core/test_typing.py -> up 3 levels.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CORE_DIR = _REPO_ROOT / "src" / "molforge" / "core"


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
def test_core_is_mypy_strict_clean() -> None:
    """``mypy --strict src/molforge/core/`` must report no errors."""
    assert _CORE_DIR.is_dir(), f"core directory not found at {_CORE_DIR}"

    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", str(_CORE_DIR)],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        check=False,
    )

    if result.returncode != 0:
        pytest.fail(
            "mypy --strict found type errors in molforge.core:\n\n"
            + result.stdout
            + result.stderr
        )
