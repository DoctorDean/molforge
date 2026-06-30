"""Shared pytest fixtures.

Place common fixtures (paths to test PDBs, mock structures, etc.) here so
they're auto-discovered by all tests.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Point molforge's result cache at a per-test temp dir.

    Without this, every test that runs an engine method would pollute
    the user's real ``~/.cache/molforge/`` and see entries leak
    between tests in the same run. ``autouse`` so every test gets
    isolation by default; cache-specific tests can override by
    constructing their own :class:`molforge.cache.Cache` pointed at
    a per-test path.
    """
    import molforge.cache as cache_module

    test_cache_dir = tmp_path_factory.mktemp("molforge_test_cache")
    saved_env = os.environ.get(cache_module.CACHE_DIR_ENV)
    saved_singleton = cache_module._default_cache

    os.environ[cache_module.CACHE_DIR_ENV] = str(test_cache_dir)
    cache_module._reset_default_cache_for_testing()
    try:
        yield
    finally:
        if saved_env is None:
            os.environ.pop(cache_module.CACHE_DIR_ENV, None)
        else:
            os.environ[cache_module.CACHE_DIR_ENV] = saved_env
        cache_module._default_cache = saved_singleton


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Path to the tests/fixtures/ directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_pdb(fixtures_dir: Path) -> Path:
    """Path to a small sample PDB file (placeholder)."""
    return fixtures_dir / "pdb" / "sample.pdb"
