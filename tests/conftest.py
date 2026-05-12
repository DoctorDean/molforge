"""Shared pytest fixtures.

Place common fixtures (paths to test PDBs, mock structures, etc.) here so
they're auto-discovered by all tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Path to the tests/fixtures/ directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_pdb(fixtures_dir: Path) -> Path:
    """Path to a small sample PDB file (placeholder)."""
    return fixtures_dir / "pdb" / "sample.pdb"
