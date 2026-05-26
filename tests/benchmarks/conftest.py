"""Shared fixtures for the performance benchmark suite.

Benchmarks need inputs that are (a) a realistic size — a few hundred
residues, not the handful in the unit-test fixtures — and (b)
reproducible without shipping large PDB files. So we synthesize
proteins parametrically: an idealized alpha-helix backbone with
N, CA, C, O, and CB atoms per residue, generated at whatever length
a benchmark asks for.

The geometry is a regular helix — not a real fold — but it has valid
per-residue backbone atoms with realistic bond lengths, so DSSP,
contact maps, and the rest do genuine work on it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molforge.core import AtomArray, Protein

# Directory this conftest lives in — used by the collection guard
# below to scope the "skip if no pytest-benchmark" behaviour.
_HERE = Path(__file__).resolve().parent


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip the benchmark suite cleanly when pytest-benchmark is absent.

    The benchmarks depend on the ``benchmark`` fixture, which only
    exists when the ``pytest-benchmark`` plugin is installed. Without
    it, every benchmark would *error* on collection ("fixture
    'benchmark' not found"). That's a poor experience for anyone
    running a plain ``pytest`` in a minimal environment. Here we
    detect the missing plugin and convert those errors into clean
    skips for the items under this directory.
    """
    if config.pluginmanager.hasplugin("benchmark"):
        return
    skip = pytest.mark.skip(reason="pytest-benchmark not installed (in the [dev] extra)")
    benchmark_dir = str(_HERE)
    for item in items:
        if str(item.fspath).startswith(benchmark_dir):
            item.add_marker(skip)


# Idealized alpha-helix parameters.
_RISE_PER_RESIDUE = 1.5  # Angstrom along the helix axis
_RESIDUES_PER_TURN = 3.6
_HELIX_RADIUS = 2.3  # Angstrom, CA radius from the axis


def make_helix_protein(n_residues: int, *, seed: int = 0) -> Protein:
    """Build a synthetic poly-alanine alpha-helix of ``n_residues``.

    Each residue gets five atoms (N, CA, C, O, CB) with realistic
    relative offsets. The CA trace is a regular helix; the other
    backbone atoms are placed at small fixed offsets from their CA so
    the result has valid per-residue geometry for DSSP and friends.

    Args:
        n_residues: How many residues to generate.
        seed: Seeds a small random perturbation so two proteins built
            with different seeds differ slightly — useful for the
            RMSD / lDDT benchmarks, which need a model and a
            reference that aren't identical.

    Returns:
        A :class:`Protein` with ``5 * n_residues`` atoms.
    """
    rng = np.random.default_rng(seed)
    n_atoms = n_residues * 5
    arr = AtomArray(n_atoms)

    # Per-atom offsets from the residue's CA position (Angstrom).
    # Order: N, CA, C, O, CB.
    offsets = np.array(
        [
            [-1.20, 0.30, -0.40],  # N
            [0.00, 0.00, 0.00],  # CA
            [1.10, 0.60, 0.55],  # C
            [1.35, 1.80, 0.60],  # O
            [-0.55, -1.05, 1.25],  # CB
        ],
        dtype=np.float32,
    )
    elements = ["N", "C", "C", "O", "C"]
    atom_names = ["N", "CA", "C", "O", "CB"]

    coords = np.empty((n_atoms, 3), dtype=np.float32)
    for i in range(n_residues):
        angle = 2.0 * np.pi * i / _RESIDUES_PER_TURN
        ca = np.array(
            [
                _HELIX_RADIUS * np.cos(angle),
                _HELIX_RADIUS * np.sin(angle),
                _RISE_PER_RESIDUE * i,
            ],
            dtype=np.float32,
        )
        # Small per-residue jitter so seed actually matters.
        jitter = rng.normal(0.0, 0.05, size=3).astype(np.float32)
        base = i * 5
        for j in range(5):
            coords[base + j] = ca + offsets[j] + jitter

    arr.coords[:] = coords
    arr.element[:] = np.tile(elements, n_residues)
    arr.atom_name[:] = np.tile(atom_names, n_residues)
    arr.residue_name[:] = "ALA"
    arr.residue_id[:] = np.repeat(np.arange(1, n_residues + 1), 5)
    arr.chain_id[:] = "A"
    arr.entity_type[:] = "protein"
    arr.b_factor[:] = 50.0
    arr.occupancy[:] = 1.0
    return Protein(arr, name=f"helix{n_residues}")


@pytest.fixture(scope="session")
def helix_200() -> Protein:
    """A 200-residue synthetic helix (1000 atoms)."""
    return make_helix_protein(200, seed=0)


@pytest.fixture(scope="session")
def helix_200_perturbed() -> Protein:
    """A second 200-residue helix, slightly perturbed from ``helix_200``.

    Same length and topology, different coordinates — the pairing the
    RMSD and lDDT benchmarks need.
    """
    return make_helix_protein(200, seed=1)
