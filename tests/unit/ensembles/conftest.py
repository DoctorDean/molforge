"""Shared fixtures for ensemble tests.

Builds synthetic poses with controllable geometries so tests can
assert on exact RMSDs, cluster membership, and density.
"""

from __future__ import annotations

import numpy as np
import pytest

from molforge.core import AtomArray, Protein
from molforge.docking import Pose


def make_ligand(coords: np.ndarray, *, with_hydrogens: bool = False) -> Protein:
    """Build a minimal ligand Protein from explicit Cartesian coords.

    All atoms are flagged as ``entity_type='ligand'``. Elements alternate
    C and N for visual variety; if ``with_hydrogens`` is True, the
    second half of the atoms are flagged as element 'H'.
    """
    n = coords.shape[0]
    arr = AtomArray(n)
    arr.coords[:] = coords.astype(np.float32)
    arr.entity_type[:] = "ligand"
    arr.atom_name[:] = [f"A{i}" for i in range(n)]
    arr.residue_name[:] = "LIG"
    arr.residue_id[:] = 1
    arr.chain_id[:] = "L"
    if with_hydrogens:
        half = n // 2
        arr.element[:half] = "C"
        arr.element[half:] = "H"
    else:
        arr.element[:] = ["C" if i % 2 == 0 else "N" for i in range(n)]
    return Protein(arr, name="lig")


def make_pose(
    coords: np.ndarray,
    score: float,
    rank: int = 0,
    *,
    with_hydrogens: bool = False,
) -> Pose:
    """Build a Pose with the given ligand coords and score."""
    return Pose(
        ligand=make_ligand(coords, with_hydrogens=with_hydrogens),
        score=score,
        rank=rank,
    )


@pytest.fixture
def two_identical_poses():
    """Two poses with identical coordinates (RMSD = 0 between them)."""
    coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32)
    return [make_pose(coords, score=-9.0, rank=0), make_pose(coords, score=-8.0, rank=1)]


@pytest.fixture
def three_collinear_poses():
    """Three poses along the x-axis, shifted by 1 Å, 2 Å.

    Pose 0: atoms at (0, 1.5, 3.0) along x.
    Pose 1: pose 0 shifted by +1 in x.
    Pose 2: pose 0 shifted by +2 in x.
    """
    base = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32)
    shift1 = base + np.array([1.0, 0.0, 0.0], dtype=np.float32)
    shift2 = base + np.array([2.0, 0.0, 0.0], dtype=np.float32)
    return [
        make_pose(base, score=-9.0, rank=0),
        make_pose(shift1, score=-8.5, rank=1),
        make_pose(shift2, score=-7.0, rank=2),
    ]


@pytest.fixture
def two_clusters_poses():
    """Five poses in two clear clusters separated by ~10 Å.

    Cluster A (3 poses near origin): poses 0, 2, 4 within ~0.5 Å of each other.
    Cluster B (2 poses far away): poses 1, 3 within ~0.5 Å.
    Order interleaves so clustering must actually do work.
    """
    base_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    base_b = np.array([[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [12.0, 0.0, 0.0]], dtype=np.float32)
    rng = np.random.default_rng(42)
    poses = [
        make_pose(
            base_a + 0.1 * rng.standard_normal((3, 3)).astype(np.float32), score=-10.0, rank=0
        ),
        make_pose(
            base_b + 0.1 * rng.standard_normal((3, 3)).astype(np.float32), score=-9.5, rank=1
        ),
        make_pose(
            base_a + 0.1 * rng.standard_normal((3, 3)).astype(np.float32), score=-9.0, rank=2
        ),
        make_pose(
            base_b + 0.1 * rng.standard_normal((3, 3)).astype(np.float32), score=-8.5, rank=3
        ),
        make_pose(
            base_a + 0.1 * rng.standard_normal((3, 3)).astype(np.float32), score=-8.0, rank=4
        ),
    ]
    return poses


@pytest.fixture
def poses_with_hydrogens():
    """Two poses with explicit hydrogens that should be stripped by default.

    Each pose has 4 atoms: 2 carbons (heavy) + 2 hydrogens.
    The carbons are at the same position across both poses, hydrogens differ.
    With heavy_atoms_only=True the RMSD should be 0.
    With heavy_atoms_only=False the RMSD should be > 0.
    """
    pose0_coords = np.array(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.5, 0.5, 0.0], [1.0, 0.5, 0.0]],
        dtype=np.float32,
    )
    pose1_coords = np.array(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [10.0, 10.0, 10.0], [11.0, 11.0, 11.0]],
        dtype=np.float32,
    )
    return [
        make_pose(pose0_coords, score=-9.0, rank=0, with_hydrogens=True),
        make_pose(pose1_coords, score=-8.0, rank=1, with_hydrogens=True),
    ]


@pytest.fixture
def single_pose():
    """One isolated pose, for edge cases."""
    coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float32)
    return [make_pose(coords, score=-9.0, rank=0)]
