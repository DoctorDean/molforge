"""Tests for ``molforge.io.trajectory``.

These need mdtraj installed (the wrapper's only hard dep). The whole
module skips cleanly when mdtraj is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mdtraj")

from molforge.io import read_pdb
from molforge.io.trajectory import (
    iter_trajectory,
    read_trajectory,
    write_trajectory,
)
from molforge.md import Trajectory

FIXTURES = Path(__file__).parents[2] / "fixtures"
_TRIPEPTIDE = FIXTURES / "pdb" / "ala_tripeptide_heavy.pdb"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _make_n_frame_dcd(tmp_path: Path, n_frames: int) -> tuple[Path, Path]:
    """Build a synthetic n-frame DCD trajectory from the tripeptide
    fixture and return (dcd_path, topology_path).

    Each frame is the same coordinates so we can identity-check the
    round-trip. The DCD file is the smallest binary format that mdtraj
    writes by default.
    """
    top = read_pdb(_TRIPEPTIDE)
    coords = np.stack(
        [top.atom_array.coords.copy() for _ in range(n_frames)],
        axis=0,
    ).astype(np.float32)
    multi = Trajectory(
        topology=top,
        coordinates=coords,
        times=np.arange(n_frames, dtype=np.float64) * 0.002,
    )
    dcd = tmp_path / "traj.dcd"
    write_trajectory(multi, dcd)
    return dcd, _TRIPEPTIDE


# ---------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------


class TestReadTrajectory:
    def test_reads_single_frame_pdb(self) -> None:
        """A single-model PDB is a valid 1-frame trajectory."""
        traj = read_trajectory(_TRIPEPTIDE)
        assert traj.n_frames == 1
        assert traj.n_atoms == 16

    def test_coordinates_in_angstrom(self) -> None:
        """molforge works in Å but mdtraj returns nm. Reading a PDB
        whose first atom is at (-1.5, -0.5, 0) must yield those values
        in molforge's Å convention — not (-0.15, -0.05, 0) in nm."""
        traj = read_trajectory(_TRIPEPTIDE)
        assert tuple(traj.coordinates[0, 0]) == pytest.approx((-1.5, -0.5, 0.0))

    def test_topology_as_protein_is_reused(self) -> None:
        """When the caller passes a :class:`Protein` topology in,
        the returned trajectory's topology IS that same Protein object
        (no PDB round-trip)."""
        top = read_pdb(_TRIPEPTIDE)
        traj = read_trajectory(_TRIPEPTIDE, topology=top)
        assert traj.topology is top

    def test_topology_as_path_loaded(self) -> None:
        traj = read_trajectory(_TRIPEPTIDE, topology=str(_TRIPEPTIDE))
        assert traj.n_atoms == 16

    def test_topology_none_for_pdb_works(self) -> None:
        """PDB carries its own topology — ``topology=None`` is fine."""
        traj = read_trajectory(_TRIPEPTIDE, topology=None)
        assert traj.n_atoms == 16

    def test_atom_indices_subset(self, tmp_path: Path) -> None:
        """Loading with ``atom_indices`` returns coords and topology
        sliced to that subset, in order."""
        dcd, top_path = _make_n_frame_dcd(tmp_path, n_frames=3)
        traj = read_trajectory(dcd, topology=top_path, atom_indices=[0, 1, 2])
        assert traj.n_atoms == 3
        assert traj.n_frames == 3
        assert traj.topology.atom_array.n_atoms == 3

    def test_atom_indices_with_protein_topology_slices_metadata(self, tmp_path: Path) -> None:
        """When the caller passes a Protein and atom_indices, the
        topology's metadata is preserved on the sliced copy."""
        dcd, _ = _make_n_frame_dcd(tmp_path, n_frames=2)
        top = read_pdb(_TRIPEPTIDE)
        top.metadata = {**top.metadata, "marker": "from-test"}
        traj = read_trajectory(dcd, topology=top, atom_indices=[0, 1, 2])
        assert traj.topology.metadata.get("marker") == "from-test"

    def test_stride(self, tmp_path: Path) -> None:
        """``stride=2`` reads every second frame."""
        dcd, top_path = _make_n_frame_dcd(tmp_path, n_frames=10)
        traj = read_trajectory(dcd, topology=top_path, stride=2)
        assert traj.n_frames == 5

    def test_times_carried_when_present(self, tmp_path: Path) -> None:
        """If the source format records frame times, they come back
        on Trajectory.times."""
        dcd, top_path = _make_n_frame_dcd(tmp_path, n_frames=5)
        traj = read_trajectory(dcd, topology=top_path)
        assert traj.times is not None
        assert traj.times.shape == (5,)

    def test_metadata_marks_mdtraj_source(self, tmp_path: Path) -> None:
        dcd, top_path = _make_n_frame_dcd(tmp_path, n_frames=2)
        traj = read_trajectory(dcd, topology=top_path)
        assert traj.metadata.get("source") == "mdtraj"


# ---------------------------------------------------------------------
# Streaming with iter_trajectory
# ---------------------------------------------------------------------


class TestIterTrajectory:
    def test_yields_correct_number_of_chunks(self, tmp_path: Path) -> None:
        """A 10-frame trajectory with chunk_size=3 yields 4 chunks
        (3 + 3 + 3 + 1)."""
        dcd, top_path = _make_n_frame_dcd(tmp_path, n_frames=10)
        chunks = list(iter_trajectory(dcd, topology=top_path, chunk_size=3))
        assert len(chunks) == 4
        sizes = [c.n_frames for c in chunks]
        assert sizes == [3, 3, 3, 1]

    def test_total_frames_match_eager_read(self, tmp_path: Path) -> None:
        """Sum of streamed chunks == eager-read frame count."""
        dcd, top_path = _make_n_frame_dcd(tmp_path, n_frames=10)
        total = sum(c.n_frames for c in iter_trajectory(dcd, topology=top_path, chunk_size=4))
        assert total == 10

    def test_each_chunk_is_a_trajectory(self, tmp_path: Path) -> None:
        dcd, top_path = _make_n_frame_dcd(tmp_path, n_frames=5)
        for chunk in iter_trajectory(dcd, topology=top_path, chunk_size=2):
            assert isinstance(chunk, Trajectory)
            assert chunk.n_atoms == 16

    def test_chunk_size_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            list(iter_trajectory(_TRIPEPTIDE, chunk_size=0))

    def test_stride_applies_per_chunk(self, tmp_path: Path) -> None:
        """Stride and chunking compose: stride=2 over 10 frames =
        5 effective frames, in 2 chunks of size 3 (3 + 2)."""
        dcd, top_path = _make_n_frame_dcd(tmp_path, n_frames=10)
        chunks = list(iter_trajectory(dcd, topology=top_path, chunk_size=3, stride=2))
        total = sum(c.n_frames for c in chunks)
        assert total == 5


# ---------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------


class TestWriteTrajectory:
    def test_round_trip_preserves_coordinates(self, tmp_path: Path) -> None:
        dcd, top_path = _make_n_frame_dcd(tmp_path, n_frames=3)
        orig = read_trajectory(dcd, topology=top_path)
        out = tmp_path / "round.dcd"
        write_trajectory(orig, out)
        rt = read_trajectory(out, topology=top_path)
        assert rt.n_frames == orig.n_frames
        assert rt.n_atoms == orig.n_atoms
        np.testing.assert_allclose(rt.coordinates, orig.coordinates, atol=1e-3)

    def test_round_trip_through_xtc(self, tmp_path: Path) -> None:
        """XTC is lossy (int16 × 0.001 nm); round-trip should still
        agree to within ~0.01 Å."""
        dcd, top_path = _make_n_frame_dcd(tmp_path, n_frames=3)
        orig = read_trajectory(dcd, topology=top_path)
        out = tmp_path / "round.xtc"
        write_trajectory(orig, out)
        rt = read_trajectory(out, topology=top_path)
        # XTC precision is 0.001 nm = 0.01 Å.
        np.testing.assert_allclose(rt.coordinates, orig.coordinates, atol=0.02)

    def test_writes_dcd_format(self, tmp_path: Path) -> None:
        dcd, top_path = _make_n_frame_dcd(tmp_path, n_frames=2)
        orig = read_trajectory(dcd, topology=top_path)
        out = tmp_path / "out.dcd"
        write_trajectory(orig, out)
        assert out.is_file()
        assert out.stat().st_size > 0

    def test_writes_xtc_format(self, tmp_path: Path) -> None:
        dcd, top_path = _make_n_frame_dcd(tmp_path, n_frames=2)
        orig = read_trajectory(dcd, topology=top_path)
        out = tmp_path / "out.xtc"
        write_trajectory(orig, out)
        assert out.is_file()
        assert out.stat().st_size > 0

    def test_writes_pdb_format(self, tmp_path: Path) -> None:
        """A 2-frame Trajectory writes to multi-MODEL PDB."""
        dcd, top_path = _make_n_frame_dcd(tmp_path, n_frames=2)
        orig = read_trajectory(dcd, topology=top_path)
        out = tmp_path / "out.pdb"
        write_trajectory(orig, out)
        assert out.is_file()
        # Multi-MODEL PDB contains "MODEL" records.
        text = out.read_text()
        assert text.count("MODEL") >= 2


# ---------------------------------------------------------------------
# Dep handling
# ---------------------------------------------------------------------


class TestMissingMdtraj:
    """When mdtraj is absent the dep helper raises a clean error."""

    def test_missing_mdtraj_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from molforge.io import trajectory as traj_mod
        from molforge.md import MDEngineNotInstalledError

        def fake_require() -> object:
            raise MDEngineNotInstalledError(
                "Trajectory I/O requires mdtraj. Install with:\n    pip install 'molforge[md]'"
            )

        monkeypatch.setattr(traj_mod, "_require_mdtraj", fake_require)

        with pytest.raises(MDEngineNotInstalledError, match="mdtraj"):
            read_trajectory(_TRIPEPTIDE)

        with pytest.raises(MDEngineNotInstalledError, match="mdtraj"):
            list(iter_trajectory(_TRIPEPTIDE))

        # write_trajectory needs a real Trajectory; build one from a
        # cached read first (before monkeypatching).
