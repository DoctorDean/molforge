"""Trajectory I/O — reading and writing MD coordinate files.

Real molecular-dynamics trajectories come in a handful of binary
formats that are not human-readable but compress well and load fast:

- **``.xtc``** — GROMACS portable lossy format. Coordinates are
  stored as int16 × 0.001 nm, so positions round to the nearest
  0.0001 nm (0.01 Å). Far and away the most common modern format.
- **``.trr``** — GROMACS lossless format. Holds velocities and
  forces in addition to coordinates.
- **``.dcd``** — CHARMM / NAMD / OpenMM lossless coordinates.
- **``.nc`` / ``.netcdf``** — AMBER NetCDF, binary, lossless.
- **``.h5`` / ``.h5md``** — HDF5-based, the modern lossless option.
- Multi-MODEL PDB — text, slow, huge files. Use for tiny trajectories
  only.

This module wraps `mdtraj <https://www.mdtraj.org/>`_ to read and
write all of the above. mdtraj is a hard dep of the ``[md]`` extra
(``pip install 'molforge[md]'``); the imports are lazy, so importing
:mod:`molforge.io` itself does not require mdtraj.

Three public functions:

- :func:`read_trajectory` loads a whole file into a
  :class:`molforge.md.Trajectory`. Convenient when the trajectory
  fits in memory.
- :func:`iter_trajectory` yields chunks of frames as
  :class:`molforge.md.Trajectory` objects. Use this for trajectories
  larger than RAM — memory is bounded by the chunk size, and the
  caller streams through the file frame-batch by frame-batch.
- :func:`write_trajectory` saves a :class:`Trajectory` to disk in
  whichever format the path's extension implies.

All three accept a ``stride`` parameter (load every Nth frame — useful
when a tight time resolution isn't needed) and ``atom_indices``
(load only some atoms — useful when an analysis only cares about
e.g. backbone Calpha carbons).

Topology handling: most binary trajectory formats do **not** store
the topology — they're coordinates and timestamps only. The reader
takes the topology as an explicit ``topology`` argument: pass a
:class:`molforge.core.Protein` (typical) or a path to a PDB file
(also accepted). PDB-format trajectories (multi-MODEL) and HDF5
formats can carry topology and so accept ``topology=None``.

Units: molforge works in Ångström and picoseconds throughout.
mdtraj works in nanometers and picoseconds. The conversion (× 10 on
read, ÷ 10 on write) is handled here so callers never see nm.

Unsupported by design (use mdtraj directly):
- Trajectory analysis (RMSD-over-time, contact maps, clustering).
- Velocity / force I/O. Only coordinates are preserved through the
  molforge Trajectory; if you need velocities, drop down to mdtraj
  via the topology and original file.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator
    from os import PathLike

    from molforge.core import Protein
    from molforge.md import Trajectory


__all__ = ["iter_trajectory", "read_trajectory", "write_trajectory"]


# ----------------------------------------------------------------------
# Dep helper
# ----------------------------------------------------------------------


def _require_mdtraj() -> Any:
    """Import mdtraj or raise a clean MDEngineNotInstalledError.

    mdtraj ships with the ``[md]`` extra; users who installed
    ``molforge[md]`` for OpenMM also get mdtraj.
    """
    try:
        import mdtraj
    except ImportError as e:
        from molforge.md import MDEngineNotInstalledError

        raise MDEngineNotInstalledError(
            "Trajectory I/O requires mdtraj. Install with:\n"
            "    pip install 'molforge[md]'\n"
            "or directly:\n"
            "    pip install mdtraj\n"
            f"Underlying error: {e}"
        ) from e
    return mdtraj


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def read_trajectory(
    path: str | PathLike[str],
    *,
    topology: Protein | str | PathLike[str] | None = None,
    stride: int = 1,
    atom_indices: list[int] | np.ndarray | None = None,
    fmt: str | None = None,
) -> Trajectory:
    """Read a trajectory file into a :class:`molforge.md.Trajectory`.

    Args:
        path: Path to the trajectory. Format inferred from the
            extension unless ``fmt`` is given. Supported: ``.xtc``,
            ``.trr``, ``.dcd``, ``.nc``, ``.netcdf``, ``.h5``,
            ``.h5md``, ``.pdb``.
        topology: The topology to attach. Required for formats that
            don't embed topology (``.xtc``, ``.trr``, ``.dcd``,
            ``.nc``). Accepts a :class:`molforge.core.Protein` or a
            path to a PDB. May be ``None`` for ``.pdb`` and ``.h5``
            files, which carry their own topology.
        stride: Read every ``stride``-th frame (default 1, all
            frames). Useful when full time resolution isn't needed.
        atom_indices: Read only these atom indices (0-based). Useful
            when an analysis only touches a subset (e.g. backbone
            atoms). When given, the resulting :class:`Trajectory`'s
            topology is sliced to match.
        fmt: Override the format inference. Passed to mdtraj as the
            file extension (without the leading dot).

    Returns:
        A :class:`molforge.md.Trajectory` with coordinates in Å, times
        in picoseconds when available, and the topology as a
        :class:`molforge.core.Protein`. The whole file is loaded into
        memory — use :func:`iter_trajectory` for files too large to
        fit.

    Raises:
        MDEngineNotInstalledError: If mdtraj is not installed.
        ValueError: If ``topology`` is missing for a format that
            requires it, or if the atom count in the trajectory
            disagrees with the topology.

    Example:
        >>> from molforge.io import read_pdb, read_trajectory
        >>> topology = read_pdb("system.pdb")
        >>> traj = read_trajectory("md.xtc", topology=topology)
        >>> traj.n_frames
        1000
        >>> traj.coordinates.shape  # (n_frames, n_atoms, 3) in Å
        (1000, 24512, 3)
    """
    mdtraj = _require_mdtraj()
    top = _resolve_topology_for_mdtraj(topology, mdtraj)
    indices = _normalize_indices(atom_indices)

    md_traj = mdtraj.load(
        str(path),
        top=top,
        stride=int(stride),
        atom_indices=indices,
    )
    return _mdtraj_to_molforge(md_traj, atom_indices=indices, source_topology=topology)


def iter_trajectory(
    path: str | PathLike[str],
    *,
    topology: Protein | str | PathLike[str] | None = None,
    chunk_size: int = 100,
    stride: int = 1,
    atom_indices: list[int] | np.ndarray | None = None,
    fmt: str | None = None,
) -> Iterator[Trajectory]:
    """Stream a trajectory in chunks of frames.

    Use this for trajectories larger than RAM. Each yielded object is
    a :class:`molforge.md.Trajectory` holding ``chunk_size`` frames
    (the last chunk may be shorter); memory usage is bounded by
    chunk_size × n_atoms × 12 bytes.

    Args:
        path: Trajectory file.
        topology: As for :func:`read_trajectory`.
        chunk_size: Number of frames per yielded Trajectory. Default
            100; balance memory vs iteration overhead.
        stride: Read every ``stride``-th frame.
        atom_indices: Read only these atom indices.

    Yields:
        A :class:`molforge.md.Trajectory` per chunk, in file order.

    Raises:
        MDEngineNotInstalledError: If mdtraj is not installed.
        ValueError: If ``topology`` is missing for a format that
            requires it.

    Example:
        >>> for chunk in iter_trajectory("big.xtc", topology=top, chunk_size=500):
        ...     # process chunk.coordinates here; memory bounded
        ...     pass
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    mdtraj = _require_mdtraj()
    top = _resolve_topology_for_mdtraj(topology, mdtraj)
    indices = _normalize_indices(atom_indices)

    for md_chunk in mdtraj.iterload(
        str(path),
        top=top,
        chunk=int(chunk_size),
        stride=int(stride),
        atom_indices=indices,
    ):
        yield _mdtraj_to_molforge(
            md_chunk,
            atom_indices=indices,
            source_topology=topology,
        )


def write_trajectory(
    trajectory: Trajectory,
    path: str | PathLike[str],
    *,
    fmt: str | None = None,
) -> None:
    """Write a :class:`Trajectory` to disk.

    The format is inferred from the path's extension. Coordinates
    are converted from Å (molforge convention) to nm (mdtraj's
    convention) on the way out.

    Args:
        trajectory: The :class:`molforge.md.Trajectory` to write.
        path: Output path. Format inferred from the extension.

    Raises:
        MDEngineNotInstalledError: If mdtraj is not installed.

    Example:
        >>> from molforge.io import write_trajectory
        >>> write_trajectory(traj, "out.xtc")
    """
    mdtraj = _require_mdtraj()
    md_traj = _molforge_to_mdtraj(trajectory, mdtraj)
    md_traj.save(str(path))


# ----------------------------------------------------------------------
# Internals: molforge <-> mdtraj conversion
# ----------------------------------------------------------------------


def _resolve_topology_for_mdtraj(
    topology: Protein | str | PathLike[str] | None,
    mdtraj: Any,
) -> Any:
    """Convert a topology argument into something mdtraj accepts.

    mdtraj's ``top=`` accepts either a path or one of its own
    ``Topology`` objects. We materialize a :class:`Protein` to a temp
    PDB so mdtraj can build the topology from it; a path is forwarded
    unchanged.
    """
    if topology is None:
        return None
    # A Protein needs to be materialized; anything else is a path-like
    # we forward to mdtraj as a string.
    from molforge.core import Protein

    if not isinstance(topology, Protein):
        return str(topology)
    # It's a Protein. Write to a temp PDB so mdtraj can read it.
    import tempfile

    from molforge.io import write_pdb

    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as fh:
        tmp = Path(fh.name)
    try:
        write_pdb(topology, tmp)
        top = mdtraj.load(str(tmp)).topology
    finally:
        tmp.unlink(missing_ok=True)
    return top


def _normalize_indices(
    atom_indices: list[int] | np.ndarray | None,
) -> np.ndarray | None:
    """Coerce a list/array of atom indices to int32 ndarray (mdtraj's
    expected type), or pass ``None`` through."""
    if atom_indices is None:
        return None
    return np.asarray(atom_indices, dtype=np.int32)


def _mdtraj_to_molforge(
    md_traj: Any,
    *,
    atom_indices: np.ndarray | None,
    source_topology: Protein | str | PathLike[str] | None,
) -> Trajectory:
    """Convert an mdtraj.Trajectory into a molforge Trajectory.

    Coordinate units are converted nm → Å. The molforge topology
    :class:`Protein` is reconstructed from the mdtraj topology by
    writing the first frame to PDB and parsing it back. (mdtraj's
    topology object is rich, but the simplest path to a
    :class:`molforge.core.Protein` is round-tripping through PDB —
    which keeps coordinates, residue ids, chain ids, atom names, and
    elements, which is what molforge needs.)
    """
    import tempfile

    from molforge.core import Protein
    from molforge.io import read_pdb
    from molforge.md import Trajectory

    topology_protein: Protein
    if isinstance(source_topology, Protein):
        # Caller gave us a Protein. Slice the AtomArray if a subset
        # was requested.
        if atom_indices is not None:
            sliced_arr = source_topology.atom_array[atom_indices]
            topology_protein = Protein(sliced_arr)
            if source_topology.metadata:
                topology_protein.metadata = {**source_topology.metadata}
        else:
            topology_protein = source_topology
    else:
        # Build a topology Protein from the mdtraj topology by
        # round-tripping through PDB. md_traj[0] is a one-frame slice.
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as fh:
            tmp = Path(fh.name)
        try:
            md_traj[0].save(str(tmp))
            topology_protein = read_pdb(tmp)
        finally:
            tmp.unlink(missing_ok=True)

    # mdtraj coords are nm; molforge wants Å.
    coords_nm = np.asarray(md_traj.xyz, dtype=np.float32)
    coords_ang = coords_nm * 10.0

    # Times are picoseconds in both worlds.
    times = np.asarray(md_traj.time, dtype=np.float64) if md_traj.time is not None else None

    metadata: dict[str, object] = {
        "source": "mdtraj",
        "mdtraj_version": _try_attr(md_traj, "__module__", "mdtraj"),
    }
    return Trajectory(
        topology=topology_protein,
        coordinates=coords_ang,
        times=times,
        metadata=metadata,
    )


def _molforge_to_mdtraj(trajectory: Trajectory, mdtraj: Any) -> Any:
    """Convert a molforge :class:`Trajectory` to an
    :class:`mdtraj.Trajectory`.

    Coordinates are converted Å → nm. The topology is round-tripped
    through PDB.
    """
    import tempfile

    from molforge.io import write_pdb

    # Build the mdtraj topology from our Protein. Round-trip through
    # PDB — same trick used in _resolve_topology_for_mdtraj.
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as fh:
        tmp = Path(fh.name)
    try:
        write_pdb(trajectory.topology, tmp)
        topology = mdtraj.load(str(tmp)).topology
    finally:
        tmp.unlink(missing_ok=True)

    coords_nm = np.asarray(trajectory.coordinates, dtype=np.float32) / 10.0
    time = np.asarray(trajectory.times, dtype=np.float32) if trajectory.times is not None else None
    return mdtraj.Trajectory(xyz=coords_nm, topology=topology, time=time)


def _try_attr(obj: object, *names: str) -> str:
    """Return the first attribute lookup that succeeds, as a string."""
    for name in names:
        val = getattr(obj, name, None)
        if val is not None:
            return str(val)
    return ""
