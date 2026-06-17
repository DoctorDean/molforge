"""Top-level load / save / fetch dispatch by file extension.

This module provides the high-level entry points exposed at the package
top level: :func:`load`, :func:`save`, and :func:`fetch`. Each looks at
the file extension (or the ``format`` keyword) and forwards to the
appropriate parser/writer.

Adding a new format means:
1. Implement ``read_<format>`` / ``write_<format>`` in a new module.
2. Add a row to the ``_READERS`` / ``_WRITERS`` tables below.
3. Add a row to the extension map.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from molforge.io.fasta import read_fasta, write_fasta
from molforge.io.mmcif import read_cif, read_cif_string, write_cif
from molforge.io.mol2 import read_mol2, write_mol2
from molforge.io.pdb import read_pdb, read_pdb_string, write_pdb
from molforge.io.pdbqt import read_pdbqt, write_pdbqt
from molforge.io.pqr import read_pqr, write_pqr
from molforge.io.sdf import read_sdf, write_sdf

if TYPE_CHECKING:
    from os import PathLike

    from molforge.core import Protein


# Map file extension -> format key.
_EXT_TO_FORMAT: dict[str, str] = {
    ".pdb": "pdb",
    ".ent": "pdb",
    ".cif": "cif",
    ".mmcif": "cif",
    ".fasta": "fasta",
    ".fa": "fasta",
    ".faa": "fasta",
    ".fna": "fasta",
    ".pdbqt": "pdbqt",
    ".pqr": "pqr",
    ".sdf": "sdf",
    ".mol": "sdf",
    ".mol2": "mol2",
}

# Per-format reader / writer callables. Functions that aren't yet
# implemented are stubbed to raise NotImplementedError with a clear hint.
_READERS: dict[str, Callable[..., object]] = {
    "pdb": read_pdb,
    "cif": read_cif,
    "fasta": read_fasta,
    "sdf": read_sdf,
    "mol2": read_mol2,
    "pdbqt": read_pdbqt,
    "pqr": read_pqr,
}
_WRITERS: dict[str, Callable[..., None]] = {
    "pdb": write_pdb,
    "cif": write_cif,
    "fasta": write_fasta,
    "sdf": write_sdf,
    "mol2": write_mol2,
    "pdbqt": write_pdbqt,
    "pqr": write_pqr,
}

# No planned readers remain — every format the dispatcher knows about
# is now implemented. Keeping the empty dict around as the stable shape
# for any future stubs.
_PLANNED_READERS: dict[str, str] = {}
_PLANNED_WRITERS = dict(_PLANNED_READERS)  # same coverage


def _resolve_format(path: str | PathLike[str], explicit: str | None) -> str:
    if explicit is not None:
        return explicit.lower().lstrip(".")
    suffix = Path(path).suffix.lower()
    # Strip .gz to look at the real extension.
    if suffix == ".gz":
        suffix = Path(str(path)[:-3]).suffix.lower()
    if suffix not in _EXT_TO_FORMAT:
        raise ValueError(
            f"could not infer format from extension {suffix!r}; "
            "pass format='pdb' (or 'fasta', 'cif', ...) explicitly."
        )
    return _EXT_TO_FORMAT[suffix]


def load(
    path: str | PathLike[str],
    *,
    format: str | None = None,
    **kwargs: object,
) -> object:
    """Load a structure or sequence file.

    Format is inferred from the extension unless ``format`` is given.
    Additional kwargs are forwarded to the underlying reader.

    Returns:
        A :class:`molforge.core.Protein` for structure formats, a list of
        :class:`molforge.io.FastaRecord` for FASTA.
    """
    fmt = _resolve_format(path, format)
    reader = _READERS.get(fmt)
    if reader is None:
        hint = _PLANNED_READERS.get(fmt, f"no reader registered for format {fmt!r}")
        raise NotImplementedError(hint)
    return reader(path, **kwargs)


def save(
    obj: object,
    path: str | PathLike[str],
    *,
    format: str | None = None,
    **kwargs: object,
) -> None:
    """Save a structure or list of FASTA records to disk.

    Format is inferred from the extension unless ``format`` is given.
    """
    fmt = _resolve_format(path, format)
    writer = _WRITERS.get(fmt)
    if writer is None:
        hint = _PLANNED_WRITERS.get(fmt, f"no writer registered for format {fmt!r}")
        raise NotImplementedError(hint)
    writer(obj, path, **kwargs)


def fetch(
    pdb_id: str,
    *,
    source: str = "rcsb",
    format: str = "pdb",
    timeout: float = 30.0,
) -> Protein:
    """Fetch a structure by ID from a remote source.

    Downloads the structure over HTTPS and parses it into a
    :class:`~molforge.core.Protein`. Uses only the standard library
    (:mod:`urllib`), so it adds no dependency.

    Args:
        pdb_id: 4-character PDB ID (for ``source="rcsb"``) or UniProt
            accession (for ``source="alphafold"``). Case-insensitive
            for RCSB.
        source: ``"rcsb"`` for the RCSB Protein Data Bank, or
            ``"alphafold"`` for the AlphaFold Protein Structure
            Database.
        format: ``"pdb"`` or ``"cif"``. AlphaFold DB only serves
            ``"pdb"`` and ``"cif"``; both are supported.
        timeout: Network timeout in seconds for the download.

    Returns:
        A :class:`~molforge.core.Protein` parsed from the downloaded
        file.

    Raises:
        ValueError: If ``source`` or ``format`` is unrecognized, or
            ``pdb_id`` is empty.
        OSError: If the download fails — network error, timeout, or a
            non-existent ID (which the server returns as HTTP 404).
            The underlying :class:`urllib.error.URLError` /
            :class:`~urllib.error.HTTPError` is chained as the cause.

    Example:
        >>> from molforge.io import fetch
        >>> protein = fetch("1ABC")                       # RCSB, PDB format
        >>> af = fetch("P00520", source="alphafold")      # AlphaFold DB
    """
    import urllib.error
    import urllib.request

    if not pdb_id or not pdb_id.strip():
        raise ValueError("pdb_id must be a non-empty string")
    pdb_id = pdb_id.strip()

    if source not in ("rcsb", "alphafold"):
        raise ValueError(f"source must be 'rcsb' or 'alphafold', got {source!r}")
    if format not in ("pdb", "cif"):
        raise ValueError(f"format must be 'pdb' or 'cif', got {format!r}")

    if source == "rcsb":
        # RCSB serves both formats from files.rcsb.org. IDs are
        # conventionally uppercase there.
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.{format}"
    else:
        # AlphaFold DB. v4 is the current model version; the filename
        # pattern is AF-<accession>-F1-model_v4.<ext>.
        ext = "cif" if format == "cif" else "pdb"
        url = f"https://alphafold.ebi.ac.uk/files/AF-{pdb_id.upper()}-F1-model_v4.{ext}"

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            text = response.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        raise OSError(
            f"fetch failed: {source} returned HTTP {e.code} for "
            f"{pdb_id!r} ({url}). Check that the ID exists and the "
            "format is available from this source."
        ) from e
    except urllib.error.URLError as e:
        raise OSError(
            f"fetch failed: could not reach {source} at {url} "
            f"({e.reason}). Check your network connection."
        ) from e

    reader = read_cif_string if format == "cif" else read_pdb_string
    return reader(text)
