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
from molforge.io.mmcif import read_cif, write_cif
from molforge.io.pdb import read_pdb, write_pdb

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
}
_WRITERS: dict[str, Callable[..., None]] = {
    "pdb": write_pdb,
    "cif": write_cif,
    "fasta": write_fasta,
}

_PLANNED_READERS: dict[str, str] = {
    "pdbqt": "PDBQT reader is planned; see molforge.io.pdbqt",
    "pqr": "PQR reader is planned; see molforge.io.pqr",
    "sdf": "SDF reader is planned; see molforge.io.sdf (will require RDKit)",
    "mol2": "MOL2 reader is planned; see molforge.io.mol2 (will require RDKit)",
}
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
    writer(obj, path, **kwargs)  # type: ignore[arg-type]


def fetch(
    pdb_id: str,
    *,
    source: str = "rcsb",
    format: str = "pdb",
) -> Protein:
    """Fetch a structure by ID from a remote source.

    Args:
        pdb_id: 4-character PDB ID (RCSB) or UniProt accession (AlphaFold DB).
        source: ``"rcsb"`` for the PDB or ``"alphafold"`` for AlphaFold DB.
        format: ``"pdb"`` or ``"cif"``.

    Returns:
        A :class:`molforge.core.Protein`.

    Notes:
        This function requires network access. The implementation is
        stubbed pending the addition of an HTTP utility; for now we
        raise :class:`NotImplementedError` rather than pull in
        ``requests`` as a core dependency.
    """
    # TODO: implement with stdlib urllib (avoid requests dep). The
    # endpoints are:
    #   https://files.rcsb.org/download/{pdb_id}.pdb
    #   https://files.rcsb.org/download/{pdb_id}.cif
    #   https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_v4.pdb
    raise NotImplementedError(
        f"fetch({pdb_id!r}, source={source!r}, format={format!r}) is planned; "
        "for now, download manually and use load()."
    )
