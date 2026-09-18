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

from molforge.cache import get_default_cache
from molforge.core.provenance import Provenance
from molforge.io.fasta import read_fasta, write_fasta
from molforge.io.mmcif import read_cif, read_cif_string, write_cif
from molforge.io.mol2 import read_mol2, write_mol2
from molforge.io.pdb import read_pdb, read_pdb_string, write_pdb
from molforge.io.pdbqt import read_pdbqt, write_pdbqt
from molforge.io.pqr import read_pqr, write_pqr
from molforge.io.sdf import read_sdf, write_sdf

if TYPE_CHECKING:
    from collections.abc import Iterable
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


#: Cache type tag for a downloaded structure file. The entry holds the
#: server's bytes rather than a parsed Protein — see
#: :func:`molforge.cache._serialize_structure_text` for why.
_DOWNLOAD_TYPE_TAG = "structure_text"


def _fetch_url(pdb_id: str, source: str, format: str) -> str:
    """Build the download URL for a structure ID."""
    if source == "rcsb":
        # RCSB serves both formats from files.rcsb.org. IDs are
        # conventionally uppercase there.
        return f"https://files.rcsb.org/download/{pdb_id.upper()}.{format}"
    # AlphaFold DB. v4 is the current model version; the filename
    # pattern is AF-<accession>-F1-model_v4.<ext>.
    ext = "cif" if format == "cif" else "pdb"
    return f"https://alphafold.ebi.ac.uk/files/AF-{pdb_id.upper()}-F1-model_v4.{ext}"


def _fetch_provenance(pdb_id: str, source: str, format: str) -> Provenance:
    """Provenance describing a download, used as its cache key.

    The ID is upper-cased because both URL builders upper-case it, so
    ``fetch("1ubq")`` and ``fetch("1UBQ")`` are the same request and
    must share one slot. ``timeout`` is deliberately absent: it changes
    how long we are willing to wait, never what comes back.
    """
    return Provenance.from_engine(
        "molforge.io.fetch",
        operation="fetch",
        parameters={"source": source, "format": format},
        inputs={"pdb_id": pdb_id.upper()},
    )


def _download_structure(url: str, *, source: str, pdb_id: str, timeout: float) -> str:
    """GET ``url``, turning urllib's error zoo into a clear OSError."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            text: str = response.read().decode("utf-8")
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
    return text


def fetch(
    pdb_id: str,
    *,
    source: str = "rcsb",
    format: str = "pdb",
    timeout: float = 30.0,
    cache: bool = True,
    force_refresh: bool = False,
) -> Protein:
    """Fetch a structure by ID from a remote source.

    Downloads the structure over HTTPS and parses it into a
    :class:`~molforge.core.Protein`. Uses only the standard library
    (:mod:`urllib`), so it adds no dependency.

    Downloads are cached. A pipeline that resolves the same entries at
    several stages — prepare, predict, score — pays the network cost
    once, and a long fetch loop that dies halfway through resumes
    almost instantly. What is stored is the **downloaded file itself**,
    not the parsed Protein, so a cache hit re-parses exactly the bytes
    a cache miss would have parsed and the two are indistinguishable.

    The cache is :mod:`molforge.cache`, so it honours the same
    environment variables as every other molforge artifact:
    ``MOLFORGE_CACHE_DIR`` moves it, ``MOLFORGE_CACHE=disabled`` turns
    it off globally. Entries are keyed on ``(pdb_id, source, format)``
    and, like all molforge cache entries, on the molforge major.minor
    version — an upgrade re-downloads once.

    Args:
        pdb_id: 4-character PDB ID (for ``source="rcsb"``) or UniProt
            accession (for ``source="alphafold"``). Case-insensitive
            for RCSB.
        source: ``"rcsb"`` for the RCSB Protein Data Bank, or
            ``"alphafold"`` for the AlphaFold Protein Structure
            Database.
        format: ``"pdb"`` or ``"cif"``. AlphaFold DB only serves
            ``"pdb"`` and ``"cif"``; both are supported.
        timeout: Network timeout in seconds for the download. Not part
            of the cache key.
        cache: Consult and populate the download cache. ``False``
            bypasses it in both directions — nothing is read, nothing
            is written — which is what you want for a one-off fetch you
            don't want to leave on disk.
        force_refresh: Ignore any cached copy, re-download, and replace
            the entry. Use it when the remote entry has been revised.
            With ``cache=False`` there is nothing to refresh, and the
            download happens anyway.

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
            A cache hit cannot raise this, having made no request.

    Example:
        >>> from molforge.io import fetch
        >>> protein = fetch("1ABC")                       # RCSB, PDB format
        >>> again = fetch("1ABC")                         # served from cache
        >>> af = fetch("P00520", source="alphafold")      # AlphaFold DB
        >>> fresh = fetch("1ABC", force_refresh=True)     # re-download
    """
    if not pdb_id or not pdb_id.strip():
        raise ValueError("pdb_id must be a non-empty string")
    pdb_id = pdb_id.strip()

    if source not in ("rcsb", "alphafold"):
        raise ValueError(f"source must be 'rcsb' or 'alphafold', got {source!r}")
    if format not in ("pdb", "cif"):
        raise ValueError(f"format must be 'pdb' or 'cif', got {format!r}")

    reader = read_cif_string if format == "cif" else read_pdb_string
    store = get_default_cache() if cache else None
    provenance = _fetch_provenance(pdb_id, source, format) if store is not None else None

    if store is not None and provenance is not None:
        if force_refresh:
            # put() will not overwrite an existing entry, so the stale
            # one has to go before the fresh download can take its slot.
            store.invalidate(provenance)
        else:
            cached = store.get(provenance, _DOWNLOAD_TYPE_TAG)
            if cached is not None:
                return reader(cached)

    text = _download_structure(
        _fetch_url(pdb_id, source, format), source=source, pdb_id=pdb_id, timeout=timeout
    )

    if store is not None and provenance is not None:
        store.put(provenance, text, _DOWNLOAD_TYPE_TAG)

    return reader(text)


def fetch_many(
    pdb_ids: Iterable[str],
    *,
    source: str = "rcsb",
    format: str = "pdb",
    timeout: float = 30.0,
    on_error: str = "raise",
    cache: bool = True,
    force_refresh: bool = False,
) -> list[Protein]:
    """Fetch several structures by ID, one :func:`fetch` per ID.

    A thin convenience over :func:`fetch` for pulling a whole set — for
    example the hits from :func:`search_rcsb`. Downloads are sequential (the
    servers rate-limit, and this keeps the dependency to the standard library),
    and cached, so the sequential cost is paid once per ID rather than once
    per call.

    Args:
        pdb_ids: The IDs to fetch, in the order you want them back.
        source: ``"rcsb"`` or ``"alphafold"`` (applied to every ID).
        format: ``"pdb"`` or ``"cif"`` (applied to every ID).
        timeout: Per-download network timeout in seconds.
        on_error: ``"raise"`` (default) stops at the first ID that fails;
            ``"skip"`` drops IDs that fail — e.g. a 404 for a non-existent
            entry — and returns the structures that did download.
        cache: Forwarded to :func:`fetch`. Left on, a re-run of the same
            ID set costs no downloads at all, and a loop killed partway
            through resumes from where it stopped.
        force_refresh: Forwarded to :func:`fetch`.

    Returns:
        The fetched proteins, in input order (minus any dropped under
        ``on_error="skip"``).

    Raises:
        ValueError: If ``on_error`` is not ``"raise"`` or ``"skip"``.
        OSError: On a download failure when ``on_error="raise"``.
    """
    if on_error not in ("raise", "skip"):
        raise ValueError(f"on_error must be 'raise' or 'skip', got {on_error!r}")

    proteins: list[Protein] = []
    for pdb_id in pdb_ids:
        try:
            proteins.append(
                fetch(
                    pdb_id,
                    source=source,
                    format=format,
                    timeout=timeout,
                    cache=cache,
                    force_refresh=force_refresh,
                )
            )
        except OSError:
            if on_error == "raise":
                raise
    return proteins
