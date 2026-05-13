"""FASTA file format reader and writer.

FASTA is a flat-text format: a header line starting with ``>`` followed
by one or more lines of sequence. Despite its simplicity it has subtle
real-world variations we handle:

- Multi-line sequences (the spec encourages but doesn't require ~80 chars
  per line) — we reassemble.
- Blank lines between records (some tools emit them) — we tolerate them.
- Whitespace and digits embedded in sequences (some old tools add residue
  numbers) — we strip them.
- Comments lines starting with ``;`` (rare, legacy) — we skip them.
- A2M / A3M alignment formats — we don't currently parse these, but the
  reader doesn't choke on them either.

The parser yields :class:`FastaRecord` objects rather than tuples to make
header parsing extensible (description field, accession, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from os import PathLike


@dataclass
class FastaRecord:
    """A single record from a FASTA file.

    Attributes:
        id: The first whitespace-delimited token after ``>`` on the header line.
        description: The rest of the header line, if any.
        sequence: The concatenated, whitespace-stripped sequence.
        metadata: Free-form metadata (e.g. for downstream tools).
    """

    id: str
    sequence: str
    description: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def header(self) -> str:
        """The full header line including ID and description."""
        if self.description:
            return f"{self.id} {self.description}"
        return self.id

    def __len__(self) -> int:
        return len(self.sequence)


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------
def read_fasta(path: str | PathLike[str]) -> list[FastaRecord]:
    """Read a FASTA file from disk.

    Args:
        path: Path to a ``.fasta`` / ``.fa`` / ``.faa`` / ``.fna`` file.
            ``.gz`` suffix triggers gzip decompression.

    Returns:
        A list of :class:`FastaRecord` objects, in file order.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".gz":
        import gzip

        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    return list(read_fasta_string(text))


def read_fasta_string(text: str) -> Iterator[FastaRecord]:
    """Parse FASTA-formatted text, yielding :class:`FastaRecord` objects.

    Memory-efficient: yields records one at a time rather than building a
    list up front.
    """
    current_id: str | None = None
    current_desc = ""
    current_seq_chunks: list[str] = []

    def _emit() -> FastaRecord | None:
        if current_id is None:
            return None
        # Strip whitespace and digits from sequence chunks.
        seq = "".join(current_seq_chunks)
        seq = "".join(c for c in seq if not c.isspace() and not c.isdigit())
        return FastaRecord(id=current_id, description=current_desc, sequence=seq)

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        if line.startswith(";"):
            # Comment line; skip.
            continue
        if line.startswith(">"):
            rec = _emit()
            if rec is not None:
                yield rec
            # Reset for the next record.
            header = line[1:].strip()
            if header:
                parts = header.split(None, 1)
                current_id = parts[0]
                current_desc = parts[1] if len(parts) > 1 else ""
            else:
                current_id = ""
                current_desc = ""
            current_seq_chunks = []
            continue
        # Sequence line
        if current_id is None:
            # Sequence before any header — malformed but tolerable.
            continue
        current_seq_chunks.append(line)

    rec = _emit()
    if rec is not None:
        yield rec


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------
def write_fasta(
    records: Iterable[FastaRecord | tuple[str, str]],
    path: str | PathLike[str],
    *,
    line_width: int = 80,
) -> None:
    """Write FASTA records to disk.

    Args:
        records: Iterable of :class:`FastaRecord` or ``(id, sequence)`` tuples.
        path: Destination path; ``.gz`` triggers gzip.
        line_width: Maximum sequence characters per line. Set to ``0`` to
            emit each sequence on a single line.
    """
    text = write_fasta_string(records, line_width=line_width)
    path = Path(path)
    if path.suffix == ".gz":
        import gzip

        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(text)
    else:
        path.write_text(text, encoding="utf-8")


def write_fasta_string(
    records: Iterable[FastaRecord | tuple[str, str]],
    *,
    line_width: int = 80,
) -> str:
    """Serialize records as FASTA-formatted text."""
    out: list[str] = []
    for rec in records:
        if isinstance(rec, tuple):
            rec_id, sequence = rec
            description = ""
        else:
            rec_id = rec.id
            sequence = rec.sequence
            description = rec.description
        header = rec_id + (f" {description}" if description else "")
        out.append(f">{header}")
        if line_width <= 0:
            out.append(sequence)
        else:
            for i in range(0, len(sequence), line_width):
                out.append(sequence[i : i + line_width])
    return "\n".join(out) + ("\n" if out else "")
