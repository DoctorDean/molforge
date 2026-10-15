"""Multiple sequence alignments — the input an AF3-class model really runs on.

:class:`~molforge.sequence.Alignment` describes two sequences lined up
against each other. A folding model wants something else entirely: hundreds
or thousands of homologues lined up against *one* query, because the
covariation between columns is where the structural signal lives. Until now
molforge had no way to represent that, which meant no way to hand a model an
alignment you had computed yourself — the folding wrappers could only ask a
public server for one.

The in-memory shape
-------------------

:class:`MultipleSequenceAlignment` is a query plus its hits, each hit an
:class:`AlignedSequence` carrying more than its letters:

- ``source_database`` — which search produced it (``uniref90``,
  ``mgnify``, ...). Chai-1 requires it; it is also the honest answer to
  "where did this homologue come from".
- ``pairing_key`` — the token used to pair hits *across chains* of a
  complex, conventionally the source organism. Two chains' hits that share
  a pairing key came from the same organism and are presumed to interact,
  which is the entire basis of complex prediction from coevolution.

Carrying those two fields is what separates an MSA that can fold a monomer
from one that can fold a dimer.

The on-disk shape
-----------------

a3m is the common currency: Boltz reads it directly, and Chai-1 ships a
converter from it. In a3m, **upper-case letters and ``-`` are alignment
columns; lower-case letters are insertions** relative to the query and
occupy no column. That rule is what makes a3m narrower than the equivalent
aligned FASTA, and it is enforced here — every sequence must have the same
number of alignment columns as the query, or the file is malformed and the
error says so rather than letting a mis-shaped alignment reach a model.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator
    from os import PathLike

__all__ = [
    "AlignedSequence",
    "MultipleSequenceAlignment",
    "read_a3m",
    "write_a3m",
]

#: Source-database labels Chai-1 accepts, plus the query's own marker.
#: Anything else is passed through — molforge does not police a vocabulary
#: it doesn't own — but these are the ones that round-trip cleanly.
KNOWN_SOURCE_DATABASES = frozenset(
    {"query", "uniprot", "uniref90", "bfd_uniclust", "mgnify", "unknown"}
)

#: In a3m, lower-case letters are insertions relative to the query and are
#: not alignment columns. Everything else (upper-case, ``-``, ``.``) is.
_INSERTION = re.compile(r"[a-z.]")

#: ``OX=`` / ``TaxID=`` in a header is the conventional pairing token: hits
#: from the same organism are the ones worth pairing across chains.
_TAXON = re.compile(r"\b(?:OX|TaxID|taxid)[=:](\w+)")


@dataclass(frozen=True)
class AlignedSequence:
    """One row of an alignment: a homologue, lined up against the query.

    Attributes:
        sequence: The aligned sequence in a3m convention — upper-case and
            ``-`` are alignment columns, lower-case are insertions.
        identifier: The accession or header text, without the ``>``.
        source_database: Which database the hit came from
            (:data:`KNOWN_SOURCE_DATABASES` lists the ones that round-trip
            cleanly). ``""`` when unrecorded.
        pairing_key: Token used to pair this hit against another chain's
            hits — conventionally the source organism. ``""`` when the hit
            can't be paired, which is the honest default: an unpairable hit
            is still a perfectly good monomer hit.
        comment: Free text, carried for readability and ignored otherwise.
    """

    sequence: str
    identifier: str = ""
    source_database: str = ""
    pairing_key: str = ""
    comment: str = ""

    @property
    def columns(self) -> str:
        """The alignment columns only, with insertions removed.

        This is the row as the model sees it: every :class:`AlignedSequence`
        in an alignment has the same ``columns`` length.
        """
        return _INSERTION.sub("", self.sequence)

    @property
    def ungapped(self) -> str:
        """The plain sequence — no gaps, no insertion casing."""
        return self.sequence.replace("-", "").replace(".", "").upper()

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe row, shaped like Chai-1's ``.aligned.pqt`` columns."""
        return {
            "sequence": self.sequence,
            "identifier": self.identifier,
            "source_database": self.source_database,
            "pairing_key": self.pairing_key,
            "comment": self.comment,
        }


@dataclass(frozen=True)
class MultipleSequenceAlignment:
    """A query sequence and the homologues aligned to it.

    Attributes:
        query: The sequence being folded. Always row 0 of the a3m, and the
            row every other is aligned against.
        hits: The homologues, in search order (best first, by convention).

    Example:
        >>> from molforge.sequence import MultipleSequenceAlignment
        >>> msa = MultipleSequenceAlignment.from_a3m(
        ...     ">query\\nMKQ\\n>hit1 OX=9606\\nMKA\\n"
        ... )
        >>> msa.depth
        2
        >>> msa.n_columns
        3
        >>> msa.hits[0].pairing_key
        '9606'
    """

    query: AlignedSequence
    hits: tuple[AlignedSequence, ...] = ()

    def __post_init__(self) -> None:
        if not self.query.sequence:
            raise ValueError("a multiple sequence alignment needs a non-empty query")
        width = len(self.query.columns)
        for i, hit in enumerate(self.hits):
            if len(hit.columns) != width:
                raise ValueError(
                    f"alignment row {i + 1} ({hit.identifier or '<unnamed>'!s}) has "
                    f"{len(hit.columns)} alignment columns but the query has {width}. "
                    "In a3m, lower-case letters are insertions and don't occupy a "
                    "column — a mismatch here means the file is malformed, not that "
                    "the hit is shorter."
                )

    # ------------------------------------------------------------------
    # Shape
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Depth — the query plus its hits."""
        return 1 + len(self.hits)

    def __iter__(self) -> Iterator[AlignedSequence]:
        """Iterate every row, query first."""
        yield self.query
        yield from self.hits

    @property
    def depth(self) -> int:
        """Number of sequences, including the query.

        This is the quantity :attr:`Boltz.msa_depth` and Chai-1's
        ``recycle_msa_subsample`` cap.
        """
        return len(self)

    @property
    def n_columns(self) -> int:
        """Alignment width, insertions excluded."""
        return len(self.query.columns)

    @property
    def query_sequence(self) -> str:
        """The query with gaps and insertion casing removed."""
        return self.query.ungapped

    # ------------------------------------------------------------------
    # Derivation
    # ------------------------------------------------------------------

    def subsample(self, depth: int, *, seed: int | None = None) -> MultipleSequenceAlignment:
        """Return an alignment of at most ``depth`` sequences.

        The query is always kept — an alignment without it has nothing to
        be aligned *to* — so ``depth=8`` means the query plus seven hits.

        Hits are sampled rather than truncated when ``seed`` is given,
        because the top of a search output is usually the most redundant
        part of it; truncation gives a shallow alignment of near-identical
        sequences, which is not the same experiment as a shallow alignment
        of diverse ones. Without a seed the first ``depth - 1`` hits are
        kept, which is deterministic and matches what the engines do with
        their own depth caps.

        Args:
            depth: Total sequences to keep, query included. Must be >= 1.
            seed: Seed for random sampling. ``None`` truncates instead.

        Returns:
            A new alignment; this one is unchanged.

        Raises:
            ValueError: If ``depth`` is below 1.
        """
        if depth < 1:
            raise ValueError(f"depth must be >= 1 (the query alone), got {depth}")
        keep = depth - 1
        if keep >= len(self.hits):
            return self
        if seed is None:
            return replace(self, hits=self.hits[:keep])
        rng = random.Random(seed)
        return replace(self, hits=tuple(rng.sample(self.hits, keep)))

    def deduplicate(self) -> MultipleSequenceAlignment:
        """Drop hits whose alignment columns repeat an earlier row.

        Search outputs routinely contain the same alignment several times
        over; depth that is really redundancy inflates every depth-based
        decision made downstream.
        """
        seen = {self.query.columns}
        unique: list[AlignedSequence] = []
        for hit in self.hits:
            if hit.columns in seen:
                continue
            seen.add(hit.columns)
            unique.append(hit)
        return replace(self, hits=tuple(unique))

    def with_source_database(self, name: str) -> MultipleSequenceAlignment:
        """Label every hit with the database it came from.

        Search wrappers know this; the a3m they parse usually doesn't say.
        """
        return replace(self, hits=tuple(replace(h, source_database=name) for h in self.hits))

    def pairable(self) -> tuple[AlignedSequence, ...]:
        """Hits carrying a pairing key, i.e. those usable for a complex."""
        return tuple(h for h in self.hits if h.pairing_key)

    # ------------------------------------------------------------------
    # a3m
    # ------------------------------------------------------------------

    @classmethod
    def from_a3m(cls, text: str, *, source_database: str = "") -> MultipleSequenceAlignment:
        """Parse a3m text. The first record is the query.

        Args:
            text: The a3m document.
            source_database: Label applied to every hit, since a3m has
                nowhere to record it.

        Returns:
            The parsed alignment.

        Raises:
            ValueError: If the text has no records, or a row's column count
                disagrees with the query's.
        """
        records = _parse_fasta_like(text)
        if not records:
            raise ValueError("a3m text contains no sequences")
        header, sequence = records[0]
        query = AlignedSequence(
            sequence=sequence,
            identifier=header,
            source_database="query",
            pairing_key=_pairing_key_from_header(header),
        )
        hits = tuple(
            AlignedSequence(
                sequence=seq,
                identifier=head,
                source_database=source_database,
                pairing_key=_pairing_key_from_header(head),
            )
            for head, seq in records[1:]
        )
        return cls(query=query, hits=hits)

    def to_a3m(self) -> str:
        """Serialise to a3m text, query first.

        The format Boltz reads directly and Chai-1 converts from, so this
        is the handover point to both.
        """
        lines: list[str] = []
        for row in self:
            lines.append(f">{row.identifier}" if row.identifier else ">")
            lines.append(row.sequence)
        return "\n".join(lines) + "\n"

    def to_pairing_rows(self) -> list[dict[str, Any]]:
        """Rows shaped for Chai-1's ``.aligned.pqt`` / Boltz's paired CSV.

        The query is included and marked ``source_database="query"``, which
        is what Chai-1 expects to find.
        """
        return [row.to_dict() for row in self]


def read_a3m(path: str | PathLike[str], *, source_database: str = "") -> MultipleSequenceAlignment:
    """Read an a3m file.

    Args:
        path: The ``.a3m`` file.
        source_database: Label applied to every hit.

    Returns:
        The parsed alignment.
    """
    from pathlib import Path

    return MultipleSequenceAlignment.from_a3m(
        Path(path).read_text(encoding="utf-8"), source_database=source_database
    )


def write_a3m(msa: MultipleSequenceAlignment, path: str | PathLike[str]) -> None:
    """Write an alignment as a3m — the format both AF3-class engines accept."""
    from pathlib import Path

    Path(path).write_text(msa.to_a3m(), encoding="utf-8")


# ---------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------


def _parse_fasta_like(text: str) -> list[tuple[str, str]]:
    """Split FASTA/a3m text into ``(header, sequence)`` pairs.

    Deliberately not reusing :func:`molforge.io.read_fasta_string`: that
    returns :class:`FastaRecord` and normalises sequences, and a3m's
    lower-case insertions must survive verbatim — normalising them away is
    exactly the corruption this format exists to avoid.
    """
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks)))
            header = line[1:].strip()
            chunks = []
        elif header is not None:
            chunks.append(line)
    if header is not None:
        records.append((header, "".join(chunks)))
    return [(h, s) for h, s in records if s]


def _pairing_key_from_header(header: str) -> str:
    """Pull the organism token out of a header, or return ``""``.

    UniProt-style headers carry ``OX=9606``; several search tools emit
    ``TaxID=``. Anything else yields no key, which is correct — a hit we
    can't attribute to an organism is one we can't responsibly pair.
    """
    match = _TAXON.search(header)
    return match.group(1) if match else ""
