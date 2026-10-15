"""MMseqs2 homology search — a local alternative to the public MSA server.

molforge's AF3-class folding wrappers have always sourced their alignments
from ColabFold's public MMseqs2 server. That is fine for a handful of
predictions and increasingly hard to defend at scale: it is a shared
community resource, it rate-limits, it requires network access from
wherever the job runs, and it means the single most influential input to a
prediction is produced somewhere you cannot inspect or reproduce.

`MMseqs2 <https://github.com/soedinglab/MMseqs2>`_ (Steinegger & Söding,
2017) is the same search, run locally against a database you control.

Installation::

    conda install -c conda-forge -c bioconda mmseqs2
    # or: brew install mmseqs2

The database
------------

``search`` needs a **pre-built** MMseqs2 database — a raw FASTA will not
do. Building one is a one-off:

.. code-block:: bash

    mmseqs databases UniRef90 /data/uniref90 /tmp/mmseqs-tmp

Expect hundreds of gigabytes and hours of wall clock. molforge deliberately
does not wrap that step: it is not something anyone should be able to
trigger by accident from a notebook, and it only needs doing once per
machine. Point molforge at the result with ``database=`` or by setting
``$MOLFORGE_MSA_DB``.

What comes back
---------------

A :class:`~molforge.sequence.MultipleSequenceAlignment`, ready to subsample
down a depth ladder or hand to a folding engine. Searches are cached on the
same content-addressed basis as every other molforge engine result, which
matters more here than usual — a deep search is minutes to hours, and the
same query against the same database is the same answer.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from molforge.cache import get_default_cache
from molforge.core.provenance import Provenance
from molforge.sequence import MultipleSequenceAlignment
from molforge.wrappers._versions import engine_version

if TYPE_CHECKING:
    from os import PathLike

__all__ = ["MMseqs2", "MMseqs2NotInstalledError", "MSADatabaseNotFoundError"]

#: Environment variable naming a default MMseqs2 database, so a machine can
#: be configured once instead of every call site repeating a long path.
MSA_DATABASE_ENV = "MOLFORGE_MSA_DB"

#: ``--msa-format-mode 5`` is A3M. Read off MMseqs2's own Parameters.cpp
#: rather than the wiki, which summarises the modes inconsistently; mode 6
#: is A3M *with alignment info*, which is not what a folding engine reads.
_A3M_FORMAT_MODE = "5"

#: Cache type tag for a stored alignment.
_MSA_TYPE_TAG = "msa"


class MMseqs2NotInstalledError(RuntimeError):
    """Raised when the ``mmseqs`` binary isn't on ``$PATH``."""


class MSADatabaseNotFoundError(RuntimeError):
    """Raised when no MMseqs2 database was given, or the path isn't one."""


class MMseqs2:
    """Search a sequence against a local MMseqs2 database.

    Args:
        database: Path to a pre-built MMseqs2 database — the ``outpath``
            given to ``mmseqs databases``, without any suffix. ``None``
            falls back to ``$MOLFORGE_MSA_DB``; with neither, searching
            raises :class:`MSADatabaseNotFoundError` naming the command
            that builds one.
        sensitivity: MMseqs2's ``-s``. Higher finds more remote homologues
            and costs more time; ``None`` leaves MMseqs2's own default.
            7.5 is the usual choice when depth matters more than speed.
        max_seqs: Cap on hits taken through to alignment
            (``--max-seqs``). This is a *search* cap, not the MSA depth a
            model sees — subsample the returned alignment for that.
        threads: ``--threads``. ``None`` lets MMseqs2 choose.
        executable: Name or path of the ``mmseqs`` binary.
        timeout: Seconds before the search is abandoned. ``None`` waits
            indefinitely, which is often right: a sensitive search against
            UniRef90 legitimately takes a long time.

    Example:
        >>> from molforge.wrappers.msa import MMseqs2
        >>> engine = MMseqs2(database="/data/uniref90", sensitivity=7.5)
        >>> msa = engine.search("MKQHKAMIVAL...")        # doctest: +SKIP
        >>> msa.depth                                     # doctest: +SKIP
        2041
        >>> shallow = msa.subsample(32, seed=0)           # doctest: +SKIP
    """

    name = "MMseqs2"
    parallelism = "process"

    def __init__(
        self,
        *,
        database: str | PathLike[str] | None = None,
        sensitivity: float | None = None,
        max_seqs: int | None = None,
        threads: int | None = None,
        executable: str = "mmseqs",
        timeout: float | None = None,
    ) -> None:
        if sensitivity is not None and not 1.0 <= sensitivity <= 10.0:
            raise ValueError(
                f"sensitivity is MMseqs2's -s and runs 1.0-10.0, got {sensitivity}. "
                "Higher finds more remote homologues and costs more time."
            )
        if max_seqs is not None and max_seqs < 1:
            raise ValueError(f"max_seqs must be >= 1, got {max_seqs}")
        self.database = str(database) if database is not None else None
        self.sensitivity = sensitivity
        self.max_seqs = max_seqs
        self.threads = threads
        self.executable = executable
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, sequence: str) -> MultipleSequenceAlignment:
        """Search ``sequence`` and return the alignment.

        Args:
            sequence: A one-letter amino-acid sequence.

        Returns:
            A :class:`~molforge.sequence.MultipleSequenceAlignment` whose
            query is ``sequence`` and whose hits are labelled with the
            database they came from.

        Raises:
            ValueError: If ``sequence`` is empty.
            MMseqs2NotInstalledError: If the binary isn't on ``$PATH``.
            MSADatabaseNotFoundError: If no database is configured.
            RuntimeError: If a search step fails or produces no alignment.
        """
        query = sequence.strip().upper()
        if not query:
            raise ValueError("sequence is empty")

        provenance = self._build_provenance(query)
        cache = get_default_cache()
        cached: MultipleSequenceAlignment | None = cache.get(provenance, _MSA_TYPE_TAG)
        if cached is not None:
            return cached

        database = self._resolve_database()
        binary = self._require_mmseqs()
        a3m = self._run_search(binary, database, query)

        msa = MultipleSequenceAlignment.from_a3m(
            a3m, source_database=Path(database).name or "mmseqs2"
        )
        cache.put(provenance, msa, _MSA_TYPE_TAG)
        return msa

    # ------------------------------------------------------------------
    # Seams
    # ------------------------------------------------------------------

    def _resolve_database(self) -> str:
        """The configured database, or an error naming how to make one."""
        database = self.database or os.environ.get(MSA_DATABASE_ENV)
        if not database:
            raise MSADatabaseNotFoundError(
                "No MMseqs2 database configured. Pass database= or set "
                f"${MSA_DATABASE_ENV}. Build one once with:\n"
                "    mmseqs databases UniRef90 /data/uniref90 /tmp/mmseqs-tmp\n"
                "(hundreds of GB, hours — molforge deliberately doesn't run it "
                "for you.)"
            )
        # An MMseqs2 database is a set of sibling files sharing a prefix;
        # the prefix itself usually exists as a file, and the .dbtype
        # always does. Checking the latter catches the common mistake of
        # pointing at the directory rather than the prefix inside it.
        if not Path(f"{database}.dbtype").is_file() and not Path(database).exists():
            raise MSADatabaseNotFoundError(
                f"{database!r} is not an MMseqs2 database: expected to find "
                f"{database}.dbtype. Point at the database *prefix* — the "
                "outpath you gave `mmseqs databases`, not the directory "
                "containing it."
            )
        return database

    def _require_mmseqs(self) -> str:
        """Resolve the binary, with an install hint rather than a traceback."""
        resolved = shutil.which(self.executable)
        if resolved is None:
            raise MMseqs2NotInstalledError(
                f"mmseqs binary not found: {self.executable!r}. Install with:\n"
                "    conda install -c conda-forge -c bioconda mmseqs2\n"
                "or `brew install mmseqs2`, and ensure it's on $PATH — or pass "
                "an explicit executable= path."
            )
        return resolved

    def _run_search(self, binary: str, database: str, query: str) -> str:
        """Drive the three-step MMseqs2 pipeline, returning a3m text.

        A single seam so tests can exercise everything around it without
        the binary or a 100 GB database. MMseqs2 has no one-shot
        sequence-to-a3m command: the query becomes a database, the search
        produces a result database, and only ``result2msa`` emits an
        alignment.
        """
        with tempfile.TemporaryDirectory(prefix="molforge_mmseqs_") as tmp:
            work = Path(tmp)
            query_fasta = work / "query.fasta"
            query_fasta.write_text(f">query\n{query}\n", encoding="utf-8")

            query_db = work / "queryDB"
            result_db = work / "resultDB"
            msa_out = work / "result.a3m"
            scratch = work / "tmp"

            self._invoke([binary, "createdb", str(query_fasta), str(query_db)], step="createdb")
            self._invoke(
                [
                    binary,
                    "search",
                    str(query_db),
                    database,
                    str(result_db),
                    str(scratch),
                    "-a",
                    *self._search_options(),
                ],
                step="search",
            )
            self._invoke(
                [
                    binary,
                    "result2msa",
                    str(query_db),
                    database,
                    str(result_db),
                    str(msa_out),
                    "--msa-format-mode",
                    _A3M_FORMAT_MODE,
                ],
                step="result2msa",
            )

            if not msa_out.is_file():
                raise RuntimeError(
                    f"MMseqs2 produced no alignment at {msa_out.name}. The search "
                    "completed but result2msa wrote nothing — check that the "
                    "database is a protein database and the query is a protein "
                    "sequence."
                )
            return msa_out.read_text(encoding="utf-8")

    def _search_options(self) -> list[str]:
        """The optional flags, omitted entirely when unset so MMseqs2 keeps
        its own defaults rather than being handed ours."""
        options: list[str] = []
        if self.sensitivity is not None:
            options += ["-s", str(self.sensitivity)]
        if self.max_seqs is not None:
            options += ["--max-seqs", str(self.max_seqs)]
        if self.threads is not None:
            options += ["--threads", str(self.threads)]
        return options

    def _invoke(self, cmd: list[str], *, step: str) -> None:
        """Run one pipeline step, turning failure into a locatable error."""
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout, check=False
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"MMseqs2 {step} timed out after {self.timeout}s. A sensitive "
                "search against a large database legitimately takes a long "
                "time — pass a larger timeout=, or timeout=None to wait."
            ) from e
        if result.returncode != 0:
            raise RuntimeError(
                f"MMseqs2 {step} exited with code {result.returncode}. "
                f"stderr: {result.stderr.strip()}"
            )

    def _build_provenance(self, query: str) -> Provenance:
        """Provenance for a search, built before running so it can be the
        cache key. The database is recorded by path: two databases are two
        different experiments even for an identical query."""
        parameters: dict[str, Any] = {
            "database": self.database or os.environ.get(MSA_DATABASE_ENV),
            "sensitivity": self.sensitivity,
            "max_seqs": self.max_seqs,
        }
        return Provenance.from_engine(
            engine="MMseqs2",
            operation="search_msa",
            engine_version=engine_version("mmseqs2"),
            parameters=parameters,
            inputs={"sequence": query},
        )
