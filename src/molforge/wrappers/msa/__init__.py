"""Local homology search — producing MSAs without a public server.

molforge's AF3-class folding wrappers have always sourced alignments from
ColabFold's public MMseqs2 server. This subpackage is the local
alternative: the same search, against a database you control, on a machine
you can inspect.

- :class:`MMseqs2` — fast profile search (Steinegger & Söding 2017),
  returning a :class:`~molforge.sequence.MultipleSequenceAlignment`.

Each engine needs its own pre-built database; molforge deliberately does
not download or build those, since they run to hundreds of gigabytes and
only need making once. The engines' errors name the command that does it.
"""

from __future__ import annotations

from molforge.wrappers.msa.mmseqs2 import (
    MMseqs2,
    MMseqs2NotInstalledError,
    MSADatabaseNotFoundError,
)

__all__ = [
    "MMseqs2",
    "MMseqs2NotInstalledError",
    "MSADatabaseNotFoundError",
]
