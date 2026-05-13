"""PQR (PDB2PQR / APBS) format reader and writer.

PQR (PDB2PQR / APBS) is PDB-like format with explicit per-atom charges and radii used in electrostatics calculations.

**Status: stub.** The API surface is committed; the implementation is
planned. Calling :func:`read_pqr` or :func:`write_pqr` currently
raises :class:`NotImplementedError` with a pointer to the relevant issue.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from os import PathLike

    from molforge.core import Protein


def read_pqr(path: str | PathLike[str], **kwargs: object) -> Protein:
    """Read a PQR file."""
    raise NotImplementedError(
        "PQR read support is planned; track progress at "
        "https://github.com/DoctorDean/molforge/issues."
    )


def write_pqr(protein: Protein, path: str | PathLike[str], **kwargs: object) -> None:
    """Write a Protein to a PQR file."""
    raise NotImplementedError(
        "PQR write support is planned; track progress at "
        "https://github.com/DoctorDean/molforge/issues."
    )
