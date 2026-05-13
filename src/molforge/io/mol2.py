"""MOL2 (Tripos) format reader and writer.

MOL2 (Tripos) is a small-molecule exchange format with bond orders and Tripos atom types.

**Status: stub.** The API surface is committed; the implementation is
planned. Calling :func:`read_mol2` or :func:`write_mol2` currently
raises :class:`NotImplementedError` with a pointer to the relevant issue.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from os import PathLike

    from molforge.core import Protein


def read_mol2(path: str | PathLike[str], **kwargs: object) -> Protein:
    """Read a MOL2 file."""
    raise NotImplementedError(
        "MOL2 read support is planned; track progress at "
        "https://github.com/DoctorDean/molforge/issues."
    )


def write_mol2(protein: Protein, path: str | PathLike[str], **kwargs: object) -> None:
    """Write a Protein to a MOL2 file."""
    raise NotImplementedError(
        "MOL2 write support is planned; track progress at "
        "https://github.com/DoctorDean/molforge/issues."
    )
