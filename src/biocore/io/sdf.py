"""SDF (Structure Data File) format reader and writer.

SDF (Structure Data File) is the standard small-molecule exchange format, supporting multi-molecule files with property fields.

**Status: stub.** The API surface is committed; the implementation is
planned. Calling :func:`read_sdf` or :func:`write_sdf` currently
raises :class:`NotImplementedError` with a pointer to the relevant issue.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from os import PathLike

    from molforge.core import Protein


def read_sdf(path: str | PathLike[str], **kwargs: object) -> Protein:
    """Read a SDF file."""
    raise NotImplementedError(
        "SDF read support is planned; track progress at "
        "https://github.com/DoctorDean/molforge/issues."
    )


def write_sdf(protein: Protein, path: str | PathLike[str], **kwargs: object) -> None:
    """Write a Protein to a SDF file."""
    raise NotImplementedError(
        "SDF write support is planned; track progress at "
        "https://github.com/DoctorDean/molforge/issues."
    )
