"""mmCIF / PDBx format reader and writer.

mmCIF / PDBx is the modern PDB format, recommended for structures exceeding PDB's 99,999-atom or 9,999-residue-per-chain limits.

**Status: stub.** The API surface is committed; the implementation is
planned. Calling :func:`read_mmcif` or :func:`write_mmcif` currently
raises :class:`NotImplementedError` with a pointer to the relevant issue.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from os import PathLike

    from molforge.core import Protein


def read_mmcif(path: str | PathLike[str], **kwargs: object) -> Protein:
    """Read a MMCIF file."""
    raise NotImplementedError(
        "MMCIF read support is planned; track progress at "
        "https://github.com/DoctorDean/molforge/issues."
    )


def write_mmcif(protein: Protein, path: str | PathLike[str], **kwargs: object) -> None:
    """Write a Protein to a MMCIF file."""
    raise NotImplementedError(
        "MMCIF write support is planned; track progress at "
        "https://github.com/DoctorDean/molforge/issues."
    )
