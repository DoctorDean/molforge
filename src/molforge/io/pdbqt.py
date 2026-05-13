"""PDBQT (AutoDock Vina) format reader and writer.

PDBQT (AutoDock Vina) is docking-prepared structures with assigned atom types, partial charges, and rotatable bonds for AutoDock Vina.

**Status: stub.** The API surface is committed; the implementation is
planned. Calling :func:`read_pdbqt` or :func:`write_pdbqt` currently
raises :class:`NotImplementedError` with a pointer to the relevant issue.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from os import PathLike

    from molforge.core import Protein


def read_pdbqt(path: str | PathLike[str], **kwargs: object) -> Protein:
    """Read a PDBQT file."""
    raise NotImplementedError(
        "PDBQT read support is planned; track progress at "
        "https://github.com/DoctorDean/molforge/issues."
    )


def write_pdbqt(protein: Protein, path: str | PathLike[str], **kwargs: object) -> None:
    """Write a Protein to a PDBQT file."""
    raise NotImplementedError(
        "PDBQT write support is planned; track progress at "
        "https://github.com/DoctorDean/molforge/issues."
    )
