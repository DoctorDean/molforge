"""Chemistry-aware ingestion of small molecules into :class:`Molecule`.

These readers preserve the chemistry — bonds, formal charges, aromaticity,
stereochemistry, and any 3D coordinates — that the coordinate-only
:func:`molforge.io.read_sdf` (which returns :class:`~molforge.core.Protein`)
drops. They are RDKit-backed and therefore lazy: calling one without RDKit
raises :class:`~molforge.core.RDKitNotInstalledError`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from molforge.core import Molecule
from molforge.core import _rdkit

if TYPE_CHECKING:
    from os import PathLike

__all__ = ["read_molecules", "read_smiles"]


_EXT_TO_FORMAT = {
    ".sdf": "sdf",
    ".mol": "sdf",
    ".smi": "smiles",
    ".smiles": "smiles",
}


def _infer_format(path: str | PathLike[str]) -> str:
    ext = Path(path).suffix.lower()
    try:
        return _EXT_TO_FORMAT[ext]
    except KeyError:
        raise ValueError(
            f"can't infer a molecule format from {ext!r}; "
            f"pass format= (one of {sorted(set(_EXT_TO_FORMAT.values()))})"
        ) from None


def read_smiles(
    text: str, *, sanitize: bool = True, source: str = "<string>"
) -> list[Molecule]:
    """Parse a SMILES block into molecules.

    One molecule per line, ``SMILES [name]`` (whitespace-separated); blank
    lines and ``#`` comments are skipped.

    Args:
        text: The SMILES text.
        sanitize: Run RDKit sanitization on each molecule.
        source: Recorded in each molecule's ``metadata["source"]``.

    Returns:
        The parsed molecules, in file order.

    Raises:
        RDKitNotInstalledError: If RDKit isn't installed.
        ValueError: If a SMILES string can't be parsed.
    """
    molecules: list[Molecule] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        smiles = parts[0]
        name = parts[1].strip() if len(parts) > 1 else ""
        mol = _rdkit.mol_from_smiles(smiles, sanitize=sanitize)
        molecules.append(
            Molecule.from_rdkit(mol, name=name, metadata={"source": source})
        )
    return molecules


def read_molecules(
    path: str | PathLike[str],
    *,
    format: str | None = None,
    sanitize: bool = True,
) -> list[Molecule]:
    """Read a molecule file into chemistry-aware :class:`Molecule` objects.

    Supports SDF (``.sdf`` / ``.mol``) and SMILES (``.smi`` / ``.smiles``).
    SDF records RDKit can't parse are skipped so one bad entry doesn't sink
    a bulk read. Each molecule records the source file in its metadata and
    takes its ``name`` from the record (the SDF title or the SMILES name
    column).

    Args:
        path: The file to read.
        format: ``"sdf"`` or ``"smiles"``; inferred from the extension
            when omitted.
        sanitize: Run RDKit sanitization on each molecule.

    Returns:
        The molecules, in file order.

    Raises:
        RDKitNotInstalledError: If RDKit isn't installed.
        ValueError: On an unknown format.
    """
    fmt = (format or _infer_format(path)).lower()
    source = str(path)
    if fmt == "sdf":
        records = _rdkit.read_sdf_records(source, sanitize=sanitize)
        return [
            Molecule.from_rdkit(mol, name=name, metadata={"source": source})
            for mol, name in records
        ]
    if fmt in ("smiles", "smi"):
        return read_smiles(Path(path).read_text(), sanitize=sanitize, source=source)
    raise ValueError(f"unknown molecule format {fmt!r}; expected 'sdf' or 'smiles'")
