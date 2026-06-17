"""PDBQT (AutoDock / Vina) format reader and writer.

PDBQT is the format AutoDock and AutoDock Vina use for docking input
and output. It is a thin extension of PDB:

- Columns 1-66 are exactly the standard PDB ``ATOM`` / ``HETATM``
  layout (atom name, residue, chain, coordinates, occupancy, B-factor).
- Columns 71-76 hold the per-atom *partial charge* as a 6-character
  right-aligned float.
- Columns 78-79 hold the *AutoDock atom type* (``C``, ``N``, ``OA``,
  ``HD``, ``NA``, ``A``, ``S``, ``P``, ...). The HAD/AD4 type
  distinguishes hydrogen-bond donors / acceptors and aromatic carbons
  from their plain counterparts.
- Lines like ``ROOT`` / ``ENDROOT`` / ``BRANCH`` / ``ENDBRANCH`` /
  ``TORSDOF`` describe the rotatable-bond tree for ligands. molforge
  doesn't carry rotatable-bond information, so these lines are
  read-tolerated and not regenerated on write.

Because the leading 66 columns are PDB-compatible, this module reuses
:func:`molforge.io.read_pdb_string` for the heavy lifting (atom-array
construction, altloc handling, entity classification, multi-model
parsing) and only post-processes the atom block to pick up the extra
columns. The result is a :class:`Protein` whose ``charge`` field is
populated from the PDBQT charges, and whose ``metadata["autodock_types"]``
is the per-atom AutoDock type list.

The writer is the symmetric operation: it calls
:func:`molforge.io.write_pdb_string`, then rewrites each ``ATOM`` /
``HETATM`` line to append the charge and AutoDock-type columns. The
AutoDock type defaults to the element when an explicit type is not
recorded — sufficient for parsing Vina's pose output back into
coordinates (which is the main consumer in this package), though
**not** sufficient for re-running AutoDock from molforge-written
PDBQTs (those require meeko / AutoDockTools for accurate typing).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from molforge.io.pdb import read_pdb_string, write_pdb_string

if TYPE_CHECKING:
    from os import PathLike

    from molforge.core import Protein


__all__ = ["read_pdbqt", "read_pdbqt_string", "write_pdbqt"]


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------


def read_pdbqt(
    path: str | PathLike[str],
    *,
    model: int | None = None,
    include_hydrogens: bool = True,
    altloc: str = "highest_occupancy",
    **_kwargs: object,
) -> Protein:
    """Read a PDBQT file.

    Args:
        path: Path to a PDBQT file (AutoDock / Vina format).
        model: 1-based model index to extract, or ``None`` for all
            models (forwarded to :func:`read_pdb_string`).
        include_hydrogens: Whether to keep hydrogen atoms (forwarded).
        altloc: Alternate-location resolution strategy (forwarded).

    Returns:
        A :class:`Protein` with coordinates and the standard PDB
        attributes populated, plus:
          - ``protein.atom_array.charge`` from the PDBQT charge column
          - ``protein.metadata["autodock_types"]`` — a list of the
            per-atom AutoDock type strings, in atom order.

    Notes:
        ROOT / BRANCH / TORSDOF lines (the rotatable-bond tree) are
        recognised and ignored; they describe ligand topology that
        :class:`AtomArray` doesn't model. Multi-MODEL files (Vina pose
        output) are handled by the underlying PDB reader.
    """
    text = Path(path).read_text(encoding="utf-8")
    return read_pdbqt_string(
        text,
        model=model,
        include_hydrogens=include_hydrogens,
        altloc=altloc,
    )


def read_pdbqt_string(
    text: str,
    *,
    model: int | None = None,
    include_hydrogens: bool = True,
    altloc: str = "highest_occupancy",
) -> Protein:
    """Parse an in-memory PDBQT string into a :class:`Protein`.

    This is the entry point used by docking wrappers that capture
    PDBQT from a subprocess. The signature mirrors
    :func:`read_pdb_string`.
    """
    # Strip lines the standard PDB parser doesn't understand. ROOT /
    # BRANCH / TORSDOF describe rotatable-bond topology that the PDB
    # reader has no place for and is mostly silent about, but skipping
    # them keeps the surface tidy.
    pdb_lines: list[str] = []
    charges: list[float] = []
    autodock_types: list[str] = []
    for raw in text.splitlines():
        if raw.startswith(("ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "TORSDOF")):
            continue
        if raw.startswith(("ATOM", "HETATM")):
            # Strip the PDBQT-specific tail so the PDB parser sees a
            # clean record (cols 1-66 are PDB-compatible).
            pdb_lines.append(raw[:66])
            charges.append(_parse_charge(raw))
            autodock_types.append(_parse_autodock_type(raw))
        else:
            pdb_lines.append(raw)

    pdb_text = "\n".join(pdb_lines)
    protein = read_pdb_string(
        pdb_text,
        model=model,
        include_hydrogens=include_hydrogens,
        altloc=altloc,
    )

    n = protein.atom_array.n_atoms
    if len(charges) == n:
        # Patch in the PDBQT charges. (The lengths can differ if the
        # PDB reader dropped atoms — e.g. altloc resolution — in which
        # case we keep the PDB reader's zero-defaulted charge column
        # rather than risk a misaligned assignment.)
        protein.atom_array.charge = np.asarray(charges, dtype=np.float32)
    if len(autodock_types) == n and any(t for t in autodock_types):
        meta: dict[str, Any] = {**protein.metadata, "autodock_types": autodock_types}
        protein.metadata = meta
    return protein


def _parse_charge(line: str) -> float:
    """Pull the partial charge out of a PDBQT atom line.

    Columns 71-76 hold the charge as a right-aligned float (e.g.
    ``"  0.103"``). Older / non-conforming writers sometimes pad
    differently, so the parser falls back to taking the second-to-last
    whitespace token when the fixed-column read fails.
    """
    if len(line) >= 76:
        chunk = line[70:76].strip()
        if chunk:
            try:
                return float(chunk)
            except ValueError:
                pass
    # Fallback: split on whitespace and take the second-to-last token,
    # which is the charge before the trailing AutoDock-type column.
    parts = line.split()
    if len(parts) >= 2:
        try:
            return float(parts[-2])
        except ValueError:
            return 0.0
    return 0.0


def _parse_autodock_type(line: str) -> str:
    """Pull the AutoDock atom type from a PDBQT atom line.

    Columns 78-79 hold the type as a 2-character right-aligned string
    (``"OA"``, ``"HD"``, ``"NA"``, ...). For lines whose AutoDock-type
    column is absent or whitespace-only, returns ``""``.
    """
    if len(line) >= 79:
        chunk = line[77:79].strip()
        if chunk:
            return chunk
    # Fallback: the last whitespace-separated token is the type when
    # the line wasn't padded to 79 columns.
    parts = line.rstrip().split()
    if parts:
        last = parts[-1]
        # Heuristic: AutoDock types are short (1-2 chars typically);
        # avoid mistaking a stray coord token for a type.
        if 1 <= len(last) <= 2 and last.isalpha():
            return last
    return ""


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------


def write_pdbqt(
    protein: Protein,
    path: str | PathLike[str],
    *,
    write_end: bool = True,
    **_kwargs: object,
) -> None:
    """Write a :class:`Protein` to a PDBQT file.

    Args:
        protein: The structure to write. Per-atom charges are taken
            from :attr:`AtomArray.charge`. AutoDock atom types are
            taken from ``protein.metadata["autodock_types"]`` when
            present; otherwise the element symbol is used as a
            best-effort fallback.
        path: Where to write.
        write_end: Whether to emit a terminating ``END`` record
            (forwarded to :func:`write_pdb_string`).

    Notes:
        This writer is **not** a substitute for ``meeko`` /
        ``prepare_ligand4.py``: it emits coordinates and charges
        usable for round-tripping or visualisation, but the AutoDock
        types are not perceived from chemistry — they're either the
        ones already on metadata, or the bare element. Use the
        ``molforge.wrappers.docking.prep`` helpers when you need a
        Vina-ready PDBQT with proper atom typing.
    """
    Path(path).write_text(_to_pdbqt_string(protein, write_end=write_end), encoding="utf-8")


def _to_pdbqt_string(protein: Protein, *, write_end: bool = True) -> str:
    """Render a :class:`Protein` as PDBQT text."""
    pdb_text = write_pdb_string(protein, write_end=write_end)
    arr = protein.atom_array
    autodock_types = protein.metadata.get("autodock_types")
    if not isinstance(autodock_types, list):
        autodock_types = None

    out_lines: list[str] = []
    atom_idx = 0
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")) and atom_idx < arr.n_atoms:
            charge = float(arr.charge[atom_idx])
            ad_type = ""
            if autodock_types is not None and atom_idx < len(autodock_types):
                ad_type = str(autodock_types[atom_idx])
            if not ad_type:
                ad_type = str(arr.element[atom_idx]).strip() or "X"
            # PDB body is exactly 66 columns; pad if shorter, truncate
            # the standard-element/charge tail (cols 67-80 in PDB) so
            # we can write our own.
            body = line[:66].ljust(66)
            # Cols 67-70 blank; cols 71-76 = charge (% 6.3f); col 77
            # blank; cols 78-79 = AutoDock type, right-aligned.
            out_lines.append(f"{body}    {charge:6.3f} {ad_type:>2}")
            atom_idx += 1
        else:
            out_lines.append(line)
    return "\n".join(out_lines) + "\n"
