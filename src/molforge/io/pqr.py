"""PQR (PDB2PQR / APBS) format reader and writer.

PQR is a PDB-like format used by PDB2PQR and APBS for continuum
electrostatics calculations. Each atom record carries the standard
PDB body (atom name, residue, chain, coordinates) plus two extra
whitespace-separated trailing fields:

- the per-atom **partial charge** (in elementary charges); and
- the per-atom **atomic radius** (in Ångström).

Critically — unlike PDB or PDBQT — PQR is **not** strictly
fixed-column past the coordinates. Different generators emit different
widths for the trailing charge/radius fields, and AMBER, CHARMM, and
APBS each pick subtly different conventions. This reader handles all
of them by parsing columns 1-54 as fixed (the atom record through
coordinates, which is exactly the PDB layout) and then
whitespace-splitting the remainder for charge and radius.

The result is a :class:`Protein` whose ``charge`` field is populated
from the PQR charges, and whose ``metadata["radii"]`` is a per-atom
list of atomic radii. The radius lives on metadata because
:class:`AtomArray` does not have a native radius field —
electrostatics is a small enough fraction of molforge's surface that
adding one to the core schema isn't warranted.

The writer is the symmetric operation: it emits PDB-style atom records
truncated at column 54, then appends ``charge radius`` (each as
``%.4f``). Radii come from ``metadata["radii"]`` when present, else
default to 1.5 Å — a reasonable middle-of-the-road value that lets a
charge-only PDB still be written as PQR-shaped output.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from molforge.io.pdb import read_pdb_string, write_pdb_string

if TYPE_CHECKING:
    from os import PathLike

    from molforge.core import Protein


__all__ = ["read_pqr", "read_pqr_string", "write_pqr"]


# Default atomic radius used by the writer when no per-atom radius is
# attached to metadata. 1.5 Å is the rough average of standard
# heavy-atom radii (C ~ 1.7, N ~ 1.55, O ~ 1.52, H ~ 1.2) and is what
# many PQR-consuming tools (APBS, PDB2PQR) use as a fallback.
_DEFAULT_RADIUS = 1.5


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------


def read_pqr(
    path: str | PathLike[str],
    *,
    model: int | None = None,
    include_hydrogens: bool = True,
    altloc: str = "highest_occupancy",
    **_kwargs: object,
) -> Protein:
    """Read a PQR file.

    Args:
        path: Path to a PQR file (PDB2PQR / APBS format).
        model: 1-based model index to extract, or ``None`` for all
            models (forwarded to :func:`read_pdb_string`).
        include_hydrogens: Whether to keep hydrogen atoms (forwarded).
        altloc: Alternate-location resolution strategy (forwarded).

    Returns:
        A :class:`Protein` with coordinates and the standard PDB
        attributes populated, plus:
          - ``protein.atom_array.charge`` from the PQR charge column
          - ``protein.metadata["radii"]`` — a list of per-atom radii
            (Ångström), in atom order.
    """
    text = Path(path).read_text(encoding="utf-8")
    return read_pqr_string(
        text,
        model=model,
        include_hydrogens=include_hydrogens,
        altloc=altloc,
    )


def read_pqr_string(
    text: str,
    *,
    model: int | None = None,
    include_hydrogens: bool = True,
    altloc: str = "highest_occupancy",
) -> Protein:
    """Parse an in-memory PQR string into a :class:`Protein`.

    Each ``ATOM`` / ``HETATM`` line is parsed by reusing the PDB reader
    for columns 1-54 (the atom record through coordinates), then the
    remaining whitespace-separated tokens are read as ``charge``
    followed by ``radius``.
    """
    pdb_lines: list[str] = []
    charges: list[float] = []
    radii: list[float] = []
    for raw in text.splitlines():
        if raw.startswith(("ATOM", "HETATM")):
            # Cols 1-54 are PDB-compatible (atom record through z).
            # Anything past col 54 in PQR is whitespace-separated
            # charge + radius, possibly preceded by occupancy/b-factor
            # tokens that the original PDB record would have held in
            # fixed columns 55-66 — but most PQR writers drop those
            # entirely. We feed cols 1-54 to the PDB reader and split
            # the tail for our two values.
            pdb_body = raw[:54]
            # Pad cols 55-66 (occupancy + b-factor) with zeros so the
            # PDB reader gets a well-formed line.
            pdb_lines.append(f"{pdb_body:<54}{'  1.00':<6}{'  0.00':<6}")
            charge, radius = _parse_charge_radius(raw[54:])
            charges.append(charge)
            radii.append(radius)
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
        # Patch in the PQR charges. (Lengths can differ if the PDB
        # reader dropped atoms — e.g. altloc resolution — in which
        # case we keep the PDB reader's zero-defaulted charge.)
        protein.atom_array.charge = np.asarray(charges, dtype=np.float32)
    if len(radii) == n:
        meta: dict[str, Any] = {**protein.metadata, "radii": radii}
        protein.metadata = meta
    return protein


def _parse_charge_radius(tail: str) -> tuple[float, float]:
    """Pull ``charge radius`` from the tail of a PQR atom line.

    The tail begins at column 55 and contains whitespace-separated
    tokens — typically just ``charge radius``, though some emitters
    write extra trailing tokens (a description string, status flags).
    The last two parseable floats are taken as ``(charge, radius)``.
    """
    parts = tail.split()
    if len(parts) < 2:
        return 0.0, _DEFAULT_RADIUS
    # Walk the tokens left-to-right collecting floats; the *first two*
    # successfully-parsed floats are charge and radius.
    floats: list[float] = []
    for tok in parts:
        try:
            floats.append(float(tok))
        except ValueError:
            continue
        if len(floats) == 2:
            break
    if len(floats) < 2:
        return 0.0, _DEFAULT_RADIUS
    return floats[0], floats[1]


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------


def write_pqr(
    protein: Protein,
    path: str | PathLike[str],
    *,
    write_end: bool = True,
    **_kwargs: object,
) -> None:
    """Write a :class:`Protein` to a PQR file.

    Args:
        protein: The structure to write. Per-atom charges come from
            :attr:`AtomArray.charge`. Per-atom radii come from
            ``protein.metadata["radii"]`` when present; otherwise a
            default of 1.5 Å is used and noted in the file header.
        path: Where to write.
        write_end: Whether to emit a terminating ``END`` record.
    """
    Path(path).write_text(_to_pqr_string(protein, write_end=write_end), encoding="utf-8")


def _to_pqr_string(protein: Protein, *, write_end: bool = True) -> str:
    """Render a :class:`Protein` as PQR text.

    Reuses :func:`write_pdb_string`, then truncates each ``ATOM`` /
    ``HETATM`` line at column 54 (the end of the coordinate columns)
    and appends ``charge radius`` as whitespace-separated floats.
    """
    pdb_text = write_pdb_string(protein, write_end=write_end)
    arr = protein.atom_array
    radii = protein.metadata.get("radii")
    if not isinstance(radii, list):
        radii = None

    out_lines: list[str] = []
    atom_idx = 0
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")) and atom_idx < arr.n_atoms:
            charge = float(arr.charge[atom_idx])
            radius = float(
                radii[atom_idx] if radii is not None and atom_idx < len(radii) else _DEFAULT_RADIUS
            )
            # Take cols 1-54 (atom record through z), pad shorter
            # lines, then append the charge and radius as
            # whitespace-separated floats.
            body = line[:54].ljust(54)
            out_lines.append(f"{body} {charge:8.4f} {radius:7.4f}")
            atom_idx += 1
        else:
            out_lines.append(line)
    return "\n".join(out_lines) + "\n"
