"""MOL2 (Tripos) format reader and writer.

MOL2 is a small-molecule exchange format that pairs 3D coordinates
with Tripos atom types (e.g. ``C.ar`` for aromatic carbon, ``N.am``
for amide nitrogen) and per-atom partial charges. Unlike SDF, MOL2 is
section-based — ``@<TRIPOS>MOLECULE``, ``@<TRIPOS>ATOM``,
``@<TRIPOS>BOND``, and so on — and supports multi-molecule files via
repeated ``@<TRIPOS>MOLECULE`` markers.

molforge represents a small molecule as a :class:`Protein` whose atoms
carry ``entity_type="ligand"`` and ``record_type="HETATM"``. The MOL2
reader populates:

- coordinates and elements (the element is the symbol before the ``.``
  in the Tripos atom type — ``C.3`` → ``C``, ``Cl`` → ``Cl``);
- the atom's Tripos type, stored on ``AtomArray.atom_name``;
- per-atom partial charges from the atom line's last column;
- residue id / name from the MOL2 substructure columns when present.

Bond orders, ring information, formal stereochemistry, and the
``@<TRIPOS>SUBSTRUCTURE`` / ``@<TRIPOS>CRYSIN`` / ``@<TRIPOS>UNITY``
sections are intentionally dropped — those need a chemistry toolkit
to interpret correctly, and users who want them should call RDKit /
OpenBabel directly.

The writer emits a minimal, spec-conformant MOL2: the
``@<TRIPOS>MOLECULE`` and ``@<TRIPOS>ATOM`` sections plus an empty
``@<TRIPOS>BOND`` section (some downstream tools error without one).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from molforge.core import AtomArray, Protein

if TYPE_CHECKING:
    from os import PathLike


__all__ = ["read_mol2", "read_mol2_string", "write_mol2"]


_RECORD_TAG = "@<TRIPOS>"
_MOLECULE_TAG = f"{_RECORD_TAG}MOLECULE"
_ATOM_TAG = f"{_RECORD_TAG}ATOM"
_BOND_TAG = f"{_RECORD_TAG}BOND"


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------


def read_mol2(path: str | PathLike[str], **_kwargs: object) -> list[Protein]:
    """Read every molecule from a ``.mol2`` file.

    Args:
        path: Path to a Tripos MOL2 file.

    Returns:
        One :class:`Protein` per ``@<TRIPOS>MOLECULE`` block in the
        file. A single-molecule MOL2 therefore returns a one-element
        list, keeping the return type uniform across callers.

    Raises:
        ValueError: If the file contains no ``@<TRIPOS>MOLECULE``
            section, or if a block is malformed.
    """
    text = Path(path).read_text(encoding="utf-8")
    return read_mol2_string(text)


def read_mol2_string(text: str) -> list[Protein]:
    """Parse an in-memory MOL2 string into a list of :class:`Protein`.

    The string-form entry point — useful when a MOL2 was returned from
    a subprocess (the docking wrappers' pattern) rather than read from
    disk.

    Args:
        text: The full MOL2 contents.

    Returns:
        One :class:`Protein` per ``@<TRIPOS>MOLECULE`` block.

    Raises:
        ValueError: If a molecule block is malformed.
    """
    blocks = _split_molecules(text)
    return [_parse_molecule_block(b) for b in blocks]


def _split_molecules(text: str) -> list[str]:
    """Split a MOL2 stream on ``@<TRIPOS>MOLECULE`` markers.

    Returns one string per molecule (each starts with the
    ``@<TRIPOS>MOLECULE`` line). Empty/whitespace-only files yield an
    empty list.
    """
    blocks: list[str] = []
    current: list[str] = []
    in_molecule = False
    for line in text.splitlines():
        if line.startswith(_MOLECULE_TAG):
            if in_molecule and current:
                blocks.append("\n".join(current))
            current = [line]
            in_molecule = True
        elif in_molecule:
            current.append(line)
    if in_molecule and current:
        blocks.append("\n".join(current))
    return blocks


def _parse_molecule_block(block: str) -> Protein:
    """Parse one ``@<TRIPOS>MOLECULE`` block into a :class:`Protein`."""
    sections = _split_sections(block)

    if "MOLECULE" not in sections:
        raise ValueError("MOL2 block missing @<TRIPOS>MOLECULE section")
    if "ATOM" not in sections:
        raise ValueError("MOL2 block missing @<TRIPOS>ATOM section")

    title, declared_n_atoms = _parse_molecule_header(sections["MOLECULE"])
    coords, elements, atom_names, residue_ids, residue_names, charges = _parse_atom_section(
        sections["ATOM"]
    )
    n = len(coords)

    # The MOLECULE header declares the atom count; trust the atom block
    # but flag a mismatch — many tools quietly truncate or pad and the
    # error is the kind of thing worth surfacing.
    if declared_n_atoms is not None and declared_n_atoms != n:
        raise ValueError(
            f"MOL2 MOLECULE header declares {declared_n_atoms} atoms but "
            f"the ATOM section contains {n}"
        )

    arr = AtomArray.from_dict(
        {
            "coords": np.asarray(coords, dtype=np.float32),
            "element": np.asarray(elements, dtype="U2"),
            "atom_name": np.asarray(atom_names, dtype="U4"),
            "residue_name": np.asarray(residue_names, dtype="U3"),
            "residue_id": np.asarray(residue_ids, dtype=np.int32),
            "charge": np.asarray(charges, dtype=np.float32),
            "chain_id": np.full(n, "L", dtype="U4"),
            "record_type": np.full(n, "HETATM", dtype="U6"),
            "entity_type": np.full(n, "ligand", dtype="U8"),
        }
    )
    protein = Protein(arr)
    metadata: dict[str, Any] = {}
    if title:
        metadata["title"] = title
    if metadata:
        protein.metadata = {**protein.metadata, **metadata}
    return protein


def _split_sections(block: str) -> dict[str, list[str]]:
    """Split a single molecule block into its ``@<TRIPOS>...`` sections.

    Returns a dict mapping section name (e.g. ``"MOLECULE"``,
    ``"ATOM"``) to the list of body lines for that section (the tag
    line itself is excluded).
    """
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in block.splitlines():
        if line.startswith(_RECORD_TAG):
            current = line[len(_RECORD_TAG) :].strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return sections


def _parse_molecule_header(lines: list[str]) -> tuple[str, int | None]:
    """Pull the title and declared atom count from the MOLECULE section.

    The MOL2 MOLECULE header is positional:

      line 0:  molecule name
      line 1:  num_atoms num_bonds num_subst num_feat num_sets
      line 2:  mol_type (SMALL / PROTEIN / ...)
      line 3:  charge_type
      line 4+: optional status_bits and comment

    Anything beyond what we use is ignored.
    """
    name = ""
    declared_n_atoms: int | None = None
    # Find the first non-empty line for the name.
    for i, line in enumerate(lines):
        if line.strip():
            name = line.strip()
            counts_idx = i + 1
            break
    else:
        return name, declared_n_atoms

    if counts_idx < len(lines):
        try:
            declared_n_atoms = int(lines[counts_idx].split()[0])
        except (ValueError, IndexError):
            declared_n_atoms = None
    return name, declared_n_atoms


def _parse_atom_section(
    lines: list[str],
) -> tuple[
    list[tuple[float, float, float]],
    list[str],
    list[str],
    list[int],
    list[str],
    list[float],
]:
    """Parse the ``@<TRIPOS>ATOM`` body into parallel column lists.

    Each MOL2 atom line is whitespace-separated and looks like::

        atom_id atom_name x y z atom_type [subst_id subst_name [charge [status_bit]]]

    All columns past ``atom_type`` are optional, so the parser handles
    short lines gracefully (filling defaults where data is absent).
    """
    coords: list[tuple[float, float, float]] = []
    elements: list[str] = []
    atom_names: list[str] = []
    residue_ids: list[int] = []
    residue_names: list[str] = []
    charges: list[float] = []
    for raw in lines:
        if not raw.strip():
            continue
        parts = raw.split()
        if len(parts) < 6:
            raise ValueError(f"malformed MOL2 atom line: {raw!r}")
        # parts[0] = atom_id (positional, not used)
        atom_name = parts[1]
        try:
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError as e:
            raise ValueError(f"malformed coordinates in MOL2 atom line: {raw!r}") from e
        atom_type = parts[5]
        # atom_type like "C.3", "N.am" — the element is the prefix.
        element = atom_type.split(".", 1)[0]
        residue_id = 1
        residue_name = "LIG"
        if len(parts) >= 7:
            try:
                residue_id = int(parts[6])
            except ValueError:
                residue_id = 1
        if len(parts) >= 8:
            residue_name = parts[7]
        charge = 0.0
        if len(parts) >= 9:
            try:
                charge = float(parts[8])
            except ValueError:
                charge = 0.0

        coords.append((x, y, z))
        elements.append(element)
        # atom_name in AtomArray is U4; MOL2 atom names like "C1", "HB2"
        # fit, but we truncate defensively.
        atom_names.append(atom_name[:4])
        residue_ids.append(residue_id)
        residue_names.append(residue_name[:3])
        charges.append(charge)
    if not coords:
        raise ValueError("MOL2 ATOM section is empty")
    return coords, elements, atom_names, residue_ids, residue_names, charges


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------


def write_mol2(
    proteins: Protein | list[Protein],
    path: str | PathLike[str],
    **_kwargs: object,
) -> None:
    """Write one or more :class:`Protein` objects to a MOL2 file.

    Args:
        proteins: A single :class:`Protein` or a list. Each becomes
            one ``@<TRIPOS>MOLECULE`` block.
        path: Where to write.

    Notes:
        Bond orders are not tracked by :class:`AtomArray`, so the
        ``@<TRIPOS>BOND`` section is empty. Downstream tools that need
        connectivity should re-perceive bonds with RDKit / OpenBabel.

        The atom-type column is filled from ``AtomArray.element`` when
        no chemistry-aware type is available — a flat ``C``, ``N``, ``O``
        rather than ``C.3``, ``N.am``. This is enough for tools that
        only need coordinates; it is **not** enough for tools that
        require accurate Tripos types (most molecular-mechanics tools
        fall in this latter group).

        Any ``title`` on :attr:`Protein.metadata` is written as the
        molecule name.
    """
    items = [proteins] if isinstance(proteins, Protein) else list(proteins)
    out = "".join(_format_molecule(p) for p in items)
    Path(path).write_text(out, encoding="utf-8")


def _format_molecule(protein: Protein) -> str:
    """Render one :class:`Protein` as a MOL2 block."""
    arr = protein.atom_array
    n_atoms = arr.n_atoms
    title = str(protein.metadata.get("title", "") or "ligand")

    lines: list[str] = []
    # MOLECULE header.
    lines.append(_MOLECULE_TAG)
    lines.append(title)
    # num_atoms num_bonds num_subst num_feat num_sets
    lines.append(f"{n_atoms} 0 1 0 0")
    lines.append("SMALL")
    lines.append("USER_CHARGES")
    lines.append("")  # mol_comment / blank

    # ATOM section.
    lines.append(_ATOM_TAG)
    for i in range(n_atoms):
        atom_id = i + 1
        atom_name = str(arr.atom_name[i] or "X")
        x, y, z = arr.coords[i]
        element = str(arr.element[i] or "C")
        # Without bond perception we don't know the Tripos type beyond
        # the element. Emit the element as the type — Tripos accepts
        # bare elements as a degenerate atom type.
        atom_type = element
        residue_id = int(arr.residue_id[i] or 1)
        residue_name = str(arr.residue_name[i] or "LIG")
        charge = float(arr.charge[i])
        lines.append(
            f"{atom_id:>7} {atom_name:<4} "
            f"{x:>10.4f} {y:>10.4f} {z:>10.4f} "
            f"{atom_type:<5} {residue_id:>4} {residue_name:<4} {charge:>9.4f}"
        )

    # Empty BOND section — some readers error without the tag.
    lines.append(_BOND_TAG)

    return "\n".join(lines) + "\n"
