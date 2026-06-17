"""SDF (Structure Data File) format reader and writer.

The Structure Data File format is the standard small-molecule exchange
format: a V2000 (or V3000) molfile followed by a block of free-form
property fields and a ``$$$$`` delimiter. A single ``.sdf`` file
routinely contains thousands of compounds, so the reader returns a
:class:`list` of :class:`Protein` objects.

molforge represents a small molecule as a :class:`Protein` whose
atoms carry ``entity_type="ligand"`` and ``record_type="HETATM"`` —
the same convention the docking wrappers use. This keeps a single
type flowing through the pipeline whether a "structure" came from a
PDB file or an SDF.

This module parses SDF without depending on RDKit. The V2000 atom
block has a fixed layout (3D coordinates + element symbols + a few
flags), which is enough for everything molforge does downstream:
coordinate handling, pose ranking, distance calculations,
visualisation. Bond orders, formal charges beyond the atom block,
aromaticity, and stereochemistry are intentionally dropped — those
need a chemistry toolkit, and users who want them should call into
RDKit / OpenBabel directly.

V3000 ("extended connection table") files are recognised and raise a
clear error pointing at the limitation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from molforge.core import AtomArray, Protein

if TYPE_CHECKING:
    from os import PathLike


__all__ = ["read_sdf", "read_sdf_string", "write_sdf"]


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------


def read_sdf(path: str | PathLike[str], **_kwargs: object) -> list[Protein]:
    """Read every molecule from a ``.sdf`` (or ``.mol``) file.

    Args:
        path: Path to a V2000 SDF or MOL file. Single-molecule
            ``.mol`` files are handled too — they're just SDF without
            the property block or the ``$$$$`` delimiter.

    Returns:
        One :class:`Protein` per molecule in the file. A
        single-molecule ``.mol`` therefore returns a one-element list,
        which keeps the return type uniform across callers.

    Raises:
        ValueError: If the file is not parseable as a V2000 molfile,
            or if it declares V3000 (extended) connectivity which this
            reader does not yet handle.
    """
    text = Path(path).read_text(encoding="utf-8")
    return read_sdf_string(text)


def read_sdf_string(text: str) -> list[Protein]:
    """Parse an in-memory SDF string into a list of :class:`Protein`.

    The string-form entry point — useful when the SDF was returned
    from a subprocess (the docking wrappers use it that way) rather
    than read from disk.

    Args:
        text: The full SDF contents.

    Returns:
        One :class:`Protein` per molecule, in file order.

    Raises:
        ValueError: If a molecule block is malformed or uses V3000.
    """
    molecules: list[Protein] = []
    for block in _split_molecules(text):
        if not block.strip():
            continue
        molecules.append(_parse_molecule_block(block))
    return molecules


def _split_molecules(text: str) -> list[str]:
    """Split an SDF on the ``$$$$`` molecule-end delimiter.

    A trailing block without a delimiter (i.e. a single-molecule
    ``.mol``) is still treated as one molecule.
    """
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip() == "$$$$":
            blocks.append("\n".join(current))
            current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def _parse_molecule_block(block: str) -> Protein:
    """Parse one molecule (V2000 molfile + optional property block)."""
    lines = block.splitlines()
    # Be tolerant of leading blank lines that some splitters / file
    # concatenations leave behind. The V2000 layout is positional, so
    # an empty header line shifts every subsequent index.
    while lines and not lines[0].strip():
        lines = lines[1:]
    if len(lines) < 4:
        raise ValueError("SDF molecule block is too short to contain a molfile")

    title = lines[0]
    # lines[1] is the program/timestamp line; lines[2] is the comment.
    counts_line = lines[3]

    # The counts line ends with the version string ("V2000" / "V3000")
    # in columns 34-39. Detect V3000 early so the error is clear.
    version = counts_line[33:39].strip() if len(counts_line) >= 39 else ""
    if version == "V3000":
        raise ValueError(
            "V3000 SDF (extended connection table) is not yet supported. "
            "Convert the file with `obabel input.sdf -O output.sdf` or use "
            "RDKit's MolFromMolBlock for V3000 input."
        )

    try:
        n_atoms = int(counts_line[:3])
    except ValueError as e:
        raise ValueError(f"could not parse atom count from SDF counts line: {counts_line!r}") from e

    # The atom block follows the counts line; one line per atom.
    first_atom = 4
    if len(lines) < first_atom + n_atoms:
        raise ValueError(f"SDF declares {n_atoms} atoms but the atom block is truncated")

    coords = np.zeros((n_atoms, 3), dtype=np.float32)
    elements = np.empty(n_atoms, dtype="U2")
    atom_names = np.empty(n_atoms, dtype="U4")
    element_counts: dict[str, int] = {}
    for i in range(n_atoms):
        atom_line = lines[first_atom + i]
        try:
            x = float(atom_line[0:10])
            y = float(atom_line[10:20])
            z = float(atom_line[20:30])
        except (ValueError, IndexError) as e:
            raise ValueError(f"malformed SDF atom line: {atom_line!r}") from e
        element = atom_line[31:34].strip() if len(atom_line) >= 34 else ""
        coords[i] = (x, y, z)
        elements[i] = element
        # Give each atom a unique, element-derived name (C1, C2, N1, ...).
        element_counts[element] = element_counts.get(element, 0) + 1
        atom_names[i] = f"{element}{element_counts[element]}"[:4]

    n = n_atoms
    arr = AtomArray.from_dict(
        {
            "coords": coords,
            "element": elements,
            "atom_name": atom_names,
            "residue_name": np.full(n, "LIG", dtype="U3"),
            "residue_id": np.ones(n, dtype=np.int32),
            "chain_id": np.full(n, "L", dtype="U4"),
            "record_type": np.full(n, "HETATM", dtype="U6"),
            "entity_type": np.full(n, "ligand", dtype="U8"),
        }
    )
    protein = Protein(arr)

    # Attach the title and any property fields as metadata.
    metadata: dict[str, Any] = {}
    if title.strip():
        metadata["title"] = title.strip()
    properties = _parse_property_block(lines)
    if properties:
        metadata["properties"] = properties
    if metadata:
        protein.metadata = {**protein.metadata, **metadata}
    return protein


def _parse_property_block(lines: list[str]) -> dict[str, str]:
    """Pick the ``> <Name>`` / value pairs out of an SDF molecule block.

    Property fields appear after the ``M  END`` line and look like::

        > <PropertyName>
        value line 1
        value line 2

        > <NextProperty>
        ...

    Returns a flat ``{name: value}`` dict. Multi-line values are joined
    with newlines.
    """
    properties: dict[str, str] = {}
    # Find where the molfile ends (M  END) — properties start after it.
    end_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "M  END":
            end_idx = i + 1
            break
    if end_idx is None:
        return properties

    i = end_idx
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("> <") and line.endswith(">"):
            name = line[3:-1]
            values: list[str] = []
            i += 1
            while i < len(lines) and lines[i].strip() != "":
                values.append(lines[i])
                i += 1
            properties[name] = "\n".join(values)
        i += 1
    return properties


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------


def write_sdf(
    proteins: Protein | list[Protein],
    path: str | PathLike[str],
    **_kwargs: object,
) -> None:
    """Write one or more :class:`Protein` objects to a V2000 SDF file.

    Args:
        proteins: A single :class:`Protein` or a list. When given a
            single :class:`Protein`, the output is effectively a
            single-molecule SDF (the ``$$$$`` delimiter is still
            written so the file remains a valid multi-molecule SDF
            that other tools can append to).
        path: Where to write.

    Notes:
        Bond orders are not tracked by :class:`AtomArray`, so the
        output has an empty bond block. Downstream tools that need
        connectivity (e.g. for chemistry-aware comparison) should
        re-perceive bonds with RDKit / OpenBabel.

        Any ``title`` and ``properties`` keys on :attr:`Protein.metadata`
        are written back out as the molfile title and the SDF property
        block respectively.
    """
    items = [proteins] if isinstance(proteins, Protein) else list(proteins)
    # Each _format_molecule output already ends with "\n", so a plain
    # concatenation produces the correct one-newline separation between
    # molecules. Using "\n".join here would inject an extra blank line
    # that bounces the line indices on read-back.
    out = "".join(_format_molecule(p) for p in items)
    Path(path).write_text(out, encoding="utf-8")


def _format_molecule(protein: Protein) -> str:
    """Render one :class:`Protein` as a V2000 SDF block + ``$$$$``."""
    arr = protein.atom_array
    n_atoms = arr.n_atoms
    title = str(protein.metadata.get("title", "") or "")

    lines: list[str] = []
    # Header: title line, program/comment lines (kept minimal), counts line.
    lines.append(title)
    lines.append("  molforge")
    lines.append("")
    # Counts line: NNNBBB ... V2000. We write zero bonds.
    lines.append(f"{n_atoms:>3}{0:>3}  0  0  0  0  0  0  0  0999 V2000")

    # Atom block: x, y, z (each 10.4f), element (right-padded to 3),
    # then 12 zero fields (mass diff, charge, stereo, ...).
    for i in range(n_atoms):
        x, y, z = arr.coords[i]
        element = str(arr.element[i] or "C")
        # SDF column layout: 10.4 / 10.4 / 10.4 / 1 space / 3-char element /
        # 12 trailing zero fields (each 3 chars wide).
        lines.append(f"{x:10.4f}{y:10.4f}{z:10.4f} {element:<3} 0  0  0  0  0  0  0  0  0  0  0  0")

    lines.append("M  END")

    # Property block.
    properties = protein.metadata.get("properties")
    if isinstance(properties, dict):
        for name, value in properties.items():
            lines.append(f"> <{name}>")
            lines.append(str(value))
            lines.append("")

    # Molecule end delimiter.
    lines.append("$$$$")
    return "\n".join(lines) + "\n"
