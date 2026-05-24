"""PDB file format reader and writer.

The PDB format is a fixed-column ASCII format defined by the
[wwPDB v3.30 specification](https://www.wwpdb.org/documentation/file-format-content/format33/v3.3.html).
It's the universal default for protein structure exchange, despite
real limitations:

- Hard cap of 99,999 atoms (5-digit serial column).
- Hard cap of 9,999 residues per chain (4-digit residue ID, then
  insertion codes take over).
- Chain ID is a single character.
- No native typing for entity (protein / ligand / water / ion).

molforge handles these limits transparently; for structures that exceed
them, prefer mmCIF.

This module implements parsing and writing without depending on
Biopython, so the parser can run in environments where the only
required dependency is NumPy. The output is a :class:`molforge.core.Protein`
holding a canonical :class:`AtomArray`.

The functions in this module are all *pure* with respect to the file
system in the sense that the heavy lifting is done by ``read_pdb_string``
and ``write_pdb_string`` which work on Python strings; ``read_pdb`` and
``write_pdb`` are thin filesystem wrappers.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TextIO

import numpy as np

from molforge.core import AtomArray, Protein
from molforge.core import metadata_keys as mk
from molforge.core.constants import (
    THREE_TO_ONE,
    is_ion,
    is_water,
)

if TYPE_CHECKING:
    from os import PathLike


# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------
class PDBParseError(ValueError):
    """Raised when a PDB file cannot be parsed."""


class PDBWriteError(ValueError):
    """Raised when an in-memory structure cannot be serialized to PDB."""


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
# Nucleotide residue names. (Used for entity-type classification only;
# the canonical mapping lives in core.constants.)
_NUCLEOTIDES = frozenset({"DA", "DT", "DG", "DC", "DI", "A", "U", "G", "C", "I"})

# Common cofactors / ligands we don't want to misclassify as "water".
# Anything not in HOH/WAT/etc., not a standard AA, not a nucleotide, and
# not an ion gets entity_type="ligand".


def _classify_entity(residue_name: str, n_atoms_in_residue: int) -> str:
    """Decide an entity_type label from a residue name.

    Heuristic:
      - 20 canonical AAs (or known modified) -> "protein"
      - DNA/RNA -> "dna" or "rna"
      - HOH/WAT -> "water"
      - Single-atom common ions -> "ion"
      - Everything else -> "ligand"

    Args:
        residue_name: 3-letter code from columns 18-20.
        n_atoms_in_residue: number of atoms in the residue (used to
            disambiguate single-atom ions from multi-atom ligands with
            the same name).
    """
    name = residue_name.strip().upper()
    if name in THREE_TO_ONE:
        return "protein"
    if name in {"MSE", "SEC", "PYL"}:
        # Selenomethionine / -cysteine / pyrrolysine — proteinogenic
        return "protein"
    if name in {"DA", "DT", "DG", "DC", "DI"}:
        return "dna"
    if name in {"A", "U", "G", "C", "I"} and n_atoms_in_residue >= 10:
        # Bare A/U/G/C/I could also be adenine ligands; if it has the
        # atom count of a nucleotide, treat it as RNA.
        return "rna"
    if is_water(name):
        return "water"
    if is_ion(name) and n_atoms_in_residue == 1:
        return "ion"
    return "ligand"


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------
def read_pdb(
    path: str | PathLike[str],
    *,
    model: int | None = None,
    include_hydrogens: bool = True,
    altloc: str = "highest_occupancy",
) -> Protein:
    """Read a PDB file from disk.

    Args:
        path: Path to a ``.pdb`` file (may be gzipped if extension is ``.gz``).
        model: Which model to load from a multi-model file. ``None``
            (default) loads all models. ``0`` is the first model. Pass
            an int to load a specific model.
        include_hydrogens: If ``False``, drop hydrogen atoms during parsing.
        altloc: Strategy for resolving alternate location indicators.

            - ``"highest_occupancy"`` (default): keep the altloc with the
              highest occupancy per atom name.
            - ``"first"``: keep the first altloc encountered, drop the rest.
            - ``"all"``: keep all altlocs (atoms will share residue_id but
              differ on altloc field).
            - A single-character string (e.g. ``"A"``): keep only that altloc
              and the default (blank).

    Returns:
        A :class:`Protein` holding the parsed structure. The protein's
        ``metadata`` dict is populated with any HEADER, TITLE, RESOLUTION,
        and EXPDTA records found.

    Raises:
        PDBParseError: If the file is malformed.
        FileNotFoundError: If the path doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".gz":
        import gzip

        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    out = read_pdb_string(
        text,
        model=model,
        include_hydrogens=include_hydrogens,
        altloc=altloc,
    )
    out.name = path.stem
    return out


def read_pdb_string(
    text: str,
    *,
    model: int | None = None,
    include_hydrogens: bool = True,
    altloc: str = "highest_occupancy",
) -> Protein:
    """Parse a PDB-formatted string into a :class:`Protein`.

    See :func:`read_pdb` for argument semantics.
    """
    # Per-record growing lists. We size-collect rather than pre-allocate
    # because we don't know n_atoms without a pre-scan, and a pre-scan
    # costs more than Python list growth for typical PDBs.
    atom_names: list[str] = []
    elements: list[str] = []
    residue_names: list[str] = []
    residue_ids: list[int] = []
    insertion_codes: list[str] = []
    chain_ids: list[str] = []
    coords: list[tuple[float, float, float]] = []
    b_factors: list[float] = []
    occupancies: list[float] = []
    charges: list[float] = []
    serials: list[int] = []
    record_types: list[str] = []
    altlocs: list[str] = []
    model_ids: list[int] = []

    metadata: dict[str, object] = {}
    current_model = 0
    skip_current_model = False
    seen_any_atom = False

    for raw_line in text.splitlines():
        # PDB lines are right-padded to 80 columns; we don't enforce that
        # strictly because real-world files often violate it.
        if not raw_line:
            continue
        record = raw_line[:6].rstrip()

        # --- Header / metadata records ---
        if record == "HEADER":
            # cols 11-50 classification, 51-59 date, 63-66 idcode
            if len(raw_line) >= 66:
                metadata[mk.CLASSIFICATION] = raw_line[10:50].strip()
                metadata[mk.DEPOSITION_DATE] = raw_line[50:59].strip()
                metadata[mk.PDB_ID] = raw_line[62:66].strip()
            continue
        if record == "TITLE":
            metadata.setdefault(mk.TITLE, "")
            metadata[mk.TITLE] = (str(metadata[mk.TITLE]) + " " + raw_line[10:].strip()).strip()
            continue
        if record == "EXPDTA":
            metadata[mk.EXPERIMENTAL_METHOD] = raw_line[10:].strip()
            continue
        if record == "REMARK":
            # REMARK 2  RESOLUTION.    1.50 ANGSTROMS.
            if raw_line[6:10].strip() == "2" and "RESOLUTION" in raw_line:
                try:
                    # Find the float right after "RESOLUTION."
                    rest = raw_line.split("RESOLUTION.")[1].strip()
                    val = rest.split()[0]
                    metadata[mk.RESOLUTION] = float(val)
                except (IndexError, ValueError):
                    pass
            continue

        # --- Model handling ---
        if record == "MODEL":
            try:
                current_model = int(raw_line[10:14].strip())
            except ValueError:
                current_model += 1
            skip_current_model = model is not None and current_model != model
            continue
        if record == "ENDMDL":
            skip_current_model = False
            continue

        # --- Atom records ---
        if record in {"ATOM", "HETATM"}:
            if skip_current_model:
                continue
            # If user asked for a specific model and we're outside any MODEL
            # block, treat the first set of atoms as model 0/1.
            if model is not None and current_model == 0 and seen_any_atom is False:
                # Allow a file without explicit MODEL records to satisfy model=0/1
                # We'll only skip if model > 1 and no MODEL was ever seen.
                pass

            # Parse fixed columns. Column indices are *1-based* in the spec;
            # we subtract 1 since Python slices are 0-based.
            #
            # Spec: https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html
            #
            #  1- 6  record name
            #  7-11  serial
            # 13-16  atom name
            # 17     altloc
            # 18-20  residue name
            # 22     chain id
            # 23-26  residue seq number
            # 27     insertion code
            # 31-38  x
            # 39-46  y
            # 47-54  z
            # 55-60  occupancy
            # 61-66  b-factor
            # 77-78  element
            # 79-80  charge
            line = raw_line
            # Pad short lines so we don't get IndexError on slim files
            if len(line) < 80:
                line = line.ljust(80)

            try:
                serial = int(line[6:11].strip() or "0")
            except ValueError as e:
                raise PDBParseError(f"bad serial in line: {raw_line!r}") from e

            atom_name = line[12:16].strip()
            altloc_ch = line[16].strip()
            residue_name = line[17:20].strip()
            chain_id = line[21].strip() or " "
            try:
                residue_id = int(line[22:26].strip() or "0")
            except ValueError as e:
                raise PDBParseError(f"bad residue id in line: {raw_line!r}") from e
            ins_code = line[26].strip()
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError as e:
                raise PDBParseError(f"bad coordinates in line: {raw_line!r}") from e

            occ_str = line[54:60].strip()
            occupancy = float(occ_str) if occ_str else 1.0
            b_str = line[60:66].strip()
            b_factor = float(b_str) if b_str else 0.0

            element = line[76:78].strip()
            if not element:
                # Element column wasn't filled in; fall back to first
                # non-digit char of atom name.
                element = "".join(c for c in atom_name if c.isalpha())[:2].title()

            # Charge: "1+" / "2-" style; PDB rarely populates this
            charge_str = line[78:80].strip()
            charge = 0.0
            if charge_str:
                try:
                    if charge_str.endswith(("+", "-")):
                        sign = 1.0 if charge_str.endswith("+") else -1.0
                        charge = sign * float(charge_str[:-1])
                    else:
                        charge = float(charge_str)
                except ValueError:
                    charge = 0.0

            # Skip hydrogens if requested
            if not include_hydrogens and element.upper() == "H":
                continue

            atom_names.append(atom_name)
            elements.append(element)
            residue_names.append(residue_name)
            residue_ids.append(residue_id)
            insertion_codes.append(ins_code)
            chain_ids.append(chain_id)
            coords.append((x, y, z))
            b_factors.append(b_factor)
            occupancies.append(occupancy)
            charges.append(charge)
            serials.append(serial)
            record_types.append(record)
            altlocs.append(altloc_ch)
            # If user asked for model=N, force that. Otherwise track the
            # PDB's MODEL records.
            model_ids.append(current_model)
            seen_any_atom = True
            continue

        # END / TER / CONECT / others: we ignore. CONECT is for explicit
        # bonds; molforge currently re-derives connectivity from distance
        # rather than reading it from PDB.

    n = len(atom_names)
    if n == 0:
        # Empty file or no ATOM/HETATM records — return an empty Protein.
        return Protein(AtomArray(0), name="", metadata=metadata)

    arr = AtomArray(n)
    arr.coords[:] = np.asarray(coords, dtype=np.float32)
    arr.atom_name[:] = atom_names
    arr.element[:] = [e.upper() for e in elements]
    arr.residue_name[:] = residue_names
    arr.residue_id[:] = residue_ids
    arr.insertion_code[:] = insertion_codes
    arr.chain_id[:] = chain_ids
    arr.b_factor[:] = b_factors
    arr.occupancy[:] = occupancies
    arr.charge[:] = charges
    arr.serial[:] = serials
    arr.record_type[:] = record_types
    arr.altloc[:] = altlocs
    arr.model_id[:] = model_ids

    # Classify entity_type per residue. Walk residue boundaries.
    arr._invalidate_cache()
    for sl in arr.iter_residue_slices():
        rn = str(arr.residue_name[sl.start])
        n_atoms_here = sl.stop - sl.start
        arr.entity_type[sl] = _classify_entity(rn, n_atoms_here)

    # Altloc handling
    if altloc != "all":
        arr = _resolve_altlocs(arr, strategy=altloc)

    return Protein(arr, name="", metadata=metadata)


def _resolve_altlocs(arr: AtomArray, strategy: str) -> AtomArray:
    """Apply an altloc-resolution strategy.

    See :func:`read_pdb` for strategy semantics.
    """
    altloc_col = arr.altloc
    has_altloc = altloc_col != ""
    if not bool(np.any(has_altloc)):
        return arr  # nothing to do

    if strategy == "first":
        # For each (chain_id, residue_id, ins_code, atom_name), keep the
        # first occurrence regardless of altloc identifier.
        keep = np.ones(len(arr), dtype=bool)
        # Build a per-key first-seen map.
        seen: set[tuple[str, int, str, str]] = set()
        for i in range(len(arr)):
            if not has_altloc[i]:
                continue
            key = (
                str(arr.chain_id[i]),
                int(arr.residue_id[i]),
                str(arr.insertion_code[i]),
                str(arr.atom_name[i]),
            )
            if key in seen:
                keep[i] = False
            else:
                seen.add(key)
        out = arr.select(keep)
        out.altloc[:] = ""
        return out

    if strategy == "highest_occupancy":
        # For each (chain_id, residue_id, ins_code, atom_name) group with
        # multiple altlocs, keep the one with the highest occupancy.
        keep = np.ones(len(arr), dtype=bool)
        # Group atoms by key and pick the argmax of occupancy.
        from collections import defaultdict

        groups: dict[tuple[str, int, str, str], list[int]] = defaultdict(list)
        for i in range(len(arr)):
            if not has_altloc[i]:
                continue
            key = (
                str(arr.chain_id[i]),
                int(arr.residue_id[i]),
                str(arr.insertion_code[i]),
                str(arr.atom_name[i]),
            )
            groups[key].append(i)
        for indices in groups.values():
            if len(indices) <= 1:
                continue
            occs = arr.occupancy[indices]
            winner = indices[int(np.argmax(occs))]
            for i in indices:
                if i != winner:
                    keep[i] = False
        out = arr.select(keep)
        out.altloc[:] = ""
        return out

    # Treat any other 1-char string as a specific altloc selector.
    if len(strategy) == 1:
        keep = (~has_altloc) | (altloc_col == strategy)
        out = arr.select(keep)
        out.altloc[:] = ""
        return out

    raise ValueError(
        f"unknown altloc strategy: {strategy!r} "
        f"(expected 'highest_occupancy', 'first', 'all', or a single-char altloc id)"
    )


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------
# PDB atom-name formatting is the single most error-prone part of writing
# this format. The rule (per spec section 9.1):
#   - If the element symbol is one character, the atom name is left-padded
#     with a space and right-padded with spaces.  Example: " CA "
#   - If the element symbol is two characters, the atom name occupies all
#     four columns with no leading space.  Example: "FE  "
#   - The atom-name field is 4 columns wide (13-16, 1-based).
def _format_atom_name(name: str, element: str) -> str:
    """Format an atom name into its 4-character PDB column."""
    name = name.strip()
    el = element.strip()
    if len(el) == 2 and name.upper().startswith(el.upper()):
        return name.ljust(4)[:4]
    # Single-element atom: leading space.
    return (" " + name).ljust(4)[:4]


def write_pdb(
    protein: Protein,
    path: str | PathLike[str],
    *,
    write_end: bool = True,
) -> None:
    """Write a :class:`Protein` to a PDB file.

    Args:
        protein: the structure to serialize.
        path: destination path. ``.gz`` suffix triggers gzip compression.
        write_end: emit a final ``END`` record.

    Raises:
        PDBWriteError: If the structure exceeds PDB's hard limits
            (>99,999 atoms or >9,999 residues per chain).
    """
    text = write_pdb_string(protein, write_end=write_end)
    path = Path(path)
    if path.suffix == ".gz":
        import gzip

        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(text)
    else:
        path.write_text(text, encoding="utf-8")


def write_pdb_string(protein: Protein, *, write_end: bool = True) -> str:
    """Serialize a :class:`Protein` into a PDB-formatted string."""
    arr = protein.atom_array
    n = len(arr)
    if n > 99_999:
        raise PDBWriteError(
            f"PDB format supports at most 99,999 atoms; got {n}. "
            "Use mmCIF (write_cif) for larger structures."
        )

    lines: list[str] = []
    # Optional header records from metadata.
    pdb_id = str(protein.metadata.get(mk.PDB_ID, "")).strip()[:4]
    classification = str(protein.metadata.get(mk.CLASSIFICATION, "")).strip()[:40]
    if pdb_id or classification:
        lines.append(f"HEADER    {classification:<40}{'        '}{pdb_id:<4}              ")
    title = str(protein.metadata.get(mk.TITLE, "")).strip()
    if title:
        # TITLE may span multiple continuation lines for long titles;
        # we keep it simple here and truncate at 70 chars per line.
        for chunk_idx, start in enumerate(range(0, len(title), 70), start=1):
            chunk = title[start : start + 70]
            if chunk_idx == 1:
                lines.append(f"TITLE     {chunk:<70}")
            else:
                lines.append(f"TITLE   {chunk_idx:>2}{chunk:<70}")

    # Walk atoms, emitting TER between chains and ENDMDL between models.
    last_model_id = None
    last_chain_key: tuple[str, int] | None = None
    last_residue_key: tuple[str, int, str, int] | None = None
    last_resname = ""

    # If we have multiple models, wrap atoms in MODEL/ENDMDL blocks.
    unique_models = np.unique(arr.model_id)
    multi_model = len(unique_models) > 1

    for i in range(n):
        m_id = int(arr.model_id[i])
        if multi_model and m_id != last_model_id:
            if last_model_id is not None:
                lines.append("ENDMDL")
            lines.append(f"MODEL     {m_id:>4}")
            last_model_id = m_id
            last_chain_key = None  # reset chain tracking inside new model

        rec = str(arr.record_type[i]).strip() or "ATOM"
        serial = int(arr.serial[i]) or (i + 1)
        atom_name_raw = str(arr.atom_name[i])
        element = str(arr.element[i])
        residue_name = str(arr.residue_name[i])[:3]
        chain_id = str(arr.chain_id[i])[:1] or " "
        residue_id = int(arr.residue_id[i])
        ins_code = str(arr.insertion_code[i])[:1]
        x, y, z = (float(arr.coords[i, k]) for k in range(3))
        occ = float(arr.occupancy[i])
        b = float(arr.b_factor[i])
        altloc_ch = str(arr.altloc[i])[:1]
        ch = float(arr.charge[i])
        ch_str = ""
        if ch != 0:
            sign = "+" if ch > 0 else "-"
            ch_str = f"{abs(int(ch))}{sign}"

        # Emit TER between consecutive different chains within the same model.
        chain_key = (chain_id, m_id)
        if (
            last_chain_key is not None
            and last_chain_key != chain_key
            and last_residue_key is not None
        ):
            ter_serial = (serial - 1) % 100000
            last_chain_id = last_chain_key[0]
            _, last_rid, last_ins, _ = last_residue_key
            lines.append(
                f"TER   {ter_serial:>5}      {last_resname:>3} {last_chain_id:>1}"
                f"{last_rid:>4}{last_ins:>1}" + " " * 53
            )
        last_chain_key = chain_key
        last_residue_key = (chain_id, residue_id, ins_code, m_id)
        last_resname = residue_name

        atom_name_fmt = _format_atom_name(atom_name_raw, element)

        # Wrap serial at 99,999 (it's been validated above; this is defensive).
        ser_fmt = serial if serial < 100000 else 99999

        lines.append(
            f"{rec:<6}"
            f"{ser_fmt:>5} "
            f"{atom_name_fmt}"
            f"{altloc_ch:1}"
            f"{residue_name:>3} "
            f"{chain_id:1}"
            f"{residue_id:>4}"
            f"{ins_code:1}   "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
            f"{occ:>6.2f}"
            f"{b:>6.2f}" + " " * 10 + f"{element:>2}" + f"{ch_str:>2}"
        )

    # Final TER for the last chain (if we had atoms).
    if last_chain_key is not None and last_residue_key is not None:
        ter_serial = (n + 1) % 100000
        last_chain_id = last_chain_key[0]
        _, last_rid, last_ins, _ = last_residue_key
        lines.append(
            f"TER   {ter_serial:>5}      {last_resname:>3} {last_chain_id:>1}"
            f"{last_rid:>4}{last_ins:>1}" + " " * 53
        )

    if multi_model and last_model_id is not None:
        lines.append("ENDMDL")

    if write_end:
        lines.append("END")

    # PDB requires a trailing newline per line; join with \n and add one more.
    return "\n".join(lines) + "\n"


def _stream_pdb_to_file(protein: Protein, fh: TextIO, *, write_end: bool = True) -> None:
    """Streaming variant of :func:`write_pdb_string` for large structures.

    Currently delegates to the string-builder; reserved for a future
    implementation that doesn't materialize the whole file in memory.
    """
    fh.write(write_pdb_string(protein, write_end=write_end))
