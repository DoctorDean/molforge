"""mmCIF / PDBx file format reader and writer.

[mmCIF](https://mmcif.wwpdb.org/) is the modern wwPDB format. Unlike
classic PDB it's a structured key-value format with no column-width
limits, making it the right choice for structures that exceed PDB's
hard caps (>99,999 atoms or >9,999 residues per chain), cryo-EM
depositions, and modern AlphaFold output.

This parser handles the subset of mmCIF that molforge needs in
practice: the ``_atom_site`` loop (the atomic coordinate data) plus a
small set of header / metadata fields (``_entry.id``, ``_struct.title``,
``_exptl.method``, ``_refine.ls_d_res_high`` for resolution).

It does **not** implement a full PDBx/mmCIF Dictionary parser — that's a
genuinely large undertaking (gemmi or biotite are the canonical full
parsers). What molforge implements is enough to round-trip every
structure in the PDB through the canonical ``AtomArray``, which is what
99% of users need.

For files that use exotic mmCIF features (multi-block files, semicolon
text fields with embedded newlines, save-frames, dictionary references),
prefer ``gemmi.read_structure(path)`` and convert via the conversion
helpers (planned).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from molforge.core import AtomArray, Protein
from molforge.core import metadata_keys as mk

if TYPE_CHECKING:
    from os import PathLike


class CIFParseError(ValueError):
    """Raised when an mmCIF file cannot be parsed."""


class CIFWriteError(ValueError):
    """Raised when an in-memory structure cannot be serialized to mmCIF."""


# ----------------------------------------------------------------------
# Tokenizer
# ----------------------------------------------------------------------
def _tokenize(text: str):  # type: ignore[no-untyped-def]
    """Yield mmCIF tokens.

    The mmCIF token grammar (simplified for what molforge needs):
      - ``#`` to end of line is a comment.
      - Whitespace separates tokens.
      - A token starting with ``"`` or ``'`` is a quoted string (closing
        quote terminates it).
      - A multi-line semicolon-bounded text field starts at column 0
        with ``;`` and ends with another ``;`` at column 0. The body
        is the concatenated lines between them.
      - Bare ``.`` and ``?`` mean "not applicable" / "unknown" — we
        emit them as literal strings; callers decide how to interpret.
      - Everything else is a bare token, terminated by whitespace.
    """
    lines = text.splitlines()
    n_lines = len(lines)
    i = 0
    while i < n_lines:
        line = lines[i]
        # Multi-line text field: starts with ; at col 0
        if line.startswith(";"):
            buf = [line[1:]]
            i += 1
            while i < n_lines and not lines[i].startswith(";"):
                buf.append(lines[i])
                i += 1
            # Skip the terminating `;` line
            i += 1
            yield "\n".join(buf).strip()
            continue
        # Strip trailing comment
        if "#" in line:
            # mmCIF: # starts a comment only at start of a token (whitespace before)
            # — we use the simpler rule: # outside quotes ends the line.
            in_quote = None
            for j, ch in enumerate(line):
                if in_quote:
                    if ch == in_quote:
                        in_quote = None
                elif ch in ('"', "'"):
                    in_quote = ch
                elif ch == "#":
                    line = line[:j]
                    break
        # Now tokenize the trimmed line
        pos = 0
        n_chars = len(line)
        while pos < n_chars:
            ch = line[pos]
            if ch.isspace():
                pos += 1
                continue
            if ch in ('"', "'"):
                # Quoted string: find matching closing quote followed by
                # whitespace or end of line (CIF rule).
                quote = ch
                start = pos + 1
                end = start
                while end < n_chars:
                    if line[end] == quote and (end + 1 == n_chars or line[end + 1].isspace()):
                        break
                    end += 1
                yield line[start:end]
                pos = end + 1
            else:
                # Bare token until whitespace
                start = pos
                while pos < n_chars and not line[pos].isspace():
                    pos += 1
                yield line[start:pos]
        i += 1


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------
def read_cif(
    path: str | PathLike[str],
    *,
    include_hydrogens: bool = True,
    altloc: str = "highest_occupancy",
) -> Protein:
    """Read an mmCIF / PDBx file from disk.

    Args:
        path: Path to a ``.cif`` or ``.mmcif`` file. ``.gz`` extension
            triggers gzip decompression.
        include_hydrogens: If False, drop hydrogen atoms during parsing.
        altloc: Altloc-resolution strategy (same as :func:`read_pdb`):
            ``"highest_occupancy"``, ``"first"``, ``"all"``, or a single
            alternate-location identifier (e.g. ``"A"``).

    Returns:
        A :class:`Protein` with the parsed structure. ``metadata`` is
        populated with ``pdb_id``, ``title``, ``experimental_method``,
        and ``resolution`` where available.

    Raises:
        CIFParseError: If the file is malformed or has no ``_atom_site``
            loop.
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
    out = read_cif_string(
        text,
        include_hydrogens=include_hydrogens,
        altloc=altloc,
    )
    if not out.name:
        out.name = path.stem
    return out


# The atom-site columns we care about, in the order the mmCIF dictionary
# defines them. Files may include more (or fewer); we extract whichever
# are present.
_ATOM_SITE_FIELDS = (
    "group_PDB",
    "id",
    "type_symbol",
    "label_atom_id",
    "label_alt_id",
    "label_comp_id",
    "label_asym_id",
    "label_entity_id",
    "label_seq_id",
    "pdbx_PDB_ins_code",
    "Cartn_x",
    "Cartn_y",
    "Cartn_z",
    "occupancy",
    "B_iso_or_equiv",
    "pdbx_formal_charge",
    "auth_seq_id",
    "auth_comp_id",
    "auth_asym_id",
    "auth_atom_id",
    "pdbx_PDB_model_num",
)


def read_cif_string(
    text: str,
    *,
    include_hydrogens: bool = True,
    altloc: str = "highest_occupancy",
) -> Protein:
    """Parse mmCIF-formatted text into a :class:`Protein`.

    See :func:`read_cif` for argument semantics.
    """
    tokens = list(_tokenize(text))
    if not tokens:
        return Protein(AtomArray(0))

    metadata: dict[str, object] = {}
    atom_rows: list[list[str]] = []
    atom_columns: list[str] = []

    pos = 0
    n = len(tokens)

    # data_<id> gives the block name; we hold it separately from
    # metadata[pdb_id] until parsing is done. The conflict-resolution
    # rule (applied after parse) is:
    #   - If _entry.id was seen with a real value, that wins.
    #   - If _entry.id was the "." sentinel, metadata[pdb_id] stays absent.
    #   - If _entry.id wasn't seen at all, the block name is used.
    # This makes the round-trip stable for proteins that had no real
    # pdb_id to begin with.
    block_data_id: str | None = None
    saw_entry_sentinel = False

    while pos < n:
        tok = tokens[pos]

        # data_<block> — first block header. We don't currently support
        # multi-block files (we'd return only the first block's atoms).
        if tok.startswith("data_"):
            candidate = tok[5:].strip()
            if candidate and candidate.lower() != "unknown":
                block_data_id = candidate
            pos += 1
            continue

        # Simple key/value: `_category.item value`
        if tok.startswith("_") and (pos + 1 < n) and not tokens[pos + 1].startswith("_"):
            # Could also be a loop header; check next token isn't another header.
            # In practice we hit the loop_ branch first because loop_ comes before its headers.
            key = tok
            value = tokens[pos + 1]
            if key == "_entry.id" and value in (".", "?"):
                # The writer's sentinel meaning "no real pdb_id"; record
                # the fact so we don't fall back to the block name.
                saw_entry_sentinel = True
            else:
                _maybe_capture_metadata(key, value, metadata)
            pos += 2
            continue

        # loop_ block
        if tok == "loop_":
            pos += 1
            cols: list[str] = []
            while pos < n and tokens[pos].startswith("_"):
                cols.append(tokens[pos])
                pos += 1
            # Detect atom_site loop
            if any(c.startswith("_atom_site.") for c in cols):
                atom_columns = [c.removeprefix("_atom_site.") for c in cols]
                n_cols = len(cols)
                while pos < n and not tokens[pos].startswith("_") and tokens[pos] != "loop_":
                    if tokens[pos].startswith("data_") or tokens[pos] == "stop_":
                        break
                    if pos + n_cols > n:
                        raise CIFParseError("atom_site loop truncated before final row")
                    row = tokens[pos : pos + n_cols]
                    atom_rows.append(row)
                    pos += n_cols
                continue
            # Other loops: skip their data rows (until next _ token or loop_)
            n_cols = len(cols)
            while pos < n and not tokens[pos].startswith("_") and tokens[pos] != "loop_":
                if tokens[pos].startswith("data_"):
                    break
                if pos + n_cols > n:
                    break
                pos += n_cols
            continue

        # Unknown / unhandled token — advance.
        pos += 1

    # Reconcile pdb_id: _entry.id (when present and non-sentinel) wins
    # over the block name; the block name is only used when no
    # _entry.id was provided. The "." sentinel explicitly suppresses
    # the block-name fallback so a Protein with no real pdb_id
    # round-trips with metadata[pdb_id] still absent.
    if mk.PDB_ID not in metadata and block_data_id and not saw_entry_sentinel:
        metadata[mk.PDB_ID] = block_data_id

    if not atom_rows:
        return Protein(AtomArray(0), metadata=metadata)

    return _atom_site_rows_to_protein(
        atom_rows,
        atom_columns,
        metadata=metadata,
        include_hydrogens=include_hydrogens,
        altloc=altloc,
    )


def _maybe_capture_metadata(key: str, value: str, metadata: dict[str, object]) -> None:
    """Record header-level metadata fields we care about."""
    if value in (".", "?"):
        return
    mapping = {
        "_entry.id": mk.PDB_ID,
        "_struct.title": mk.TITLE,
        "_struct_keywords.text": mk.CLASSIFICATION,
        "_exptl.method": mk.EXPERIMENTAL_METHOD,
        "_pdbx_database_status.recvd_initial_deposition_date": mk.DEPOSITION_DATE,
        "_refine.ls_d_res_high": mk.RESOLUTION,
        "_reflns.d_resolution_high": mk.RESOLUTION,
    }
    if key in mapping:
        target = mapping[key]
        if target == mk.RESOLUTION:
            import contextlib

            with contextlib.suppress(ValueError):
                metadata[target] = float(value)
        else:
            metadata[target] = value


def _atom_site_rows_to_protein(
    rows: list[list[str]],
    columns: list[str],
    *,
    metadata: dict[str, object],
    include_hydrogens: bool,
    altloc: str,
) -> Protein:
    """Convert tokenized atom_site rows into a Protein."""
    # Build a column index for fast lookup; tolerate missing optional cols.
    col_idx = {name: i for i, name in enumerate(columns)}

    def col(name: str, default: str = "") -> int | None:
        return col_idx.get(name)

    # The mmCIF dictionary distinguishes label_* (canonical) from auth_*
    # (author-assigned, what PDB uses). For round-trip compatibility with
    # PDB we prefer auth_* when present, falling back to label_*.
    serial_i = col_idx.get("id")
    elem_i = col_idx.get("type_symbol")
    name_i = col_idx.get("auth_atom_id") or col_idx.get("label_atom_id")
    altloc_i = col_idx.get("label_alt_id")
    resname_i = col_idx.get("auth_comp_id") or col_idx.get("label_comp_id")
    chain_i = col_idx.get("auth_asym_id") or col_idx.get("label_asym_id")
    resid_i = col_idx.get("auth_seq_id") or col_idx.get("label_seq_id")
    ins_i = col_idx.get("pdbx_PDB_ins_code")
    x_i, y_i, z_i = col_idx.get("Cartn_x"), col_idx.get("Cartn_y"), col_idx.get("Cartn_z")
    occ_i = col_idx.get("occupancy")
    b_i = col_idx.get("B_iso_or_equiv")
    charge_i = col_idx.get("pdbx_formal_charge")
    rec_i = col_idx.get("group_PDB")
    model_i = col_idx.get("pdbx_PDB_model_num")

    required = {
        "id": serial_i,
        "type_symbol": elem_i,
        "Cartn_x": x_i,
        "Cartn_y": y_i,
        "Cartn_z": z_i,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise CIFParseError(
            f"atom_site loop is missing required columns: {missing}. "
            "molforge needs at minimum id, type_symbol, Cartn_x/y/z."
        )

    def _get(row: list[str], i: int | None, default: str = "") -> str:
        if i is None:
            return default
        v = row[i]
        return default if v in (".", "?") else v

    # First pass: drop hydrogens here if requested (saves array allocation).
    kept_rows: list[list[str]] = []
    for row in rows:
        elem = _get(row, elem_i).upper()
        if not include_hydrogens and elem == "H":
            continue
        kept_rows.append(row)

    n_atoms = len(kept_rows)
    if n_atoms == 0:
        return Protein(AtomArray(0), metadata=metadata)

    arr = AtomArray(n_atoms)

    for i, row in enumerate(kept_rows):
        arr.coords[i] = (
            float(_get(row, x_i, "0")),
            float(_get(row, y_i, "0")),
            float(_get(row, z_i, "0")),
        )
        arr.element[i] = _get(row, elem_i).upper()
        # mmCIF often quotes atom names like "'CA'" — already stripped.
        arr.atom_name[i] = _get(row, name_i)[:4]
        arr.residue_name[i] = _get(row, resname_i)[:3]
        try:
            arr.residue_id[i] = int(_get(row, resid_i, "0") or "0")
        except ValueError as e:
            raise CIFParseError(f"bad residue id in row: {row!r}") from e
        arr.insertion_code[i] = _get(row, ins_i)[:1]
        arr.chain_id[i] = _get(row, chain_i)[:4]
        try:
            arr.b_factor[i] = float(_get(row, b_i, "0") or 0)
        except ValueError:
            arr.b_factor[i] = 0.0
        try:
            arr.occupancy[i] = float(_get(row, occ_i, "1") or 1.0)
        except ValueError:
            arr.occupancy[i] = 1.0
        try:
            arr.charge[i] = float(_get(row, charge_i, "0") or 0)
        except ValueError:
            arr.charge[i] = 0.0
        try:
            arr.serial[i] = int(_get(row, serial_i, str(i + 1)) or i + 1)
        except ValueError:
            arr.serial[i] = i + 1
        rec = _get(row, rec_i, "ATOM")
        arr.record_type[i] = rec if rec else "ATOM"
        arr.altloc[i] = _get(row, altloc_i)
        try:
            # Default to 0 to match read_pdb's convention for files
            # without explicit MODEL records — keeps the PDB <-> CIF
            # round-trip stable.
            arr.model_id[i] = int(_get(row, model_i, "0") or 0)
        except ValueError:
            arr.model_id[i] = 0

    # Classify entity_type per residue, same logic as the PDB parser.
    from molforge.io.pdb import _classify_entity

    arr._invalidate_cache()
    for sl in arr.iter_residue_slices():
        rn = str(arr.residue_name[sl.start])
        n_here = sl.stop - sl.start
        arr.entity_type[sl] = _classify_entity(rn, n_here)

    # Apply altloc strategy
    from molforge.io.pdb import _resolve_altlocs

    if altloc != "all":
        arr = _resolve_altlocs(arr, strategy=altloc)

    return Protein(arr, name=str(metadata.get(mk.PDB_ID, "")), metadata=metadata)


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------
def write_cif(
    protein: Protein,
    path: str | PathLike[str],
) -> None:
    """Write a :class:`Protein` to an mmCIF file."""
    text = write_cif_string(protein)
    path = Path(path)
    if path.suffix == ".gz":
        import gzip

        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(text)
    else:
        path.write_text(text, encoding="utf-8")


def write_cif_string(protein: Protein) -> str:
    """Serialize a :class:`Protein` as mmCIF text.

    Produces a compact CIF with a ``data_<id>`` header, the structure's
    metadata (where present), and a complete ``_atom_site`` loop. Round-trips
    cleanly through :func:`read_cif_string`.
    """
    arr = protein.atom_array
    # Pick a single identifier and use it for BOTH the data_<id>
    # block header AND _entry.id. If the two disagree, the reader's
    # later _entry.id wins and overwrites the block-derived pdb_id,
    # which silently changes Protein.name on round-trip. Resolving
    # both to one value keeps the identity stable.
    #
    # Preference order:
    #   1. metadata[PDB_ID] — the explicit identifier when one exists
    #      (set by read_pdb from the HEADER record, by read_cif from
    #      _entry.id, etc.).
    #   2. protein.name — a fallback when no metadata identifier is
    #      set (used by tooling that constructs Proteins by hand).
    #   3. "molforge" — last-ditch sentinel.
    identifier = str(protein.metadata.get(mk.PDB_ID, "")).strip()
    has_pdb_id = bool(identifier)
    if not identifier:
        identifier = (protein.name or "").strip()
    if not identifier:
        identifier = "molforge"
    # Block IDs must not contain whitespace; replace any with
    # underscore. The _entry.id below preserves the original value
    # (quoted if necessary) so a pdb_id containing whitespace —
    # which read_pdb tolerates from malformed HEADER lines — survives
    # the round-trip in metadata even though the block name can't.
    block_id = "".join(c if not c.isspace() else "_" for c in identifier)

    lines: list[str] = [f"data_{block_id}", "#"]
    # Emit _entry.id only when we have a real pdb_id metadata value.
    # When we fell back to protein.name or "molforge", emitting the
    # sentinel "." tells the reader "no _entry.id available" — so
    # the round-trip doesn't manufacture a pdb_id from nothing.
    if has_pdb_id:
        # Quote if the identifier contains characters that would
        # otherwise split tokens (whitespace).
        entry_token = f"'{identifier}'" if any(c.isspace() for c in identifier) else identifier
        lines.append(f"_entry.id  {entry_token}")
    else:
        lines.append("_entry.id  .")
    title = str(protein.metadata.get(mk.TITLE, "")).strip()
    if title:
        # Quote the title to safely include spaces.
        lines.append(f"_struct.title  '{title}'")
    # Classification (the PDB HEADER's classification field) lives in
    # _struct_keywords.text in mmCIF. Always quote to safely include
    # spaces — most classifications are multi-word ("OXIDOREDUCTASE").
    classification = str(protein.metadata.get(mk.CLASSIFICATION, "")).strip()
    if classification:
        lines.append(f"_struct_keywords.text  '{classification}'")
    method = str(protein.metadata.get(mk.EXPERIMENTAL_METHOD, "")).strip()
    if method:
        lines.append(f"_exptl.method  '{method}'")
    # Deposition date round-trips through
    # _pdbx_database_status.recvd_initial_deposition_date — the
    # standard mmCIF home for the date that PDB stores in cols 51-59
    # of the HEADER record.
    dep_date = str(protein.metadata.get(mk.DEPOSITION_DATE, "")).strip()
    if dep_date:
        lines.append(f"_pdbx_database_status.recvd_initial_deposition_date  '{dep_date}'")
    resolution = protein.metadata.get(mk.RESOLUTION)
    if isinstance(resolution, (int, float)):
        lines.append(f"_refine.ls_d_res_high  {float(resolution):.2f}")
    lines.append("#")

    # _atom_site loop
    lines.append("loop_")
    headers = [
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.pdbx_formal_charge",
        "_atom_site.auth_seq_id",
        "_atom_site.auth_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_atom_id",
        "_atom_site.pdbx_PDB_model_num",
    ]
    lines.extend(headers)

    def _val(s: str) -> str:
        """Quote if the string contains whitespace, replace empty with `.`."""
        if not s:
            return "."
        if any(c.isspace() for c in s) or s in (".", "?"):
            return f"'{s}'"
        return s

    for i in range(len(arr)):
        rec = str(arr.record_type[i]) or "ATOM"
        # serial is 1-based in PDB convention but the AtomArray may
        # legitimately carry 0 (synthetic data); only synthesize a
        # default when serial is non-positive.
        raw_serial = int(arr.serial[i])
        serial = raw_serial if raw_serial > 0 else (i + 1)
        elem = str(arr.element[i]) or "X"
        atom_name = str(arr.atom_name[i])
        altloc = str(arr.altloc[i]) or "."
        resname = str(arr.residue_name[i])
        chain = str(arr.chain_id[i])
        resid = int(arr.residue_id[i])
        ins = str(arr.insertion_code[i]) or "?"
        x, y, z = (float(arr.coords[i, k]) for k in range(3))
        occ = float(arr.occupancy[i])
        b = float(arr.b_factor[i])
        charge = float(arr.charge[i])
        # Charge can be a partial (non-integer) value when the structure
        # came from PDBQT, PQR, or a force-field-typed source. Emit
        # 4-decimal precision; only fall back to "?" when truly zero
        # (the AtomArray default) so the absence of charge information
        # round-trips as "absent" rather than as 0.0000.
        charge_str = f"{charge:.4f}" if charge != 0.0 else "?"
        # model_id is preserved verbatim, including 0. molforge's
        # read_pdb uses model_id=0 as the implicit value for files
        # without MODEL records — this is non-canonical relative to
        # mmCIF (which is 1-based), but emitting it verbatim is the
        # only way the CIF round-trip preserves the in-memory state.
        # A future change to read_pdb's convention would let this
        # become a strict 1-based emit.
        model = int(arr.model_id[i])

        lines.append(
            " ".join(
                [
                    rec,
                    str(serial),
                    elem,
                    _val(atom_name),
                    altloc,
                    resname,
                    _val(chain),
                    str(resid),
                    ins,
                    f"{x:.3f}",
                    f"{y:.3f}",
                    f"{z:.3f}",
                    f"{occ:.2f}",
                    f"{b:.2f}",
                    charge_str,
                    str(resid),
                    resname,
                    _val(chain),
                    _val(atom_name),
                    str(model),
                ]
            )
        )
    lines.append("#")
    return "\n".join(lines) + "\n"
