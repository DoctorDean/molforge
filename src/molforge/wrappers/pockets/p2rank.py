"""P2Rank — machine-learning pocket detection.

`P2Rank <https://github.com/rdk/p2rank>`_ (Krivák & Hoksza 2018) is a
template-free, machine-learning ligand-binding-site predictor. It scores
points on the solvent-accessible surface with a random forest trained on
physicochemical and geometric features, then clusters high-scoring points
into ranked pockets — consistently outperforming classical geometric
detectors like fpocket on benchmark recall.

P2Rank is a Java application (not pip-installable): users download it from
the project releases or install via conda / a package manager, and expose
the ``prank`` launcher on ``$PATH``. molforge calls ``prank predict -f
<input.pdb> -o <outdir>``, parses the ``*_predictions.csv`` it writes, and
returns :class:`molforge.docking.Pocket` objects — the same shape fpocket
returns, so the two detectors are drop-in alternatives:

.. code-block:: python

    from molforge.io import fetch
    from molforge.wrappers.pockets import detect_pockets_p2rank
    from molforge.wrappers.docking import Vina

    protein = fetch("1AKE")
    pockets = detect_pockets_p2rank(protein)
    print(f"{len(pockets)} pockets, top probability "
          f"{pockets[0].druggability:.2f}")

    result = Vina().dock(
        receptor=protein,
        ligand=ligand_smiles,
        center=tuple(pockets[0].center.tolist()),
        box_size=(20.0, 20.0, 20.0),
    )

Field mapping from P2Rank's CSV: ``score`` → :attr:`Pocket.score`,
``probability`` (the calibrated [0, 1] confidence that the pocket is a true
binding site) → :attr:`Pocket.druggability`, ``center_x/y/z`` →
:attr:`Pocket.center`, ``residue_ids`` → :attr:`Pocket.residues`. P2Rank
does not report a pocket volume, so :attr:`Pocket.volume` is ``None``; the
full CSV row is kept in ``metadata["descriptors"]``.

Like the fpocket wrapper, this exposes the common ``predict`` path, not
P2Rank's full CLI surface (custom models, config files, the conservative
vs. alternative model variants). Extending it is straightforward when a
concrete need surfaces.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from molforge.core import metadata_keys as mk
from molforge.core.provenance import Provenance
from molforge.docking import Pocket

if TYPE_CHECKING:
    from molforge.core import Protein


class P2RankNotInstalledError(RuntimeError):
    """Raised when the P2Rank ``prank`` launcher isn't on PATH (or wherever
    ``prank_executable`` points).

    P2Rank isn't pip-installable — it's a Java application users download
    from the project releases (or install via conda). The error message
    points at the install path so users don't have to grep for it.
    """


def detect_pockets_p2rank(
    protein: Protein,
    *,
    prank_executable: str = "prank",
    timeout: float = 300.0,
) -> list[Pocket]:
    """Detect candidate ligand-binding pockets with P2Rank.

    Runs ``prank predict`` against the protein, parses the prediction CSV,
    and returns a list of :class:`Pocket` ranked by P2Rank's score
    (best first).

    Args:
        protein: The :class:`molforge.core.Protein` to analyse. Written to
            a temporary PDB and passed via ``-f``.
        prank_executable: Name or full path of the P2Rank launcher.
            Defaults to ``"prank"`` (expected on ``$PATH``).
        timeout: Subprocess timeout in seconds. P2Rank's ML scoring is
            slower than a geometric detector — the default 300s is
            comfortable for normal-sized inputs while still forcing a hard
            failure rather than an indefinite hang.

    Returns:
        A list of :class:`Pocket` ranked by P2Rank's score (descending;
        pocket 0 is the best). Empty list when P2Rank finds no pockets.

    Raises:
        P2RankNotInstalledError: If ``prank_executable`` isn't on ``$PATH``
            (or wherever it points).
        RuntimeError: If P2Rank exits non-zero, times out, or produces no
            parseable output. The error echoes P2Rank's stderr where
            available.
    """
    if shutil.which(prank_executable) is None:
        raise P2RankNotInstalledError(
            f"P2Rank launcher not found: {prank_executable!r}. "
            "P2Rank isn't pip-installable — download it from "
            "https://github.com/rdk/p2rank/releases (or `conda install -c "
            "bioconda p2rank`) and ensure the `prank` launcher is on $PATH, "
            "or pass an explicit path via prank_executable=."
        )

    with tempfile.TemporaryDirectory(prefix="molforge_p2rank_") as tmp:
        tmp_dir = Path(tmp)
        input_pdb = tmp_dir / "structure.pdb"
        out_dir = tmp_dir / "p2rank_out"

        # Lazy import to keep top-of-module imports lean.
        from molforge.io import save

        save(protein, input_pdb)

        cmd = [prank_executable, "predict", "-f", str(input_pdb), "-o", str(out_dir)]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"P2Rank timed out after {timeout}s. Pass a larger timeout= for big systems."
            ) from e

        if result.returncode != 0:
            raise RuntimeError(
                f"P2Rank exited with code {result.returncode}. stderr: {result.stderr.strip()}"
            )

        csv_path = next(out_dir.glob("*_predictions.csv"), None)
        if csv_path is None:
            raise RuntimeError(
                f"P2Rank produced no *_predictions.csv in {out_dir}. "
                f"stdout: {result.stdout.strip()}"
            )

        return _parse_p2rank_predictions(
            csv_path.read_text(),
            protein=protein,
            prank_executable=prank_executable,
        )


# ----------------------------------------------------------------------
# Output parsing
# ----------------------------------------------------------------------

# P2Rank's <name>_predictions.csv is comma-separated with space-padded
# fields:
#   rank,   name,   score, probability, sas_points, surf_atoms,  center_x, ...
#      1, pocket1,  28.14,       0.812,        142,         56,     12.34, ...
# The residue_ids / surf_atom_ids columns hold *space*-separated lists, so
# a plain comma split keeps them intact as single fields; we strip padding
# per field.
_EXPECTED_COLUMNS = (
    "rank",
    "name",
    "score",
    "probability",
    "center_x",
    "center_y",
    "center_z",
    "residue_ids",
)


def _parse_p2rank_predictions(
    csv_text: str,
    *,
    protein: Protein,
    prank_executable: str,
) -> list[Pocket]:
    """Parse P2Rank's predictions CSV into a list of :class:`Pocket`."""
    rows = _read_csv_rows(csv_text)
    if not rows:
        # No pockets found is legitimate — return empty rather than raising.
        return []

    # Shared provenance across all pockets from this one detection call —
    # same pattern as the fpocket wrapper.
    parent = protein.metadata.get(mk.PROVENANCE)
    shared_provenance = Provenance.from_engine(
        engine="p2rank",
        parameters={"prank_executable": prank_executable},
        inputs={"protein": protein.name or "<Protein>"},
        parent=parent if isinstance(parent, Provenance) else None,
    )

    pockets: list[Pocket] = []
    for idx, row in enumerate(rows):
        center = np.array(
            [_coord(row.get(axis)) for axis in ("center_x", "center_y", "center_z")],
            dtype=np.float32,
        )
        pockets.append(
            Pocket(
                center=center,
                residues=_parse_residue_ids(row.get("residue_ids", "")),
                volume=None,  # P2Rank does not report pocket volume.
                score=_maybe_float(row.get("score")),
                druggability=_maybe_float(row.get("probability")),
                rank=idx,
                metadata={
                    mk.PROVENANCE: shared_provenance,
                    "engine": "p2rank",
                    "descriptors": row,
                },
            )
        )
    return pockets


def _read_csv_rows(csv_text: str) -> list[dict[str, str]]:
    """Parse the P2Rank CSV into a list of column→value dicts.

    Header and values are space-padded; fields are split on commas (the
    space-separated list columns contain no commas, so they survive intact).
    """
    lines = [ln for ln in csv_text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return []
    header = [h.strip() for h in lines[0].split(",")]
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        values = [v.strip() for v in line.split(",")]
        if len(values) != len(header):
            # Skip malformed rows rather than crashing the whole run.
            continue
        rows.append(dict(zip(header, values, strict=True)))
    return rows


def _parse_residue_ids(field: str) -> list[tuple[str, int, str]]:
    """Parse P2Rank's ``residue_ids`` (``"A_45 A_46 B_12"``) into
    ``(chain_id, residue_id, insertion_code)`` triples.

    Tokens that don't parse (unexpected shape, non-numeric residue number)
    are skipped — a stray token shouldn't sink the whole pocket.
    """
    out: list[tuple[str, int, str]] = []
    for token in field.split():
        parts = token.split("_")
        if len(parts) < 2:
            continue
        chain = parts[0]
        try:
            resid = int(parts[1])
        except ValueError:
            continue
        insertion = parts[2].strip() if len(parts) > 2 else ""
        out.append((chain, resid, insertion))
    return out


def _maybe_float(s: str | None) -> float | None:
    """Parse a string into a float, returning ``None`` on failure/missing."""
    if s is None or s == "":
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _coord(s: str | None) -> float:
    """Parse a coordinate to float, ``nan`` when missing/malformed.

    Distinct from :func:`_maybe_float` so a legitimate ``0.0`` coordinate is
    preserved rather than collapsing to ``nan``.
    """
    value = _maybe_float(s)
    return float("nan") if value is None else value
