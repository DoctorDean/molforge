"""fpocket — Voronoi-based pocket detection.

`fpocket <https://github.com/Discngine/fpocket>`_ is a fast, classical
pocket detector. Given a protein, it computes a Voronoi tessellation,
extracts "alpha spheres" tangent to four atoms each, clusters spheres
into candidate pockets, and ranks the pockets by an empirical
druggability score.

This module wraps the external ``fpocket`` binary; users install it
themselves via a system package or by building from source — there's
no Python package. molforge calls ``fpocket -f <input.pdb>``, parses
the output directory, and returns
:class:`molforge.docking.Pocket` objects.

The typical workflow:

.. code-block:: python

    from molforge.io import fetch
    from molforge.wrappers.pockets import detect_pockets
    from molforge.wrappers.docking import Vina

    protein = fetch("1AKE")
    pockets = detect_pockets(protein)
    print(f"{len(pockets)} pockets, top druggability "
          f"{pockets[0].druggability:.2f}")

    # Dock into the top pocket.
    result = Vina().dock(
        receptor=protein,
        ligand=ligand_smiles,
        center=tuple(pockets[0].center.tolist()),
        box_size=(20.0, 20.0, 20.0),
    )

What this wrapper does and doesn't do
-------------------------------------

It runs the basic ``fpocket -f <pdb>`` pipeline and parses the standard
output directory. The user can override the alpha-sphere filtering
parameters (``min_alpha_spheres``, ``min_volume``) and the timeout.

It does *not* expose fpocket's full command-line surface — e.g. the
chain-keep/drop flags, custom scoring functions, the ``mdpocket``
trajectory mode. Those are deliberately out of scope for v1;
extending them is straightforward when concrete user needs surface.
The wrapper's role is making the common case ergonomic, not
mirroring every fpocket flag.
"""

from __future__ import annotations

import re
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


class FpocketNotInstalledError(RuntimeError):
    """Raised when the fpocket binary isn't on PATH (or wherever
    ``fpocket_executable`` points to).

    fpocket isn't pip-installable — users install it through their
    system package manager or build from source. The error message
    points at the install path so users don't have to grep for it.
    """


def detect_pockets(
    protein: Protein,
    *,
    fpocket_executable: str = "fpocket",
    min_alpha_spheres: int | None = None,
    min_volume: float | None = None,
    timeout: float = 60.0,
) -> list[Pocket]:
    """Detect candidate ligand-binding pockets with fpocket.

    Runs the ``fpocket`` binary against the protein, parses the
    output, and returns a list of :class:`Pocket` ranked by
    fpocket's own scoring (best first).

    Args:
        protein: The :class:`molforge.core.Protein` to analyse.
            Written to a temporary PDB and passed via ``-f``.
        fpocket_executable: Name or full path of the fpocket binary.
            Defaults to ``"fpocket"`` (expected on ``$PATH``).
        min_alpha_spheres: If set, passed as ``-i N`` to fpocket —
            minimum number of alpha spheres a pocket must contain
            to appear in the results. fpocket's own default is 15;
            this argument lets callers tighten or relax that.
        min_volume: If set, post-filter: drop pockets whose volume
            is below this threshold (Å³). fpocket itself doesn't
            have a clean volume filter; we apply it after parsing.
        timeout: Subprocess timeout in seconds. Real proteins
            complete in well under a minute; the default 60s is
            comfortable for normal-sized inputs and forces a hard
            failure rather than indefinite hang for pathological
            cases.

    Returns:
        A list of :class:`Pocket` ranked by fpocket's score
        (descending; pocket 0 is the best by fpocket). Empty list
        when fpocket finds no pockets meeting its criteria.

    Raises:
        FpocketNotInstalledError: If ``fpocket_executable`` isn't
            on ``$PATH`` (or wherever it points).
        RuntimeError: If fpocket exits non-zero, times out, or
            produces no parseable output. The error message echoes
            fpocket's stderr where available.
    """
    # Resolve the binary first — friendlier error than the
    # subprocess "FileNotFoundError" with no context.
    if shutil.which(fpocket_executable) is None:
        raise FpocketNotInstalledError(
            f"fpocket binary not found: {fpocket_executable!r}. "
            "Install via your system package manager "
            "(brew install fpocket / apt install fpocket / "
            "build from https://github.com/Discngine/fpocket) "
            "and ensure it's on $PATH, or pass an explicit path "
            "via fpocket_executable=."
        )

    # fpocket writes its output as a sibling of the input file
    # named <stem>_out/. We use a temp dir so we don't clutter the
    # caller's working directory and so multiple calls don't
    # collide.
    with tempfile.TemporaryDirectory(prefix="molforge_fpocket_") as tmp:
        tmp_dir = Path(tmp)
        input_pdb = tmp_dir / "structure.pdb"

        # Lazy import to keep top-of-module imports lean.
        from molforge.io import save

        save(protein, input_pdb)

        cmd = [fpocket_executable, "-f", str(input_pdb)]
        if min_alpha_spheres is not None:
            cmd += ["-i", str(int(min_alpha_spheres))]

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
                f"fpocket timed out after {timeout}s. Pass a larger timeout= for big systems."
            ) from e

        if result.returncode != 0:
            raise RuntimeError(
                f"fpocket exited with code {result.returncode}. stderr: {result.stderr.strip()}"
            )

        out_dir = tmp_dir / "structure_out"
        if not out_dir.is_dir():
            raise RuntimeError(
                f"fpocket produced no output directory at {out_dir}. "
                f"stdout: {result.stdout.strip()}"
            )

        pockets = _parse_fpocket_output(
            out_dir,
            protein=protein,
            fpocket_executable=fpocket_executable,
            min_alpha_spheres=min_alpha_spheres,
            min_volume=min_volume,
        )

    # Post-filter by volume after parsing (fpocket doesn't have a
    # native --min-volume flag; doing it here keeps the wrapper
    # ergonomic without bloating the binary's args).
    if min_volume is not None:
        pockets = [p for p in pockets if p.volume is not None and p.volume >= float(min_volume)]
        # Re-index ranks after filtering.
        for i, p in enumerate(pockets):
            p.rank = i

    return pockets


# ----------------------------------------------------------------------
# Output parsing
# ----------------------------------------------------------------------


# fpocket's info.txt has blocks like:
#   Pocket 1 :
#       Score :                  0.490
#       Druggability Score :     0.019
#       Number of Alpha Spheres : 21
#       Volume :                 270.934
#       ...
# Each pocket starts with "Pocket N :" and contains key-colon-value
# lines. We don't parse every descriptor (there are ~20); we extract
# the headline ones and stash the rest in metadata as a single block
# of text for users who want to grep it.

_POCKET_HEADER_RE = re.compile(r"^Pocket\s+(\d+)\s*:\s*$")
_KEY_VAL_RE = re.compile(r"^\s*([^:]+?)\s*:\s*(.+?)\s*$")


def _parse_fpocket_output(
    out_dir: Path,
    *,
    protein: Protein,
    fpocket_executable: str,
    min_alpha_spheres: int | None,
    min_volume: float | None,
) -> list[Pocket]:
    """Parse fpocket's output directory into a list of Pocket.

    fpocket writes ``<stem>_info.txt`` (per-pocket descriptor blocks)
    plus per-pocket ``pockets/pocketN_vert.pqr`` (alpha-sphere centres
    as PQR coords) and ``pockets/pocketN_atm.pdb`` (lining-residue
    atoms). We use the info.txt for descriptors, vert.pqr for the
    pocket centre, and atm.pdb for the lining residues.
    """
    info_txt = next(out_dir.glob("*_info.txt"), None)
    if info_txt is None:
        raise RuntimeError(f"fpocket output missing *_info.txt in {out_dir}.")

    pocket_descriptors = _parse_info_txt(info_txt.read_text())
    if not pocket_descriptors:
        # fpocket found nothing meeting its criteria. Empty result is
        # legitimate — return an empty list rather than raising.
        return []

    # Lazy imports for the per-pocket file parsers. read_pqr lives
    # in the submodule rather than being re-exported through
    # ``molforge.io``; that's an oversight in the io package's
    # __init__ rather than a wrapper-level decision (the other
    # format readers SDF / MOL2 / PDBQT have the same gap and will
    # be re-exported in a follow-up commit). Using the deep import
    # here keeps this commit focused on fpocket.
    from molforge.io.pdb import read_pdb
    from molforge.io.pqr import read_pqr

    pockets_dir = out_dir / "pockets"

    # Shared provenance across all pockets from this one detection
    # call. Each pocket's call-site arguments are identical; only
    # the per-pocket descriptors differ. Same pattern as
    # ProteinMPNN sharing one Provenance across its designs.
    shared_provenance = Provenance.from_engine(
        engine="fpocket",
        parameters={
            "fpocket_executable": fpocket_executable,
            "min_alpha_spheres": min_alpha_spheres,
            "min_volume": min_volume,
        },
        inputs={"protein": protein.name or "<Protein>"},
        parent=(
            protein.metadata.get(mk.PROVENANCE)
            if isinstance(protein.metadata.get(mk.PROVENANCE), Provenance)
            else None
        ),
    )

    pockets: list[Pocket] = []
    for idx, descriptors in enumerate(pocket_descriptors):
        # fpocket numbers pockets from 1 in info.txt but its files
        # are 0-indexed (pocket0_vert.pqr corresponds to "Pocket 1").
        vert_pqr = pockets_dir / f"pocket{idx}_vert.pqr"
        atm_pdb = pockets_dir / f"pocket{idx}_atm.pdb"

        if vert_pqr.is_file():
            try:
                vert_protein = read_pqr(vert_pqr)
                center = np.asarray(
                    vert_protein.atom_array.coords.mean(axis=0),
                    dtype=np.float32,
                )
            except Exception:
                # If the PQR is malformed (or empty), fall back to
                # NaN-coords — the user can detect the failure
                # without us crashing the whole detection run.
                center = np.full(3, np.nan, dtype=np.float32)
        else:
            center = np.full(3, np.nan, dtype=np.float32)

        residues: list[tuple[str, int, str]] = []
        if atm_pdb.is_file():
            try:
                atm_protein = read_pdb(atm_pdb)
                seen: set[tuple[str, int, str]] = set()
                arr = atm_protein.atom_array
                for i in range(arr.n_atoms):
                    key = (
                        str(arr.chain_id[i]),
                        int(arr.residue_id[i]),
                        str(arr.insertion_code[i]).strip(),
                    )
                    if key not in seen:
                        seen.add(key)
                        residues.append(key)
            except Exception:
                # Don't fail the whole detection on a single
                # malformed per-pocket file.
                residues = []

        # Extract the headline descriptors. Volume key reads
        # "Volume :", druggability is "Druggability Score :",
        # main score is "Score :". The rest goes into metadata
        # for users who want to grep.
        volume = _maybe_float(descriptors.get("Volume"))
        score = _maybe_float(descriptors.get("Score"))
        drugg = _maybe_float(descriptors.get("Druggability Score"))

        pockets.append(
            Pocket(
                center=center,
                residues=residues,
                volume=volume,
                score=score,
                druggability=drugg,
                rank=idx,
                metadata={
                    mk.PROVENANCE: shared_provenance,
                    "engine": "fpocket",
                    # Keep the full descriptor dict for users who
                    # want polar SASA, hydrophobicity, etc.
                    "descriptors": descriptors,
                },
            )
        )

    return pockets


def _parse_info_txt(text: str) -> list[dict[str, str]]:
    """Parse fpocket's *_info.txt into a list of descriptor dicts.

    Returns one dict per pocket, in fpocket's reported order (which
    is best-first by fpocket's own score).

    The format is forgiving — fpocket's whitespace varies, so we
    don't pin column positions. The shape is essentially INI-without
    section headers, segmented by "Pocket N :" lines.
    """
    pockets: list[dict[str, str]] = []
    current: dict[str, str] | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        # New pocket block.
        if _POCKET_HEADER_RE.match(line.strip()):
            if current is not None:
                pockets.append(current)
            current = {}
            continue
        # Key:value line inside a block.
        if current is not None:
            m = _KEY_VAL_RE.match(line)
            if m:
                key = m.group(1).strip()
                value = m.group(2).strip()
                current[key] = value
    if current is not None:
        pockets.append(current)

    return pockets


def _maybe_float(s: str | None) -> float | None:
    """Parse a string into a float, returning ``None`` on failure
    or missing input.

    fpocket's info.txt is mostly well-formed but the wrapper should
    never crash on a stray non-numeric value — it's a side effect
    of running on an unusual structure, not a bug in either tool.
    """
    if s is None:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None
