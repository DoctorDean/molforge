"""AlphaFold prediction helpers.

AlphaFold (and the related AlphaFold-Multimer / ColabFold / ESMFold)
emits structures in standard PDB format, but with two important
peculiarities:

1. The per-atom confidence score (**pLDDT**) is encoded in the B-factor
   column. molforge surfaces this as a first-class
   ``protein.metadata["plddt"]`` field rather than leaving it confusingly
   labeled as "b_factor".
2. The HEADER record typically begins with ``"ALPHAFOLD"`` or contains
   ``"PREDICTED MODEL"``. We use this to detect AlphaFold structures
   automatically, but you can also force the interpretation with
   :func:`load_alphafold`.

Per-residue pLDDT is computed as the mean across atoms in the residue
(equivalent to the per-residue pLDDT in AlphaFold's CIF output, since
all atoms in a residue share the same value).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from molforge.io.pdb import read_pdb

if TYPE_CHECKING:
    from os import PathLike

    from molforge.core import Protein


def is_alphafold_pdb(text_or_path: str | PathLike[str]) -> bool:
    """Detect whether a PDB file/string is an AlphaFold prediction.

    Heuristic: looks for ``ALPHAFOLD``, ``PREDICTED MODEL``, or ``ESMFOLD``
    in the first 100 lines of HEADER / TITLE / REMARK records.
    """
    # Distinguish "this is the literal contents" from "this is a path".
    if isinstance(text_or_path, str) and "\n" in text_or_path:
        text = text_or_path
    else:
        p = Path(text_or_path)
        if not p.exists():
            return False
        text = p.read_text(encoding="utf-8", errors="replace")

    needles = ("ALPHAFOLD", "PREDICTED MODEL", "ESMFOLD", "COLABFOLD")
    for i, line in enumerate(text.splitlines()):
        if i > 100:
            break
        if line.startswith(("HEADER", "TITLE", "REMARK")):
            upper = line.upper()
            if any(n in upper for n in needles):
                return True
    return False


def load_alphafold(path: str | PathLike[str]) -> Protein:
    """Load an AlphaFold prediction, exposing pLDDT as metadata.

    The protein is read via :func:`molforge.io.read_pdb`, then its
    ``metadata`` is populated with confidence information under two
    sets of keys:

    - **Uniform folding-engine keys** (preferred) — the same keys
      every molforge folding-engine wrapper sets, so downstream code
      can read confidence without caring which engine ran:
      ``confidence_per_atom``, ``confidence_per_residue``,
      ``mean_confidence``, and ``engine`` (= ``"AlphaFold"``).
    - **Legacy AlphaFold-specific keys** (retained for backward
      compatibility): ``plddt``, ``plddt_per_residue``,
      ``mean_plddt``, ``source`` (= ``"alphafold"``).

    The two sets carry the same values; new code should prefer the
    uniform keys. See :mod:`molforge.core.metadata_keys` for the
    documented vocabulary.

    The B-factor column is left intact for compatibility with
    downstream tools that still expect to find pLDDT there.
    """
    from molforge.core import metadata_keys as mk
    from molforge.core.provenance import Provenance

    protein = read_pdb(path)
    arr = protein.atom_array
    plddt = np.asarray(arr.b_factor, dtype=np.float32).copy()

    # Per-residue mean (residues are guaranteed by AlphaFold to have
    # uniform per-atom pLDDT, but we compute the mean anyway for safety).
    per_res: list[float] = []
    for sl in arr.iter_residue_slices():
        per_res.append(float(plddt[sl].mean()))
    plddt_per_residue = np.asarray(per_res, dtype=np.float32)
    mean_plddt = float(plddt.mean()) if plddt.size else 0.0

    # Provenance: this isn't an engine run, it's a loader. Engine name
    # reflects that (the prediction came from AlphaFold at some prior
    # time, but the *molforge* operation is "load from file"). Inputs
    # carry the file path so the chain is traceable.
    protein.metadata[mk.PROVENANCE] = Provenance.from_engine(
        engine="load_alphafold",
        inputs={"path": str(path)},
    )

    # Uniform folding-engine keys (preferred).
    protein.metadata[mk.ENGINE] = "AlphaFold"
    protein.metadata[mk.CONFIDENCE_PER_ATOM] = plddt
    protein.metadata[mk.CONFIDENCE_PER_RESIDUE] = plddt_per_residue
    protein.metadata[mk.MEAN_CONFIDENCE] = mean_plddt

    # Legacy AlphaFold-specific keys (retained for backward compatibility;
    # carry the same values as the uniform keys above).
    protein.metadata[mk.PLDDT] = plddt
    protein.metadata[mk.PLDDT_PER_RESIDUE] = plddt_per_residue
    protein.metadata[mk.MEAN_PLDDT] = mean_plddt
    protein.metadata[mk.SOURCE] = "alphafold"

    return protein
