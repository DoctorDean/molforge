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
    """Load an AlphaFold prediction, exposing pLDDT as a metadata field.

    The protein is read via :func:`molforge.io.read_pdb`, then:

    - ``protein.metadata["plddt"]`` is set to a ``(n_atoms,)`` float32
      array of per-atom pLDDT scores (copied from the B-factor column).
    - ``protein.metadata["plddt_per_residue"]`` is set to a
      ``(n_residues,)`` float32 array of per-residue pLDDT (mean across
      the residue's atoms).
    - ``protein.metadata["mean_plddt"]`` is the overall mean.
    - ``protein.metadata["source"]`` is set to ``"alphafold"``.

    The B-factor column is left intact for compatibility with downstream
    tools that still expect to find pLDDT there.
    """
    protein = read_pdb(path)
    arr = protein.atom_array
    plddt = np.asarray(arr.b_factor, dtype=np.float32).copy()
    protein.metadata["plddt"] = plddt
    protein.metadata["mean_plddt"] = float(plddt.mean()) if plddt.size else 0.0
    protein.metadata["source"] = "alphafold"

    # Per-residue mean (residues are guaranteed by AlphaFold to have
    # uniform per-atom pLDDT, but we compute the mean anyway for safety).
    per_res: list[float] = []
    for sl in arr.iter_residue_slices():
        per_res.append(float(plddt[sl].mean()))
    protein.metadata["plddt_per_residue"] = np.asarray(per_res, dtype=np.float32)
    return protein
