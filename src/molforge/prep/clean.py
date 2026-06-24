"""Drop crystallographic / non-standard atoms before MD.

A typical X-ray PDB carries far more than the protein you want to
simulate: buffer salts, glycerol, the precipitant from the
crystallization condition, sometimes ions or ligands you don't care
about, and crystallographic waters that an MD setup will replace
with explicit solvent anyway.

This module's :func:`remove_heterogens` is a small, opinionated
filter that keeps the canonical residues (amino acids + nucleotides)
and drops everything else, with toggles for the things users
sometimes want to keep — waters, ions, ligands.

The implementation is pure-Python: no PDBFixer / OpenMM needed.
"""

from __future__ import annotations

import numpy as np

from molforge.core import AtomArray, Protein
from molforge.core.constants import NUCLEOTIDE_TO_ONE, THREE_TO_ONE

__all__ = ["remove_heterogens"]


# The "canonical" residues we always keep — the 20 amino acids plus the
# standard DNA/RNA nucleotides. Anything outside this set is a
# heterogen as far as remove_heterogens() is concerned.
_CANONICAL_RESIDUES: frozenset[str] = frozenset(THREE_TO_ONE.keys() | NUCLEOTIDE_TO_ONE.keys())

# Common water residue names. Different programs emit different
# strings; this list covers the ones we see in the wild.
_WATER_RESIDUES: frozenset[str] = frozenset(
    {"HOH", "WAT", "H2O", "DOD", "OH2", "SOL", "TIP", "TIP3", "TIP4"}
)

# Common monatomic-ion residue names. We don't try to be exhaustive
# (PDB's chemical-component dictionary has hundreds); these are the
# ones that show up in protein structures regularly.
_ION_RESIDUES: frozenset[str] = frozenset(
    {
        "NA",
        "CL",
        "K",
        "MG",
        "CA",
        "ZN",
        "FE",
        "MN",
        "CU",
        "NI",
        "CO",
        "F",
        "BR",
        "IOD",
        "I",
        "RB",
        "CS",
    }
)


def remove_heterogens(
    protein: Protein,
    *,
    keep_water: bool = False,
    keep_ions: bool = False,
    keep_ligands: bool = False,
    keep: frozenset[str] | set[str] | None = None,
) -> Protein:
    """Return a new :class:`Protein` with non-standard atoms removed.

    By default, only canonical amino acids and nucleotides are kept.
    Waters, ions, ligands, and other heterogens are dropped. The
    ``keep_*`` flags add categories back; ``keep`` is an explicit
    residue-name allow-list for anything else (a specific cofactor
    you want to preserve, for example).

    Args:
        protein: The input structure.
        keep_water: Keep water residues (HOH, WAT, SOL, ...). MD
            workflows usually solvate explicitly, so the default is
            ``False`` — even crystallographic waters get replaced.
        keep_ions: Keep monatomic ions (Na+, Cl-, Mg2+, Zn2+, ...).
            Defaults to ``False``; some structural ions (e.g. zinc in
            a zinc finger) you'll want to keep — pass ``True`` or use
            the explicit ``keep`` allow-list.
        keep_ligands: Keep atoms marked as ``entity_type == "ligand"``.
            Useful when you want to MD a protein-ligand complex.
        keep: Explicit residue-name allow-list. Any residue whose name
            (case-insensitive) is in this set is kept regardless of
            the toggles above.

    Returns:
        A new :class:`Protein` containing only the atoms that passed
        the filter. The original protein is not modified.

    Example:
        >>> protein = mf.load("4hhb.pdb")
        >>> clean = remove_heterogens(protein, keep_ions=True)
        >>> # Hemoglobin's HEM cofactors would also be dropped here.
        >>> # If we want them: keep={"HEM"}.
        >>> clean = remove_heterogens(protein, keep={"HEM"})
    """
    arr = protein.atom_array
    n = arr.n_atoms
    if n == 0:
        return Protein(arr)

    # Normalize residue names to upper-case for matching. AtomArray
    # stores them as fixed-width unicode; we compare against our
    # uppercase reference sets.
    res_names = np.asarray([str(r).strip().upper() for r in arr.residue_name])
    entity_types = np.asarray([str(t).strip().lower() for t in arr.entity_type])

    extra_keep = frozenset(s.strip().upper() for s in (keep or set()))

    # An atom is kept if any of these clauses is true.
    canonical = np.isin(res_names, list(_CANONICAL_RESIDUES))
    explicit = np.isin(res_names, list(extra_keep)) if extra_keep else np.zeros(n, dtype=bool)
    water = (
        np.isin(res_names, list(_WATER_RESIDUES)) | (entity_types == "water")
        if keep_water
        else np.zeros(n, dtype=bool)
    )
    ions = (
        np.isin(res_names, list(_ION_RESIDUES)) | (entity_types == "ion")
        if keep_ions
        else np.zeros(n, dtype=bool)
    )
    ligands = (entity_types == "ligand") if keep_ligands else np.zeros(n, dtype=bool)

    mask = canonical | explicit | water | ions | ligands
    filtered: AtomArray = arr.select(mask)
    out = Protein(filtered)
    # Preserve metadata; the structure shrank but the provenance is
    # the same.
    if protein.metadata:
        out.metadata = {**protein.metadata}

    # Chain a Provenance step. The output's chain() then reads as
    # the workflow oldest-first.
    from molforge.prep._provenance import chain_prep_provenance

    chain_prep_provenance(
        out,
        engine="molforge.prep.remove_heterogens",
        parameters={
            "keep_water": keep_water,
            "keep_ions": keep_ions,
            "keep_ligands": keep_ligands,
            "keep": sorted(keep) if keep else None,
        },
        input_protein=protein,
    )
    return out
