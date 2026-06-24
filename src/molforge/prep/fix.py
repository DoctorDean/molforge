"""Fix missing atoms and cap free termini with PDBFixer.

PDBFixer is the OpenMM-group tool for the heavy structural fixes
that molforge's pure-Python tooling can't (and shouldn't) attempt:

- **Missing heavy atoms.** X-ray structures often have partial side
  chains where the electron density was weak. PDBFixer's rotamer
  library rebuilds them.
- **Missing residues.** Some loops are too disordered to model;
  PDBFixer can re-thread them given a known sequence.
- **N/C-terminus capping.** Force fields don't know what to do with
  a free amine or carboxyl at a chain end. Adding ACE (acetyl) at
  the N-terminus and NME (N-methyl amide) at the C-terminus blocks
  the charges.
- **Non-standard residue replacement.** Replace e.g. selenomethionine
  (MSE) with methionine (MET).

This module exposes molforge-shaped wrappers for the operations that
typically matter before MD. Each function is composable and
:class:`Protein` → :class:`Protein`. For the all-in-one path, see
:func:`molforge.prep.prepare_for_md`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from molforge.core import Protein
from molforge.prep._deps import require_openmm, require_pdbfixer

__all__ = ["add_caps", "fix_missing_atoms"]


def fix_missing_atoms(
    protein: Protein,
    *,
    fix_missing_residues: bool = False,
    replace_nonstandard: bool = True,
) -> Protein:
    """Rebuild missing heavy atoms with PDBFixer's rotamer library.

    Args:
        protein: The input structure. Typically an X-ray or
            cryo-EM PDB with side-chain atoms missing in flexible
            regions.
        fix_missing_residues: If ``True``, also fill in entire missing
            residues (e.g. a disordered loop). Off by default because
            de-novo loop modelling is risky — the rebuilt geometry is
            often unphysical. Turn on only when you know the missing
            stretch is short and well-constrained.
        replace_nonstandard: If ``True`` (default), non-standard
            residues like selenomethionine (MSE) are replaced with
            their canonical counterparts (MET). Most force fields
            don't have templates for non-standard residues, so this is
            usually what you want.

    Returns:
        A new :class:`Protein` with the fixes applied. The input is
        not modified.

    Raises:
        MDEngineNotInstalledError: If PDBFixer or OpenMM is not
            installed.

    Example:
        >>> import molforge as mf
        >>> from molforge.prep import fix_missing_atoms
        >>> p = mf.fetch("1abc")
        >>> p_fixed = fix_missing_atoms(p)
    """
    fixer = _build_fixer(protein)

    if replace_nonstandard:
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()

    if fix_missing_residues:
        fixer.findMissingResidues()
    else:
        # PDBFixer's addMissingAtoms reads from .missingResidues; if we
        # don't want missing residues filled, we set it to an empty
        # dict explicitly so the call only addresses missing *atoms*
        # within existing residues.
        fixer.missingResidues = {}

    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    out = _fixer_to_protein(fixer, original=protein)
    from molforge.prep._provenance import chain_prep_provenance

    chain_prep_provenance(
        out,
        engine="molforge.prep.fix_missing_atoms",
        parameters={
            "fix_missing_residues": fix_missing_residues,
            "replace_nonstandard": replace_nonstandard,
        },
        input_protein=protein,
    )
    return out


def add_caps(
    protein: Protein,
    *,
    n_cap: str = "ACE",
    c_cap: str = "NME",
) -> Protein:
    """Cap free termini with neutral blocking groups.

    Adds ACE (acetyl) at every chain's N-terminus and NME (N-methyl
    amide) at every C-terminus, masking the charged free-amine /
    free-carboxyl that force fields don't have templates for. The
    capping atoms are placed by PDBFixer's rotamer routine.

    Args:
        protein: The input structure.
        n_cap: Residue name for the N-terminal cap. Default ``"ACE"``
            (acetyl). Pass ``None`` (or an empty string) to skip
            N-terminal capping.
        c_cap: Residue name for the C-terminal cap. Default ``"NME"``
            (N-methyl amide). Pass ``None`` to skip.

    Returns:
        A new :class:`Protein` with capping residues added. The input
        is not modified.

    Raises:
        MDEngineNotInstalledError: If PDBFixer or OpenMM is not
            installed.

    Notes:
        Capping is per chain. A multi-chain protein gets one cap of
        each kind per chain. Chains already terminated by something
        non-standard (a cyclic peptide, a disulfide-bonded terminus)
        will get capped anyway — pre-clean those by hand if it
        matters.

    Example:
        >>> from molforge.prep import add_caps
        >>> p_capped = add_caps(my_protein)
    """
    fixer = _build_fixer(protein)

    # Build a missingResidues entry per chain that PDBFixer's
    # addMissingAtoms() will materialize. The key is a (chain_index,
    # residue_position) tuple — using position 0 inserts before the
    # first residue, position len(chain) appends after the last.
    missing_residues: dict[tuple[int, int], list[str]] = {}
    for chain_idx, chain in enumerate(fixer.topology.chains()):
        residues = list(chain.residues())
        if not residues:
            continue
        # Only cap chains that look like protein (the first residue is
        # a recognised amino acid name). Capping a DNA strand or a
        # ligand chain would be wrong.
        first_resname = residues[0].name.upper()
        if first_resname not in _PROTEIN_RESIDUES:
            continue
        if n_cap:
            missing_residues[(chain_idx, 0)] = [n_cap]
        if c_cap:
            missing_residues[(chain_idx, len(residues))] = [c_cap]

    fixer.missingResidues = missing_residues
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    out = _fixer_to_protein(fixer, original=protein)
    from molforge.prep._provenance import chain_prep_provenance

    chain_prep_provenance(
        out,
        engine="molforge.prep.add_caps",
        parameters={"n_cap": n_cap, "c_cap": c_cap},
        input_protein=protein,
    )
    return out


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


# The 20 canonical amino acid three-letter codes. Used by add_caps to
# decide which chains are protein chains worth capping. Kept inline
# rather than imported because we only need a frozenset check here.
_PROTEIN_RESIDUES: frozenset[str] = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLU",
        "GLN",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }
)


def _build_fixer(protein: Protein) -> Any:
    """Materialize a molforge Protein into a PDBFixer object.

    PDBFixer wraps an OpenMM Topology; the simplest construction path
    is from a PDB file on disk, so we write a temp PDB.
    """
    pdbfixer_mod = require_pdbfixer()
    # Touch OpenMM eagerly so a missing dependency surfaces with our
    # message rather than PDBFixer's, even though we don't use the
    # returned modules here.
    require_openmm()

    from molforge.io import write_pdb

    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as fh:
        in_pdb = Path(fh.name)
    try:
        write_pdb(protein, in_pdb)
        fixer = pdbfixer_mod.PDBFixer(filename=str(in_pdb))
    finally:
        in_pdb.unlink(missing_ok=True)
    return fixer


def _fixer_to_protein(fixer: Any, *, original: Protein) -> Protein:
    """Write a PDBFixer's current state back to a molforge :class:`Protein`."""
    _, app, _ = require_openmm()
    from molforge.io import read_pdb

    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as fh:
        out_pdb = Path(fh.name)
    try:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, str(out_pdb))
        result = read_pdb(out_pdb)
    finally:
        out_pdb.unlink(missing_ok=True)
    if original.metadata:
        result.metadata = {**original.metadata, **result.metadata}
    return result
