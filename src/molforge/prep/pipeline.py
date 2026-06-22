"""The all-in-one ``prepare_for_md`` convenience entry point.

For the common "I have a PDB, I want to MD it" case, this composes
the individual prep functions in the right order with defaults that
work for the typical AlphaFold-or-RCSB-to-OpenMM workflow.

For finer control, call the individual functions from
:mod:`molforge.prep.clean`, :mod:`molforge.prep.fix`, and
:mod:`molforge.prep.protonate` directly.
"""

from __future__ import annotations

from molforge.core import Protein
from molforge.prep.clean import remove_heterogens
from molforge.prep.fix import add_caps, fix_missing_atoms
from molforge.prep.protonate import add_hydrogens

__all__ = ["prepare_for_md"]


def prepare_for_md(
    protein: Protein,
    *,
    pH: float = 7.4,  # noqa: N803 — pH is the correct chemistry capitalization
    keep_water: bool = False,
    keep_ions: bool = False,
    keep_ligands: bool = False,
    keep: frozenset[str] | set[str] | None = None,
    fix_missing_residues: bool = False,
    replace_nonstandard: bool = True,
    add_caps_to_termini: bool = True,
    add_explicit_hydrogens: bool = True,
    force_field: str = "amber14",
) -> Protein:
    """Convert a raw PDB into an MD-ready :class:`Protein`.

    Chains four pre-MD steps in the order they should run:

    1. :func:`~molforge.prep.remove_heterogens` — drop crystallographic
       clutter (waters, buffer salts, ligands) unless the caller asks
       to keep them.
    2. :func:`~molforge.prep.fix_missing_atoms` — rebuild missing
       heavy atoms (and, optionally, missing residues).
    3. :func:`~molforge.prep.add_caps` — cap free termini with ACE /
       NME so the force field can template them.
    4. :func:`~molforge.prep.add_hydrogens` — add explicit hydrogens
       at the requested pH.

    Args:
        protein: The input structure (typically heavy-atom-only).
        pH: pH at which to assign protonation states for step 4
            (default 7.4 — physiological).
        keep_water: Forwarded to :func:`remove_heterogens`. Default
            ``False`` — MD setups solvate explicitly.
        keep_ions: Forwarded to :func:`remove_heterogens`. Default
            ``False`` — turn on for structures with bound ions you
            need to preserve (e.g. zinc fingers, metalloenzymes).
        keep_ligands: Forwarded to :func:`remove_heterogens`. Default
            ``False`` — turn on for protein-ligand MD.
        keep: Forwarded to :func:`remove_heterogens`. Explicit
            residue-name allow-list for cofactors and other entities
            you want to preserve.
        fix_missing_residues: Forwarded to :func:`fix_missing_atoms`.
            Default ``False`` — de-novo loop modelling is risky.
        replace_nonstandard: Forwarded to :func:`fix_missing_atoms`.
            Default ``True`` — replace MSE → MET, etc.
        add_caps_to_termini: If ``True`` (default), add ACE/NME caps.
            Set ``False`` if you've already capped the structure or
            you want charged termini.
        add_explicit_hydrogens: If ``True`` (default), add hydrogens.
            Set ``False`` if you've already protonated the structure.
        force_field: Force-field name passed to
            :func:`add_hydrogens`. Default ``"amber14"``.

    Returns:
        A new :class:`Protein` ready to be passed to an MD engine's
        ``prepare`` method. The input is not modified.

    Raises:
        MDEngineNotInstalledError: If OpenMM / PDBFixer is required
            (any step beyond ``remove_heterogens``) and not installed.

    Example:
        >>> import molforge as mf
        >>> from molforge.prep import prepare_for_md
        >>> from molforge.wrappers.md import OpenMM
        >>>
        >>> raw = mf.load("alphafold_output.pdb")
        >>> system = prepare_for_md(raw, pH=7.4)
        >>> sim = OpenMM().prepare(system)
        >>> sim = OpenMM().minimize(sim)
        >>> traj = OpenMM().run(sim, n_steps=50_000, save_every=500)

    Notes:
        Order matters. Heterogen removal first means we don't waste
        cycles fixing or hydrogenating atoms we're about to throw
        away. Capping before protonation means the cap residues get
        their own hydrogens placed correctly. Missing-atom completion
        before capping means residues whose terminal atoms were
        absent in the input aren't capped on top of incomplete
        backbones.
    """
    result = remove_heterogens(
        protein,
        keep_water=keep_water,
        keep_ions=keep_ions,
        keep_ligands=keep_ligands,
        keep=keep,
    )
    result = fix_missing_atoms(
        result,
        fix_missing_residues=fix_missing_residues,
        replace_nonstandard=replace_nonstandard,
    )
    if add_caps_to_termini:
        result = add_caps(result)
    if add_explicit_hydrogens:
        result = add_hydrogens(result, pH=pH, force_field=force_field)
    return result
