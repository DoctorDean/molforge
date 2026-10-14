"""Reduce molecules to their Bemis-Murcko scaffolds.

A compound library's *chemistry* is often better described by its scaffolds
than by its individual structures: the Bemis-Murcko scaffold (Bemis & Murcko,
*J. Med. Chem.* 1996) keeps a molecule's ring systems and the linkers between
them and strips the side chains, which is the level at which medicinal
chemists talk about series, and the level at which "how diverse is this set?"
is a meaningful question.

:func:`murcko_scaffold` returns the scaffold as a
:class:`~molforge.core.Molecule`, so it composes with the rest of the chem
layer — standardize then scaffold, or scaffold then descriptor-filter.
Scaffold-level *grouping* has dedicated support:
:meth:`molforge.chem.MoleculeDataset.group_by_scaffold` for the groups
themselves, and ``unique(..., key="scaffold")`` /
``MoleculeDataset.dedup(key="scaffold")`` to keep one representative per
scaffold. :attr:`molforge.core.Molecule.scaffold_smiles` is the identity
those compare on.

Like the rest of :mod:`molforge.chem`, this is RDKit-backed (a thin wrapper
over ``rdkit.Chem.Scaffolds.MurckoScaffold``) and lazy — calling it without
RDKit raises :class:`~molforge.core.RDKitNotInstalledError`.
"""

from __future__ import annotations

from molforge.core import Molecule, _rdkit

__all__ = ["murcko_scaffold"]


def murcko_scaffold(molecule: Molecule, *, generic: bool = False) -> Molecule:
    """Extract a molecule's Bemis-Murcko scaffold.

    Returns a *new* :class:`~molforge.core.Molecule`, leaving the input
    untouched, and preserves its ``name`` while noting the extraction in
    ``metadata["scaffold"]`` (``"murcko"``, or ``"murcko_generic"`` when
    ``generic``) — the same convention the standardization ops follow.

    An acyclic molecule has no scaffold, so the result is an *empty* molecule
    (``n_atoms == 0``, ``smiles == ""``). That is Bemis-Murcko's own
    convention, not an error; a set of acyclic molecules therefore shares a
    single empty scaffold.

    Args:
        molecule: The molecule to reduce (left unmodified).
        generic: Return the generic framework instead — every atom becomes a
            carbon and every bond a single bond, so scaffolds that differ
            only in their heteroatoms or bond orders (benzene vs. pyridine)
            come out identical. Useful for coarser, topology-only grouping.

    Returns:
        The scaffold as a new :class:`~molforge.core.Molecule`.

    Raises:
        RDKitNotInstalledError: If RDKit isn't installed.

    Example:
        >>> from molforge.core import Molecule
        >>> from molforge.chem import murcko_scaffold
        >>> aspirin = Molecule.from_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
        >>> murcko_scaffold(aspirin).smiles
        'c1ccccc1'
        >>> nicotinamide = Molecule.from_smiles("c1ccncc1C(N)=O")
        >>> murcko_scaffold(nicotinamide).smiles          # pyridine, amide stripped
        'c1ccncc1'
        >>> murcko_scaffold(nicotinamide, generic=True).smiles   # same frame as benzene
        'C1CCCCC1'
    """
    scaffold = _rdkit.murcko_scaffold(molecule.to_rdkit(), generic=generic)
    metadata = dict(molecule.metadata)
    metadata["scaffold"] = "murcko_generic" if generic else "murcko"
    return Molecule.from_rdkit(scaffold, name=molecule.name, metadata=metadata)
