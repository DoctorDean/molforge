"""Add hydrogens to a heavy-atom structure at a given pH.

The standard PDB file on disk is heavy-atom-only. Every MD force
field needs every hydrogen explicit, with the right protonation
state for the side chain at the system's pH (histidine especially —
its protonation flips around pH ~6).

This module wraps OpenMM's ``Modeller.addHydrogens`` to do the job.
That's a battle-tested rotamer-aware H-placement routine that knows
the standard residue templates and handles pH-dependent states
correctly for the easy cases (histidine, terminal residues, free
cysteines).

The step is idempotent: a structure that already has hydrogens is
returned unchanged. The atom count usually changes, so the returned
:class:`Protein` is a fresh object — the input is not mutated.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from molforge.core import Protein
from molforge.prep._deps import require_openmm

__all__ = ["add_hydrogens"]


# Force-field XML files OpenMM ships, mapped to the protein-template
# subset that Modeller.addHydrogens needs to know which residues to
# protonate. The default ("amber14") is a good general choice; users
# who need CHARMM can override.
_FORCE_FIELD_FILES: dict[str, list[str]] = {
    "amber14": ["amber14-all.xml"],
    "amber14-all": ["amber14-all.xml"],
    "amber99sb": ["amber99sb.xml"],
    "amber99sbildn": ["amber99sbildn.xml"],
    "charmm36": ["charmm36.xml"],
}


def add_hydrogens(
    protein: Protein,
    *,
    pH: float = 7.4,  # noqa: N803 — pH is the correct chemistry capitalization
    force_field: str = "amber14",
) -> Protein:
    """Add hydrogens to a :class:`Protein` at the given pH.

    Args:
        protein: The input structure. Typically heavy-atom-only —
            standard PDB / AlphaFold / docking-engine output.
        pH: The pH at which to assign protonation states. Default 7.4
            (physiological). Histidine is the residue this matters
            most for: at pH 7.4 most His are HID (neutral, H on δ),
            occasionally HIE (neutral, H on ε), rarely HIP (charged).
            Modeller picks per the side-chain environment.
        force_field: An OpenMM force-field name (see
            :data:`_FORCE_FIELD_FILES`) or any XML filename OpenMM
            can find. Determines the residue templates Modeller
            consults — the default (``amber14``) covers all standard
            amino acids.

    Returns:
        A new :class:`Protein` with hydrogens added. The input is not
        modified. Calling on a structure that already has explicit
        hydrogens is a no-op (returns an equivalent :class:`Protein`).

    Raises:
        MDEngineNotInstalledError: If OpenMM is not installed.

    Example:
        >>> import molforge as mf
        >>> from molforge.prep import add_hydrogens
        >>> p = mf.load("alphafold_output.pdb")  # heavy atoms only
        >>> p_h = add_hydrogens(p, pH=7.4)
        >>> p_h.atom_array.n_atoms > p.atom_array.n_atoms
        True
    """
    _openmm, app, _unit = require_openmm()

    # We round-trip through a PDB file: it's the cheapest way to get a
    # molforge Protein into an OpenMM Topology + positions, and back.
    # Modeller would also accept positions in nm directly, but the
    # topology setup is identical either way.
    from molforge.io import read_pdb, write_pdb

    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as fh:
        in_pdb = Path(fh.name)
    out_pdb: Path | None = None
    try:
        write_pdb(protein, in_pdb)
        pdb = app.PDBFile(str(in_pdb))

        ff_files = _FORCE_FIELD_FILES.get(force_field, [force_field])
        forcefield = app.ForceField(*ff_files)

        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(forcefield, pH=float(pH))

        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as out_fh:
            out_pdb = Path(out_fh.name)
        app.PDBFile.writeFile(modeller.topology, modeller.positions, str(out_pdb))
        result = read_pdb(out_pdb)
    finally:
        in_pdb.unlink(missing_ok=True)
        if out_pdb is not None:
            out_pdb.unlink(missing_ok=True)

    # Preserve metadata across the round-trip.
    if protein.metadata:
        result.metadata = {**protein.metadata, **result.metadata}
    return result
