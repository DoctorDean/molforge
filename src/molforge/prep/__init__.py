"""Structure-preparation utilities — turn raw PDBs into MD-ready systems.

A molforge user with a PDB file from AlphaFold, RoseTTAFold, the
RCSB, or a cryo-EM deposition almost always needs the same kind of
clean-up before running molecular dynamics:

1. **Drop crystallographic clutter** — buffer salts, glycerol,
   cryoprotectants, sometimes the ligand. (Or keep them, depending on
   what you're simulating.)
2. **Fix missing heavy atoms** — X-ray structures often have partial
   side chains where the electron density was weak; AlphaFold output
   doesn't suffer from this but other structure-prediction tools can.
3. **Cap free termini** — terminal residues with bare N/C ends aren't
   standard amino acids as far as a force field is concerned. Capping
   with ACE / NME makes them tractable.
4. **Add hydrogens at the right pH** — most PDBs are heavy-atom-only;
   force fields need every hydrogen explicit, with the right
   protonation state for the side chain at the system's pH.

This subpackage provides:

- :func:`remove_heterogens` — drop non-standard residues (waters,
  ions, ligands, crystallization additives) on a configurable
  allow-list.
- :func:`fix_missing_atoms` — rebuild missing heavy atoms with
  PDBFixer's rotamer library.
- :func:`add_caps` — terminate free amine / carboxyl ends with ACE /
  NME caps.
- :func:`add_hydrogens` — add hydrogens at a given pH using OpenMM's
  Modeller.
- :func:`prepare_for_md` — the convenience entry point that chains the
  above with sensible defaults for an "AlphaFold-PDB-to-MD" workflow.

Heavy deps (OpenMM, PDBFixer) are loaded lazily inside the functions
that need them — importing :mod:`molforge.prep` itself does **not**
require either. Functions that need them raise a clean
:class:`molforge.md.MDEngineNotInstalledError` with install
instructions when the deps are absent.

Install once with::

    pip install 'molforge[prep]'

For composable use, call the individual functions in whatever order
fits your case. For the common case, :func:`prepare_for_md` does the
right thing.
"""

from __future__ import annotations

from molforge.prep.clean import remove_heterogens
from molforge.prep.fix import add_caps, fix_missing_atoms
from molforge.prep.pipeline import prepare_for_md
from molforge.prep.protonate import add_hydrogens

__all__ = [
    "add_caps",
    "add_hydrogens",
    "fix_missing_atoms",
    "prepare_for_md",
    "remove_heterogens",
]
