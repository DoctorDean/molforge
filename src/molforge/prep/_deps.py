"""Lazy-import helpers for the heavy MD-prep dependencies.

PDBFixer and OpenMM are not imported when :mod:`molforge.prep` is
imported — only when a function that needs them is actually called.
Both raise :class:`molforge.md.MDEngineNotInstalledError` with the
same install hint so the user error story is consistent across the
package: "you can't do MD-flavoured things without the MD extra."
"""

from __future__ import annotations

from typing import Any

from molforge.md import MDEngineNotInstalledError


def require_pdbfixer() -> Any:
    """Import PDBFixer or raise a clean error.

    PDBFixer is the OpenMM-group tool for the heavy structural fixes
    (missing-atom completion, missing-residue completion, residue
    renaming, capping). It depends on OpenMM, so a user without
    OpenMM can't have PDBFixer either.
    """
    try:
        import pdbfixer
    except ImportError as e:
        raise MDEngineNotInstalledError(
            "Structure preparation requires PDBFixer. Install with:\n"
            "    pip install 'molforge[prep]'\n"
            "or directly:\n"
            "    pip install pdbfixer\n"
            "PDBFixer depends on OpenMM; on Windows prefer conda:\n"
            "    conda install -c conda-forge pdbfixer\n"
            f"Underlying error: {e}"
        ) from e
    return pdbfixer


def require_openmm() -> tuple[Any, Any, Any]:
    """Import OpenMM and return (openmm, openmm.app, openmm.unit).

    Same shape as :meth:`molforge.wrappers.md.OpenMM._require_openmm`;
    duplicated here so the prep subpackage doesn't have to
    instantiate an MD engine just to borrow the import.
    """
    try:
        import openmm
        import openmm.app as app
        import openmm.unit as unit
    except ImportError as e:
        raise MDEngineNotInstalledError(
            "Structure preparation requires OpenMM. Install with:\n"
            "    pip install 'molforge[prep]'\n"
            "On Windows, prefer conda: `conda install -c conda-forge openmm`.\n"
            f"Underlying error: {e}"
        ) from e
    return openmm, app, unit
