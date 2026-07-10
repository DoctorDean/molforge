"""Standardize (clean) small molecules for consistent downstream use.

Raw molecules from vendors, databases, or docking output are inconsistent:
salts and solvents tagged along, charges written arbitrarily, the same
structure drawn as different tautomers. Standardizing brings them to a
canonical form so that identity comparison, deduplication, and modelling
all see the same molecule the same way.

Each function returns a *new* :class:`~molforge.core.Molecule`, leaving the
input untouched, and preserves its ``name`` while noting the cleaning in
metadata. All are RDKit-backed (via :mod:`molforge.core._rdkit`) and lazy —
calling one without RDKit raises
:class:`~molforge.core.RDKitNotInstalledError`.
"""

from __future__ import annotations

from typing import Any

from molforge.core import Molecule
from molforge.core import _rdkit

__all__ = [
    "canonical_tautomer",
    "cleanup",
    "largest_fragment",
    "neutralize",
    "standardize",
]


def _rebuild(mol: Any, source: Molecule, step: str) -> Molecule:
    """Wrap a cleaned RDKit mol, carrying name and noting the step."""
    metadata = dict(source.metadata)
    applied = metadata.get("standardized")
    steps = list(applied) if isinstance(applied, list) else []
    steps.append(step)
    metadata["standardized"] = steps
    return Molecule.from_rdkit(mol, name=source.name, metadata=metadata)


def cleanup(molecule: Molecule) -> Molecule:
    """Sanitize, normalize functional groups, and reionize."""
    return _rebuild(_rdkit.cleanup(molecule.to_rdkit()), molecule, "cleanup")


def largest_fragment(molecule: Molecule) -> Molecule:
    """Keep the largest organic fragment — strips salts and solvents."""
    return _rebuild(_rdkit.largest_fragment(molecule.to_rdkit()), molecule, "largest_fragment")


def neutralize(molecule: Molecule) -> Molecule:
    """Remove formal charges where chemically reasonable."""
    return _rebuild(_rdkit.uncharge(molecule.to_rdkit()), molecule, "neutralize")


def canonical_tautomer(molecule: Molecule) -> Molecule:
    """Convert to RDKit's canonical tautomer."""
    return _rebuild(_rdkit.canonical_tautomer(molecule.to_rdkit()), molecule, "canonical_tautomer")


def standardize(
    molecule: Molecule,
    *,
    desalt: bool = True,
    neutralize: bool = True,
    tautomer: bool = False,
) -> Molecule:
    """Run a standard cleaning pipeline over a molecule.

    Applies, in order: cleanup (always), keep-largest-fragment (``desalt``),
    neutralize (``neutralize``), and canonical tautomer (``tautomer``). The
    default is a sensible desalt + neutralize; the canonical tautomer step
    is off by default because it's the slowest and occasionally surprising.

    Args:
        molecule: The molecule to standardize (left unmodified).
        desalt: Keep only the largest organic fragment.
        neutralize: Remove formal charges where reasonable.
        tautomer: Convert to the canonical tautomer.

    Returns:
        A new standardized :class:`~molforge.core.Molecule`; its
        ``metadata["standardized"]`` lists the steps applied.

    Raises:
        RDKitNotInstalledError: If RDKit isn't installed.
    """
    mol = _rdkit.cleanup(molecule.to_rdkit())
    steps = ["cleanup"]
    if desalt:
        mol = _rdkit.largest_fragment(mol)
        steps.append("largest_fragment")
    if neutralize:
        mol = _rdkit.uncharge(mol)
        steps.append("neutralize")
    if tautomer:
        mol = _rdkit.canonical_tautomer(mol)
        steps.append("canonical_tautomer")

    metadata = dict(molecule.metadata)
    metadata["standardized"] = steps
    return Molecule.from_rdkit(mol, name=molecule.name, metadata=metadata)
