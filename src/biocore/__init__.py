"""biocore — a unified library for structural bioinformatics, MD, and ML.

This package exposes a small top-level surface. Subpackages are the primary
import points; users should typically import them directly:

    >>> from biocore.core import Protein, Chain, Residue, Atom
    >>> from biocore.io import load, save
    >>> from biocore.structure import rmsd

`biocore` is a *library*, not a framework: there is no runtime, no
orchestration layer, and no required entry point. Import what you need.
"""

from __future__ import annotations

__version__ = "0.0.1"

__all__ = ["__version__"]
