"""molforge — a unified library for structural bioinformatics, MD, and ML.

This package exposes a small top-level surface. Subpackages are the primary
import points; users should typically import them directly:

    >>> from molforge.core import Protein, Chain, Residue, Atom
    >>> from molforge.io import load, save
    >>> from molforge.structure import rmsd

`molforge` is a *library*, not a framework: there is no runtime, no
orchestration layer, and no required entry point. Import what you need.

The one thing worth having at the top level is the question you ask
*about* an install rather than with it:

    >>> import molforge
    >>> molforge.engine_versions()["OpenMM"].version  # doctest: +SKIP
    '8.1.1'
"""

from __future__ import annotations

from molforge.versions import BackendVersion, engine_versions

__version__ = "0.8.0"
__all__ = ["BackendVersion", "__version__", "engine_versions"]
