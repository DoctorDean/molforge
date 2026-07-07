"""Concrete endpoint free-energy engines and result parsers.

Wrappers for the external MM/PB(GB)SA tools — Amber's ``MMPBSA.py`` and
``gmx_MMPBSA`` — that implement :class:`molforge.freeenergy.MMGBSAEngine`.
Currently exposes the Amber output parser; the engines that build inputs
and invoke the tools are layered on top.
"""

from __future__ import annotations

from molforge.wrappers.freeenergy.alchemlyb import (
    absolute_binding_free_energy,
    from_alchemlyb,
    from_delta_f,
    relative_binding_free_energy,
)
from molforge.wrappers.freeenergy.amber import (
    AmberMMGBSA,
    build_mmpbsa_input,
    parse_mmpbsa_dat,
    parse_mmpbsa_decomp,
    selection_to_amber_mask,
)
from molforge.wrappers.freeenergy.cinnabar import from_cinnabar
from molforge.wrappers.freeenergy.gromacs import (
    GromacsMMGBSA,
    parse_gmx_mmpbsa_dat,
    selection_to_ndx_group,
)

__all__ = [
    "AmberMMGBSA",
    "GromacsMMGBSA",
    "absolute_binding_free_energy",
    "build_mmpbsa_input",
    "from_alchemlyb",
    "from_cinnabar",
    "from_delta_f",
    "parse_gmx_mmpbsa_dat",
    "parse_mmpbsa_dat",
    "parse_mmpbsa_decomp",
    "relative_binding_free_energy",
    "selection_to_amber_mask",
    "selection_to_ndx_group",
]
