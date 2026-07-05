"""Concrete endpoint free-energy engines and result parsers.

Wrappers for the external MM/PB(GB)SA tools — Amber's ``MMPBSA.py`` and
``gmx_MMPBSA`` — that implement :class:`molforge.freeenergy.MMGBSAEngine`.
Currently exposes the Amber output parser; the engines that build inputs
and invoke the tools are layered on top.
"""

from __future__ import annotations

from molforge.wrappers.freeenergy.amber import (
    AmberMMGBSA,
    build_mmpbsa_input,
    parse_mmpbsa_dat,
    selection_to_amber_mask,
)
from molforge.wrappers.freeenergy.gromacs import (
    parse_gmx_mmpbsa_dat,
    selection_to_ndx_group,
)

__all__ = [
    "AmberMMGBSA",
    "build_mmpbsa_input",
    "parse_gmx_mmpbsa_dat",
    "parse_mmpbsa_dat",
    "selection_to_amber_mask",
    "selection_to_ndx_group",
]
