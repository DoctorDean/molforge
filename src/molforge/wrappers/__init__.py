"""Thin wrappers around external engines.

Each wrapper exposes a stable, typed interface to a third-party tool
so that engines are swappable from user code. Heavy dependencies are
imported lazily inside method bodies to keep ``import molforge`` cheap.

Subpackages by engine category:
    - :mod:`molforge.wrappers.folding` — ESMFold, AlphaFold/ColabFold
    - :mod:`molforge.wrappers.docking` — AutoDock Vina
    - :mod:`molforge.wrappers.md` — OpenMM
    - :mod:`molforge.wrappers.generative` — RFdiffusion, ProteinMPNN
"""

from __future__ import annotations

__all__: list[str] = []
