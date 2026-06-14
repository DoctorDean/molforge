"""Folding-engine wrappers.

Concrete engines:
    - :class:`ESMFold` — implemented (single-sequence transformer; fast)
    - :class:`AlphaFold` — implemented (MSA-based via ColabFold)
    - :class:`Boltz` — implemented (Boltz-1 / Boltz-2 via subprocess)
    - :class:`RoseTTAFold` — implemented (RoseTTAFold All-Atom; subprocess)

Shared:
    - :class:`FoldingEngine` — abstract base for the engine contract
    - :class:`FoldingEngineNotInstalledError` — raised when heavy
      dependencies (torch, transformers, colabfold, the boltz CLI,
      the RFAA repo + databases, ...) aren't installed.

All engines write per-residue confidence to
``protein.metadata["confidence_per_residue"]`` so downstream code can
read confidence uniformly regardless of which engine produced the
structure.
"""

from __future__ import annotations

from molforge.wrappers.folding._base import (
    FoldingEngine,
    FoldingEngineNotInstalledError,
)
from molforge.wrappers.folding.alphafold import AlphaFold
from molforge.wrappers.folding.boltz import Boltz
from molforge.wrappers.folding.esmfold import ESMFold
from molforge.wrappers.folding.rosettafold import RoseTTAFold

__all__ = [  # noqa: RUF022 — grouped: base classes, then engines
    "FoldingEngine",
    "FoldingEngineNotInstalledError",
    "ESMFold",
    "AlphaFold",
    "Boltz",
    "RoseTTAFold",
]
