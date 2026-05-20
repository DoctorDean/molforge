"""Generative-model wrappers: backbone generation and sequence design.

Concrete engines:
    - :class:`RFdiffusion` — RoseTTAFold diffusion for *de novo*
      protein backbone generation (Watson et al. 2023). Unconditional
      monomer generation, motif scaffolding, binder design.
    - :class:`ProteinMPNN` — message-passing neural network for
      protein sequence design (Dauparas et al. 2022). Given a
      backbone, propose sequences that should fold to it.

Shared:
    - :class:`GenerativeEngine` — abstract base for the engine contract
    - :class:`GenerativeEngineNotInstalledError` — raised when the
      heavy dependencies aren't installed.

These engines complete the *de novo* design loop in molforge: combined
with the folding wrappers (ESMFold, AlphaFold) and the analysis layer
(structure, metrics), you can go from "I want a new protein for X" to
"here are 50 candidate sequences ranked by predicted quality."
"""

from __future__ import annotations

from molforge.generative import GenerativeEngine, GenerativeEngineNotInstalledError
from molforge.wrappers.generative.proteinmpnn import ProteinMPNN
from molforge.wrappers.generative.rfdiffusion import RFdiffusion

__all__ = [  # noqa: RUF022 — grouped: base, then engines
    "GenerativeEngine",
    "GenerativeEngineNotInstalledError",
    "RFdiffusion",
    "ProteinMPNN",
]
