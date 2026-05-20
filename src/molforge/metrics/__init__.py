"""Evaluation and benchmarking metrics.

The metrics here are *task-level*: they grade how good a prediction is
relative to a known reference. They build on top of the lower-level
geometry in :mod:`molforge.structure` (RMSD, superposition, etc.).

What's here:

**Fold-similarity metrics** (single-chain):
    - :func:`tm_score` — TM-score (Zhang & Skolnick 2004). Length-
      normalized, fold-level. > 0.5 ≈ same fold.
    - :func:`gdt_ts` — CASP's GDT-TS. Average pass-fraction at 1/2/4/8 Å.
    - :func:`gdt_ha` — GDT high-accuracy. 0.5/1/2/4 Å (near-experimental).
    - :func:`gdt_per_cutoff` — per-cutoff fractions for custom analysis.
    - :func:`lddt` — alignment-free local Distance Difference Test
      (Mariani et al. 2013). What AlphaFold's pLDDT estimates.
    - :func:`lddt_per_residue` — per-residue lDDT (the per-residue
      confidence pLDDT actually predicts).

**Complex-quality metrics** (multi-chain docking):
    - :func:`dockq` — DockQ score (Basu & Wallner 2016). Single-number
      docking quality with per-component breakdown.
    - :func:`fnat`, :func:`irms`, :func:`lrms` — the underlying CAPRI
      measures (fraction of native contacts, interface RMSD, ligand
      RMSD).

All metrics return float scalars (or, where relevant, NumPy arrays
or dicts) in the conventional direction — higher = better for
TM-score / GDT / lDDT / DockQ; lower = better for RMSDs.
"""

from __future__ import annotations

from molforge.metrics.dockq import dockq, fnat, irms, lrms
from molforge.metrics.gdt import gdt_ha, gdt_per_cutoff, gdt_ts
from molforge.metrics.lddt import lddt, lddt_per_residue
from molforge.metrics.tm import tm_score

__all__ = [  # noqa: RUF022 — grouped by concern
    # Fold similarity (single chain)
    "tm_score",
    "gdt_ts",
    "gdt_ha",
    "gdt_per_cutoff",
    "lddt",
    "lddt_per_residue",
    # Complex quality (multi-chain)
    "dockq",
    "fnat",
    "irms",
    "lrms",
]
