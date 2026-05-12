"""Evaluation and benchmarking metrics.

Metrics here are *task-level*: TM-score, lDDT, GDT-TS, docking RMSD,
binding-affinity correlation, etc. Lower-level geometry (per-atom RMSD,
distance maps) lives in :mod:`biocore.structure`.
"""

from __future__ import annotations

__all__ = ["gdt_ts", "lddt", "tm_score"]


def tm_score(reference: object, model: object) -> float:
    """TM-score between two structures. TODO."""
    raise NotImplementedError


def lddt(reference: object, model: object) -> float:
    """lDDT score between two structures. TODO."""
    raise NotImplementedError


def gdt_ts(reference: object, model: object) -> float:
    """GDT-TS score between two structures. TODO."""
    raise NotImplementedError
