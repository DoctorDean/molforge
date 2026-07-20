"""Pocket detection engine wrappers.

A pocket detector takes a :class:`molforge.core.Protein` and returns a
list of :class:`molforge.docking.Pocket` — the candidate ligand-binding
pockets on its surface, ranked by the detector's own scoring.

Concrete detectors:

- :func:`molforge.wrappers.pockets.fpocket.detect_pockets` — fpocket,
  the Voronoi-based (geometric) algorithm from the Discngine/fpocket
  project.
- :func:`molforge.wrappers.pockets.p2rank.detect_pockets_p2rank` —
  P2Rank, the machine-learning (random-forest) detector, which returns
  the same :class:`Pocket` shape and is a drop-in alternative.

Detectors are free functions rather than classes because they're
stateless: there's no in-process model to load, no per-call reuse
benefit. They shell out to an external binary and parse its output —
P2Rank's ML model is loaded by its own Java process. A Python-native
ML detector (PUResNet, ScanNet) that warm-loads weights in-process may
adopt a class-based pattern; that's a per-detector decision when each
lands.

Pocket detection sits *next to* docking in the workflow taxonomy: a
typical use is detection -> pick a pocket -> dock against it, which
is also why :class:`molforge.docking.Pocket` lives alongside
:class:`Pose` and :class:`DockingResult` in the docking module.
"""

from __future__ import annotations

from molforge.wrappers.pockets.fpocket import (
    FpocketNotInstalledError,
    detect_pockets,
)
from molforge.wrappers.pockets.p2rank import (
    P2RankNotInstalledError,
    detect_pockets_p2rank,
)

__all__ = [
    "FpocketNotInstalledError",
    "P2RankNotInstalledError",
    "detect_pockets",
    "detect_pockets_p2rank",
]
