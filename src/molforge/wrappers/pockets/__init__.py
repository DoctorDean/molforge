"""Pocket detection engine wrappers.

A pocket detector takes a :class:`molforge.core.Protein` and returns a
list of :class:`molforge.docking.Pocket` — the candidate ligand-binding
pockets on its surface, ranked by the detector's own scoring.

Concrete detectors:

- :func:`molforge.wrappers.pockets.fpocket.detect_pockets` — fpocket,
  the Voronoi-based algorithm from the Discngine/fpocket project.

Detectors are free functions rather than classes because they're
stateless: there's no model to load, no per-call reuse benefit. They
shell out to a small external binary and parse its output. Future
ML-based detectors (PUResNet, ScanNet, P2Rank when the install path
is cleaner) may follow a class-based pattern for warm-loading model
weights; that's a per-detector decision when each lands.

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

__all__ = [
    "FpocketNotInstalledError",
    "detect_pockets",
]
