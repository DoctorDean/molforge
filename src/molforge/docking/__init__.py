"""Docking abstractions: receptor/ligand prep, pose handling, scoring.

A *docking engine* takes a receptor (typically a protein) and a ligand
(typically a small molecule) and returns a set of plausible binding
poses ranked by an engine-specific score. The interface here is
intentionally narrow:

    receptor + ligand -> DockingResult (list of Pose)

Concrete engines (AutoDock Vina, DiffDock, ...) inherit from
:class:`DockingEngine` and live under :mod:`molforge.wrappers.docking`.

By convention, every docking engine writes its lowest-energy / top-scored
pose first in :attr:`DockingResult.poses`, with scores sorted ascending
(more negative = better, matching Vina's convention).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from molforge.core import Protein


@dataclass
class Pose:
    """A single docked ligand pose.

    Attributes:
        ligand: The ligand structure for this pose as a :class:`Protein`
            (with ``entity_type == "ligand"`` atoms; small-molecule
            chemistry-aware features like bond orders are out of scope
            for the core data model — pose comparison and ranking only
            need coordinates and energy).
        score: Engine-specific scalar score. Convention: lower = better
            (e.g. Vina returns kcal/mol affinity estimates as negative
            numbers).
        rank: 0-indexed rank within the result (0 = best).
        rmsd_lb: Vina-style "RMSD lower bound" between this pose and the
            best pose, if reported by the engine.
        rmsd_ub: Vina-style "RMSD upper bound" between this pose and the
            best pose, if reported.
        metadata: Engine-specific extras (component-energy breakdown,
            confidence scores, etc.).
    """

    ligand: Protein
    score: float
    rank: int = 0
    rmsd_lb: float | None = None
    rmsd_ub: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class DockingResult:
    """Collection of poses returned by a docking engine.

    Attributes:
        poses: List of :class:`Pose` objects, sorted best-first (lowest
            score first).
        receptor: The receptor structure that was docked against.
        engine: Engine name (``"Vina"``, ``"DiffDock"``, etc.).
        metadata: Engine-specific run metadata (search box, exhaustiveness,
            walltime, etc.).
    """

    poses: list[Pose] = field(default_factory=list)
    receptor: Protein | None = None
    engine: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.poses)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.poses)

    @property
    def best(self) -> Pose:
        """The top-scoring pose. Raises ``IndexError`` if empty."""
        return self.poses[0]

    def top_n(self, n: int) -> list[Pose]:
        """Return the n best-scoring poses."""
        return self.poses[:n]


@dataclass
class Pocket:
    """A predicted ligand-binding pocket on a protein surface.

    Pockets are the output of *pocket detection* algorithms (fpocket,
    P2Rank, mdpocket on a trajectory, etc.). They sit alongside
    :class:`Pose` and :class:`DockingResult` in this module because
    pockets are *for* docking — the canonical downstream use is
    passing :attr:`center` to a docking engine's ``center=`` argument
    as a binding-site hint.

    A :class:`Pocket` is intentionally *thin*. Detection algorithms
    expose dozens of descriptors (volume, alpha-sphere count,
    hydrophobicity, polarity, surface-area splits, etc.); only the
    handful most universally useful for downstream code live as
    typed attributes. The rest goes in :attr:`metadata` so
    engine-specific extras don't pollute the type.

    Attributes:
        center: ``(3,)`` float32 array of the pocket's geometric
            centre (Å). The natural input for a docking box centre.
        residues: List of residue identifiers (``(chain_id,
            residue_id, insertion_code)`` triples) lining the pocket.
            Empty list when the detector doesn't report lining
            residues.
        volume: Pocket volume in Å³. ``None`` when the detector
            doesn't report a volume.
        score: Detector's pocket score (higher = more pocket-like).
            ``None`` when the detector doesn't compute one. For
            cross-detector comparability, prefer :attr:`druggability`
            where available.
        druggability: Detector's druggability score (typically 0–1,
            higher = more drug-binding-like). ``None`` when not
            available. fpocket and similar detectors compute this.
        rank: 0-indexed rank within the detection result (0 = best
            by the detector's own scoring).
        metadata: Detector-specific extras (alpha-sphere count,
            SASA splits, hydrophobicity descriptors, etc.) plus a
            :class:`molforge.core.Provenance` under the
            ``provenance`` key.
    """

    center: NDArray[np.float32]
    residues: list[tuple[str, int, str]] = field(default_factory=list)
    volume: float | None = None
    score: float | None = None
    druggability: float | None = None
    rank: int = 0
    metadata: dict[str, object] = field(default_factory=dict)


class DockingEngine(ABC):
    """Abstract base for receptor-ligand docking engines.

    Subclasses must implement :meth:`dock`. They should also handle their
    own receptor and ligand preparation (charge assignment, conversion
    to engine-specific formats) inside :meth:`dock` rather than exposing
    that complexity to users — that's the whole point of having a wrapper.

    Attributes:
        name: Human-readable engine name (set by subclasses).
    """

    name: str = "DockingEngine"

    @abstractmethod
    def dock(
        self,
        receptor: Protein,
        ligand: object,
        **kwargs: object,
    ) -> DockingResult:
        """Dock ``ligand`` against ``receptor``.

        Args:
            receptor: The receptor structure.
            ligand: The ligand. Type is engine-dependent — typically a
                path to an SDF/MOL2, a SMILES string, or a
                :class:`Protein` (with ligand atoms).
            **kwargs: Engine-specific options.

        Returns:
            A :class:`DockingResult` with poses sorted best-first.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class DockingEngineNotInstalledError(ImportError):
    """Raised when a docking engine's heavy dependencies aren't installed.

    The message points at the relevant ``pip install`` extras so users
    can fix it without grepping the docs.
    """


__all__ = [
    "DockingEngine",
    "DockingEngineNotInstalledError",
    "DockingResult",
    "Pocket",
    "Pose",
]
