"""Binding free energy: result types, ranking, and engine base.

An *endpoint* binding free energy method estimates the free energy of
association ΔG_bind for a receptor–ligand complex from an ensemble of
structures (typically an MD trajectory), without the alchemical
machinery of FEP/TI. The workhorse is **MM/GBSA** (and its Poisson–
Boltzmann sibling MM/PBSA): molecular-mechanics interaction energy plus
an implicit-solvent correction, averaged over frames.

molforge treats these as *post-processors* over a
:class:`molforge.md.Trajectory`: the MD layer produces the ensemble,
this layer consumes it and returns a :class:`FreeEnergyResult`. Concrete
engines that invoke ``MMPBSA.py`` / ``gmx_MMPBSA`` live under
:mod:`molforge.wrappers.freeenergy`; this module holds the shared types
and the :class:`MMGBSAEngine` base they implement.

A word on what these numbers are worth. Endpoint methods are notoriously
poor at *absolute* affinities — the implicit-solvent term is
systematically biased and the configurational entropy is usually dropped
— but the *rank order* across a congeneric series is often useful. The
types here lean into that: every result carries an uncertainty, entropy
is explicitly ``None`` when it wasn't computed (rather than silently
zero), and :class:`FreeEnergyRanking` exposes pairwise ΔΔG with
propagated error rather than a bare ordering, so ties within the error
bars stay visible.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    import numpy as np
    from numpy.typing import NDArray

    from molforge.core import Provenance
    from molforge.md import Trajectory


@dataclass(frozen=True)
class FreeEnergyComponents:
    """Per-term decomposition of a binding free energy, in kcal/mol.

    The four energetic terms are the standard MM/PB(GB)SA breakdown; each
    is a *difference* (complex − receptor − ligand). ``entropy`` is the
    configurational entropy contribution −TΔS, carried separately because
    it comes from a distinct calculation (normal-mode or quasi-harmonic)
    that single-trajectory runs usually skip.

    Attributes:
        vdw: Van der Waals interaction energy.
        electrostatic: Electrostatic (Coulomb) interaction energy.
        polar_solvation: Polar solvation free energy (GB or PB term).
        nonpolar_solvation: Nonpolar / cavity + dispersion solvation term.
        entropy: The −TΔS contribution, or ``None`` when entropy was not
            computed. ``None`` is deliberately distinct from ``0.0``: a
            dropped entropy term is unknown, not zero.
    """

    vdw: float
    electrostatic: float
    polar_solvation: float
    nonpolar_solvation: float
    entropy: float | None = None

    @property
    def enthalpy(self) -> float:
        """Interaction enthalpy: the sum of the four energetic terms.

        This is the ΔH part of ΔG = ΔH − TΔS. When ``entropy`` is
        ``None``, this equals the reported binding free energy (endpoint
        runs without an entropy calculation report ΔH as ΔG).
        """
        return (
            self.vdw
            + self.electrostatic
            + self.polar_solvation
            + self.nonpolar_solvation
        )


@dataclass
class FreeEnergyResult:
    """A binding free energy estimate with its uncertainty.

    Attributes:
        delta_g: Binding free energy ΔG_bind in kcal/mol. Lower (more
            negative) means tighter binding.
        uncertainty: Standard error of ``delta_g`` in kcal/mol, from the
            spread across frames or blocks. A ΔG without this is close to
            meaningless, so it is required.
        method: Method label, e.g. ``"MM/GBSA"`` or ``"MM/PBSA"``.
        components: Per-term :class:`FreeEnergyComponents`, or ``None``
            if the engine did not report a breakdown.
        convergence: Optional ``(n_frames,)`` running estimate of
            ``delta_g`` as frames accumulate — a flat tail indicates a
            converged average. ``None`` when not tracked.
        provenance: Optional :class:`molforge.core.Provenance` recording
            the engine, parameters and inputs that produced this result.
        metadata: Engine-specific extras (per-term std devs, frame count,
            solvent model, walltime, ...).
    """

    delta_g: float
    uncertainty: float
    method: str
    components: FreeEnergyComponents | None = None
    convergence: NDArray[np.float64] | None = None
    provenance: Provenance | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.uncertainty < 0:
            raise ValueError(f"uncertainty must be non-negative, got {self.uncertainty}")


@dataclass(frozen=True)
class DeltaDeltaG:
    """A relative binding free energy between two ligands, with error.

    Attributes:
        reference: Label of the reference ligand.
        other: Label of the other ligand.
        value: ΔΔG = ΔG(other) − ΔG(reference) in kcal/mol. Negative
            means ``other`` binds more tightly than ``reference``.
        uncertainty: Propagated standard error,
            ``sqrt(σ_reference² + σ_other²)`` (the two estimates come
            from independent runs).
        tighter: Label of the tighter-binding ligand (the lower ΔG). On
            an exact tie this is ``reference``; consult ``value`` and
            ``uncertainty`` to judge whether a difference is real.
    """

    reference: str
    other: str
    value: float
    uncertainty: float
    tighter: str


class FreeEnergyRanking:
    """An affinity ranking over labelled :class:`FreeEnergyResult`\\ s.

    Ranking — "which of these ligands binds tightest?" — is the tractable
    job for endpoint methods, so this is the headline output over a
    congeneric series. Results are ordered by ``delta_g`` ascending
    (tightest first). Comparisons are exposed as :class:`DeltaDeltaG`
    with propagated uncertainty rather than a bare rank, so a pair that
    is tied within error stays visibly tied. Deliberately absent: any
    significance verdict — the right test depends on assumptions
    (frame correlation, Gaussianity) molforge should not bake in.
    """

    def __init__(self, results: Mapping[str, FreeEnergyResult]) -> None:
        """Build a ranking.

        Args:
            results: Mapping from ligand label to its
                :class:`FreeEnergyResult`. Must be non-empty.

        Raises:
            ValueError: If ``results`` is empty.
        """
        if not results:
            raise ValueError("FreeEnergyRanking needs at least one result")
        self._results: dict[str, FreeEnergyResult] = dict(results)

    @property
    def results(self) -> dict[str, FreeEnergyResult]:
        """A copy of the label → result mapping."""
        return dict(self._results)

    @property
    def ranked(self) -> list[tuple[str, FreeEnergyResult]]:
        """``(label, result)`` pairs, tightest binder first."""
        return sorted(self._results.items(), key=lambda kv: kv[1].delta_g)

    @property
    def best(self) -> tuple[str, FreeEnergyResult]:
        """The tightest-binding ``(label, result)`` pair."""
        return self.ranked[0]

    def delta_delta_g(self, reference: str, other: str) -> DeltaDeltaG:
        """Relative binding free energy of ``other`` vs ``reference``.

        Args:
            reference: Label of the reference ligand.
            other: Label of the ligand compared against the reference.

        Returns:
            A :class:`DeltaDeltaG` with the signed ΔΔG and its propagated
            uncertainty.

        Raises:
            KeyError: If either label is not in the ranking.
        """
        a = self._results[reference]
        b = self._results[other]
        value = b.delta_g - a.delta_g
        uncertainty = math.hypot(a.uncertainty, b.uncertainty)
        tighter = other if b.delta_g < a.delta_g else reference
        return DeltaDeltaG(
            reference=reference,
            other=other,
            value=value,
            uncertainty=uncertainty,
            tighter=tighter,
        )

    def __len__(self) -> int:
        return len(self._results)

    def __iter__(self) -> Iterator[tuple[str, FreeEnergyResult]]:
        return iter(self.ranked)


@dataclass(frozen=True)
class ResidueContribution:
    """One residue's contribution to the binding free energy.

    From a per-residue MM/PB(GB)SA decomposition (the DELTA section):
    ``total`` is the residue's net contribution to ΔG_bind, split into the
    same energy terms as the overall estimate. A large negative ``total``
    marks a binding hotspot; a positive one, a residue that opposes
    binding.

    Attributes:
        residue: Residue label, e.g. ``"LEU 40"``.
        total: Net contribution to ΔG_bind in kcal/mol (the sum of the
            component terms below).
        uncertainty: Standard error of ``total`` across frames.
        internal: Internal term (bond/angle/dihedral; plus 1-4 for
            ``idecomp=1``).
        vdw: van der Waals contribution.
        electrostatic: Electrostatic contribution.
        polar_solvation: Polar solvation contribution.
        nonpolar_solvation: Non-polar solvation contribution.
    """

    residue: str
    total: float
    uncertainty: float
    internal: float
    vdw: float
    electrostatic: float
    polar_solvation: float
    nonpolar_solvation: float

    def __post_init__(self) -> None:
        if self.uncertainty < 0:
            raise ValueError(f"uncertainty must be >= 0, got {self.uncertainty}")


class Decomposition:
    """A per-residue decomposition of a binding free energy.

    A mapping from residue label to its :class:`ResidueContribution`,
    preserving the order the residues were reported, with
    :meth:`hotspots` to surface the residues that drive (or oppose)
    binding. This is the endpoint-method answer to "*where* does the
    affinity come from?" once :class:`FreeEnergyRanking` has answered
    "which ligand?".
    """

    def __init__(self, contributions: Sequence[ResidueContribution]) -> None:
        """Build a decomposition.

        Args:
            contributions: Per-residue contributions, in report order.

        Raises:
            ValueError: If two contributions share a residue label.
        """
        by_residue: dict[str, ResidueContribution] = {}
        for c in contributions:
            if c.residue in by_residue:
                raise ValueError(f"duplicate residue label {c.residue!r}")
            by_residue[c.residue] = c
        self._by_residue = by_residue

    @property
    def residues(self) -> list[ResidueContribution]:
        """All contributions, in report order."""
        return list(self._by_residue.values())

    @property
    def total(self) -> float:
        """Sum of every residue's contribution (the decomposed total)."""
        return sum(c.total for c in self._by_residue.values())

    def hotspots(
        self, n: int | None = None, *, favorable: bool = True
    ) -> list[ResidueContribution]:
        """Residues ranked by contribution.

        Args:
            n: Return at most this many; ``None`` returns all.
            favorable: If true (default), most binding-favorable first
                (most negative ``total``); if false, most opposing first.

        Returns:
            The ranked contributions.
        """
        ordered = sorted(
            self._by_residue.values(), key=lambda c: c.total, reverse=not favorable
        )
        return ordered if n is None else ordered[:n]

    def __getitem__(self, residue: str) -> ResidueContribution:
        return self._by_residue[residue]

    def __iter__(self) -> Iterator[str]:
        return iter(self._by_residue)

    def __len__(self) -> int:
        return len(self._by_residue)

    def __contains__(self, residue: object) -> bool:
        return residue in self._by_residue


class MMGBSAEngine(ABC):
    """Abstract base for endpoint binding-free-energy engines.

    Concrete engines (under :mod:`molforge.wrappers.freeenergy`) invoke an
    external tool — ``MMPBSA.py`` or ``gmx_MMPBSA`` — building its input
    from a trajectory, running it, and parsing the results into a
    :class:`FreeEnergyResult`. They handle their own topology/endpoint
    marshalling rather than exposing it to callers.

    Despite the name, these engines do both MM/GBSA and MM/PBSA; the
    solvent model is a parameter (``solvent_model``) that defaults to
    generalized Born (``"gb"``, i.e. MM/GBSA). Poisson–Boltzmann
    (``"pb"``, MM/PBSA) is the slower, off-default alternative.

    Attributes:
        name: Human-readable engine name (set by subclasses).
    """

    name: str = "MMGBSAEngine"

    @abstractmethod
    def run(
        self,
        trajectory: Trajectory,
        *,
        receptor: object,
        ligand: object,
        solvent_model: str = "gb",
        **kwargs: object,
    ) -> FreeEnergyResult:
        """Estimate ΔG_bind from a trajectory.

        Args:
            trajectory: The ensemble to average over; its topology
                defines the complex.
            receptor: Selection identifying the receptor atoms within the
                complex (resolved against the topology, consistent with
                molforge's selection machinery).
            ligand: Selection identifying the ligand atoms.
            solvent_model: ``"gb"`` for MM/GBSA (default) or ``"pb"`` for
                MM/PBSA.
            **kwargs: Engine-specific options.

        Returns:
            A :class:`FreeEnergyResult` with ΔG, uncertainty, and — where
            the tool reports them — a component breakdown.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class MMGBSAEngineNotInstalledError(ImportError):
    """Raised when an endpoint-free-energy engine's tool isn't installed.

    The message points at the relevant install instructions (Amber's
    ``MMPBSA.py`` or ``gmx_MMPBSA``) so users can fix it without grepping
    the docs.
    """


__all__ = [
    "Decomposition",
    "DeltaDeltaG",
    "FreeEnergyComponents",
    "FreeEnergyRanking",
    "FreeEnergyResult",
    "MMGBSAEngine",
    "MMGBSAEngineNotInstalledError",
    "ResidueContribution",
]
