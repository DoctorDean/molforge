"""Ingest alchemical free-energy results (FEP/TI via alchemlyb).

molforge doesn't run alchemical free-energy calculations; it ingests the
*analysis* — the output of an `alchemlyb <https://alchemlyb.readthedocs.io>`_
estimator (MBAR, BAR, TI, …) — into the same :class:`FreeEnergyResult`
the endpoint (MM/PB(GB)SA) engines produce, so alchemical and endpoint
estimates rank through the same :class:`~molforge.freeenergy.FreeEnergyRanking`.

An alchemlyb estimator exposes ``delta_f_`` and ``d_delta_f_`` — square
matrices of the free-energy difference (and its error) between every pair
of λ windows. The full transformation is the top-right corner,
``[0, -1]`` (first state → last state). alchemlyb reports these in units
of :math:`k_B T` by default, carrying the temperature and unit in the
DataFrame's ``.attrs``; :func:`from_delta_f` converts to kcal/mol.

Nothing here imports pandas or alchemlyb: the matrices are read through
``numpy.asarray`` and the unit metadata through ``.attrs`` if present, so
molforge takes on no dependency — the caller brings alchemlyb.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from molforge.freeenergy import DeltaDeltaG, FreeEnergyResult

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "absolute_binding_free_energy",
    "from_alchemlyb",
    "from_delta_f",
    "relative_binding_free_energy",
]

# Boltzmann constant in kcal/(mol·K), = R / 4184 with the CODATA gas
# constant R = 8.314462618 J/(mol·K); this is what alchemlyb's to_kcalmol
# uses, so kT results match it (3.041156 kT at 300 K -> 1.813021 kcal/mol).
_KB_KCAL_PER_MOL_K = 8.314462618 / 4184.0

# kcal per kJ, for kJ/mol -> kcal/mol.
_KCAL_PER_KJ = 1.0 / 4.184


def _corner(matrix: Any) -> float:
    """The first-state → last-state value of an alchemlyb delta matrix."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(
            f"expected a 2-D lambda-by-lambda matrix, got shape {arr.shape}"
        )
    return float(arr[0, -1])


def _n_states(matrix: Any) -> int:
    return int(np.asarray(matrix).shape[0])


def _unit_factor(energy_unit: str, temperature: float | None) -> float:
    """Multiplier from ``energy_unit`` to kcal/mol."""
    unit = energy_unit.lower().replace(" ", "")
    if unit in ("kcal/mol", "kcalmol", "kcal"):
        return 1.0
    if unit in ("kj/mol", "kjmol", "kj"):
        return _KCAL_PER_KJ
    if unit in ("kt", "kbt"):
        if temperature is None:
            raise ValueError(
                "temperature is required to convert kT to kcal/mol; pass "
                "temperature=, or ingest a DataFrame whose .attrs carries it "
                "(alchemlyb sets it from the estimator)."
            )
        return _KB_KCAL_PER_MOL_K * float(temperature)
    raise ValueError(
        f"unknown energy_unit {energy_unit!r}; expected 'kT', 'kcal/mol', or 'kJ/mol'"
    )


def from_delta_f(
    delta_f: Any,
    d_delta_f: Any,
    *,
    temperature: float | None = None,
    energy_unit: str | None = None,
    method: str = "FEP",
    metadata: Mapping[str, object] | None = None,
) -> FreeEnergyResult:
    """Ingest alchemlyb ``delta_f_`` / ``d_delta_f_`` matrices.

    Takes the full first-state → last-state transformation (the ``[0, -1]``
    corner) and its error, converting to kcal/mol.

    Args:
        delta_f: The estimator's ``delta_f_`` matrix (a pandas DataFrame,
            or any 2-D array-like). λ states must be ordered, so the
            corner is the whole transformation.
        d_delta_f: The matching ``d_delta_f_`` error matrix.
        temperature: Temperature in K, needed only when the unit is kT and
            the matrix doesn't carry a ``temperature`` in ``.attrs``. An
            explicit value overrides ``.attrs``.
        energy_unit: ``"kT"`` (default when unknown), ``"kcal/mol"``, or
            ``"kJ/mol"``. An explicit value overrides ``.attrs``; otherwise
            the matrix's ``.attrs['energy_unit']`` is used.
        method: Value for :attr:`FreeEnergyResult.method` (e.g. ``"FEP"``,
            ``"TI"``); :func:`from_alchemlyb` fills the estimator name.
        metadata: Extra items merged into the result's metadata.

    Returns:
        A :class:`FreeEnergyResult` with ``delta_g`` / ``uncertainty`` in
        kcal/mol and ``components`` set to ``None`` — an alchemical ΔG is
        a single number, not an MM/GBSA-style term breakdown.

    Raises:
        ValueError: If a matrix isn't 2-D, the unit is unknown, or a kT
            matrix has no temperature (neither argument nor ``.attrs``).
    """
    attrs = getattr(delta_f, "attrs", {}) or {}
    unit = energy_unit if energy_unit is not None else str(attrs.get("energy_unit", "kT"))
    temp = temperature if temperature is not None else attrs.get("temperature")

    factor = _unit_factor(unit, temp)
    delta_g = _corner(delta_f) * factor
    uncertainty = abs(_corner(d_delta_f) * factor)

    meta: dict[str, object] = {
        "n_lambda_states": _n_states(delta_f),
        "source_energy_unit": unit,
    }
    if temp is not None:
        meta["temperature"] = float(temp)
    if metadata:
        meta.update(metadata)

    return FreeEnergyResult(
        delta_g=delta_g,
        uncertainty=uncertainty,
        method=method,
        components=None,
        metadata=meta,
    )


def from_alchemlyb(
    estimator: Any,
    *,
    temperature: float | None = None,
    energy_unit: str | None = None,
    method: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> FreeEnergyResult:
    """Ingest a fitted alchemlyb estimator into a :class:`FreeEnergyResult`.

    Reads ``estimator.delta_f_`` / ``estimator.d_delta_f_`` and, unless
    overridden, labels the result with the estimator's class name
    (``"MBAR"``, ``"BAR"``, ``"TI"``, …).

    Args:
        estimator: A fitted alchemlyb estimator (anything exposing
            ``delta_f_`` and ``d_delta_f_``).
        temperature: See :func:`from_delta_f`.
        energy_unit: See :func:`from_delta_f`.
        method: Overrides the estimator-class-name default.
        metadata: Extra items merged into the result's metadata; the
            estimator name is always recorded under ``"estimator"``.

    Returns:
        A :class:`FreeEnergyResult` in kcal/mol.

    Raises:
        AttributeError: If ``estimator`` lacks ``delta_f_`` / ``d_delta_f_``.
        ValueError: Propagated from :func:`from_delta_f`.
    """
    estimator_name = type(estimator).__name__
    merged: dict[str, object] = {"estimator": estimator_name}
    if metadata:
        merged.update(metadata)
    return from_delta_f(
        estimator.delta_f_,
        estimator.d_delta_f_,
        temperature=temperature,
        energy_unit=energy_unit,
        method=method if method is not None else estimator_name,
        metadata=merged,
    )


def relative_binding_free_energy(
    complex_leg: FreeEnergyResult,
    solvent_leg: FreeEnergyResult,
    *,
    reference: str,
    other: str,
) -> DeltaDeltaG:
    """Close a relative-FEP thermodynamic cycle into a binding ΔΔG.

    A relative FEP perturbation transforms one ligand into another
    (``reference`` → ``other``) along two legs: bound to the receptor (the
    *complex* leg) and free in solution (the *solvent* leg). The
    thermodynamic cycle gives the relative binding free energy

        ΔΔG_bind = ΔG_complex − ΔG_solvent
                 = ΔG_bind(other) − ΔG_bind(reference)

    so a single perturbation's two ingested legs become the binding ΔΔG
    between the two ligands — the quantity relative FEP actually reports.
    A single leg's ΔG is not a binding affinity; only the cycle is.

    Both legs must be the *same* perturbation direction (reference →
    other) and in the same units (kcal/mol, as returned by
    :func:`from_alchemlyb` / :func:`from_delta_f`).

    Args:
        complex_leg: ΔG of the reference → other transformation in the
            complex.
        solvent_leg: ΔG of the same transformation in solvent.
        reference: Label of the reference ligand.
        other: Label of the ligand it is perturbed into.

    Returns:
        A :class:`~molforge.freeenergy.DeltaDeltaG`: the signed ΔΔG_bind
        (negative means ``other`` binds more tightly) with the two legs'
        errors propagated in quadrature. Treat a difference within its
        uncertainty as a tie.
    """
    value = complex_leg.delta_g - solvent_leg.delta_g
    uncertainty = math.hypot(complex_leg.uncertainty, solvent_leg.uncertainty)
    tighter = other if value < 0 else reference
    return DeltaDeltaG(
        reference=reference,
        other=other,
        value=value,
        uncertainty=uncertainty,
        tighter=tighter,
    )


def _leg_terms(leg: FreeEnergyResult | float) -> tuple[float, float]:
    """(value, uncertainty) from a leg given as a result or a bare float."""
    if isinstance(leg, FreeEnergyResult):
        return leg.delta_g, leg.uncertainty
    return float(leg), 0.0


def absolute_binding_free_energy(
    complex_leg: FreeEnergyResult,
    solvent_leg: FreeEnergyResult,
    *,
    restraint_correction: FreeEnergyResult | float = 0.0,
    method: str = "ABFE",
    metadata: Mapping[str, object] | None = None,
) -> FreeEnergyResult:
    """Close a double-decoupling cycle into an absolute binding ΔG.

    Absolute binding free energy (ABFE) by double decoupling annihilates
    the ligand's interactions with its environment in two phases —
    restrained in the complex, and free in solvent — and adds a
    standard-state restraint correction. The thermodynamic cycle gives

        ΔG_bind = ΔG_solvent − ΔG_complex + restraint_correction

    where the two legs are **decoupling** free energies (coupled →
    non-interacting): a strong binder is hard to decouple from the complex
    (large positive ``ΔG_complex``), so ``ΔG_solvent − ΔG_complex`` comes
    out negative — favorable — as it should.

    Unlike the relative cycle, this yields an *absolute* ΔG_bind, so the
    result is a :class:`FreeEnergyResult` that ranks directly.

    Convention notes:

    - Pass the legs as decoupling free energies. If your λ schedule runs
      the other way (coupling), negate them (or reverse the λ order before
      :func:`from_alchemlyb`).
    - ``restraint_correction`` is a *signed* contribution added as-is —
      supply it with the sign your protocol uses (e.g. a Boresch
      standard-state correction). It may be a float or a
      :class:`FreeEnergyResult` (whose uncertainty then propagates).

    Args:
        complex_leg: ΔG of decoupling the (restrained) ligand in the
            complex.
        solvent_leg: ΔG of decoupling the ligand in solvent.
        restraint_correction: Signed standard-state / restraint term.
        method: Value for :attr:`FreeEnergyResult.method`.
        metadata: Extra items merged into the result's metadata.

    Returns:
        A :class:`FreeEnergyResult` with the absolute ΔG_bind in kcal/mol
        (``components`` is ``None``) and the three terms' errors propagated
        in quadrature.
    """
    rc, rc_unc = _leg_terms(restraint_correction)
    delta_g = solvent_leg.delta_g - complex_leg.delta_g + rc
    uncertainty = math.hypot(complex_leg.uncertainty, solvent_leg.uncertainty, rc_unc)

    meta: dict[str, object] = {
        "complex_leg": complex_leg.delta_g,
        "solvent_leg": solvent_leg.delta_g,
        "restraint_correction": rc,
    }
    if metadata:
        meta.update(metadata)

    return FreeEnergyResult(
        delta_g=delta_g,
        uncertainty=uncertainty,
        method=method,
        components=None,
        metadata=meta,
    )
