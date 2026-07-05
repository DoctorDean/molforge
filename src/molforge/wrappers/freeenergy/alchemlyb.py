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

from typing import TYPE_CHECKING, Any

import numpy as np

from molforge.freeenergy import FreeEnergyResult

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["from_alchemlyb", "from_delta_f"]

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
