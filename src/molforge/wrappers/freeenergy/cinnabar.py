"""Ingest a free-energy network from cinnabar into per-ligand results.

Relative FEP produces a *network* of ΔΔG edges between ligands. To rank
the whole network — especially a non-star map with edges between
arbitrary pairs — you solve the graph for a per-ligand absolute estimate;
`cinnabar <https://cinnabar.openfree.energy>`_ does that with a
maximum-likelihood estimator.

This turns cinnabar's per-ligand output into a ``dict`` of
:class:`FreeEnergyResult`, keyed by ligand label, that drops straight into
:class:`~molforge.freeenergy.FreeEnergyRanking`. The MLE-fit absolute
values float on the network's reference (they're only defined up to a
constant offset unless cinnabar was anchored to experiment), which is
irrelevant to ranking — the offset cancels in every pairwise ΔΔG.

Nothing here imports cinnabar or pandas: the results are read through the
DataFrame's ``.columns`` / ``.to_dict`` and values coerced with
``float`` (handling openff ``Quantity`` via ``.magnitude``), so molforge
takes on no dependency — the caller brings cinnabar.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from molforge.freeenergy import FreeEnergyResult

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = ["from_cinnabar"]


def _pick(columns: Sequence[str], candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in columns:
            return name
    raise ValueError(
        f"none of {candidates} found in cinnabar columns {list(columns)}; "
        "pass the output of FEMap.get_absolute_dataframe()"
    )


def _to_float(value: Any) -> float:
    # openff.units Quantity -> magnitude; plain number -> itself.
    return float(getattr(value, "magnitude", value))


def from_cinnabar(
    source: Any,
    *,
    computational_only: bool = True,
    method: str = "FEP (network)",
    metadata: Mapping[str, object] | None = None,
) -> dict[str, FreeEnergyResult]:
    """Ingest cinnabar's per-ligand absolute estimates.

    Args:
        source: Either a cinnabar ``FEMap`` (with absolute values already
            generated via ``generate_absolute_values()``) or the DataFrame
            its ``get_absolute_dataframe()`` returns. A ``FEMap`` is
            detected by that method and read through it.
        computational_only: Keep only calculated ligands, dropping any
            experimental reference rows (the ``computational`` column). If
            that column is absent, all rows are kept.
        method: Value for :attr:`FreeEnergyResult.method`.
        metadata: Extra items merged into every result's metadata.

    Returns:
        ``{label: FreeEnergyResult}`` in kcal/mol (``components`` is
        ``None``). Wrap in :class:`~molforge.freeenergy.FreeEnergyRanking`
        to rank. Because the absolute values share one network offset,
        ranking and every pairwise ΔΔG are unaffected by it.

    Raises:
        ValueError: If the label / ΔG / uncertainty columns can't be found.
    """
    df = source.get_absolute_dataframe() if hasattr(source, "get_absolute_dataframe") else source
    columns = list(df.columns)

    label_col = _pick(columns, ("label", "ligand"))
    dg_col = _pick(columns, ("DG (kcal/mol)", "DG"))
    unc_col = _pick(columns, ("uncertainty (kcal/mol)", "uncertainty"))
    has_computational = "computational" in columns
    has_source = "source" in columns

    results: dict[str, FreeEnergyResult] = {}
    for row in df.to_dict(orient="records"):
        if computational_only and has_computational and not row["computational"]:
            continue
        meta: dict[str, object] = {"source": "cinnabar"}
        if has_computational:
            meta["computational"] = bool(row["computational"])
        if has_source and row.get("source") is not None:
            meta["cinnabar_source"] = row["source"]
        if metadata:
            meta.update(metadata)
        results[str(row[label_col])] = FreeEnergyResult(
            delta_g=_to_float(row[dg_col]),
            uncertainty=abs(_to_float(row[unc_col])),
            method=method,
            components=None,
            metadata=meta,
        )
    return results
