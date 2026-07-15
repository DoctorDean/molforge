"""Run a callable over many inputs, in parallel, with order and error control.

Every user of an engine ends up writing the same ``multiprocessing.Pool``
loop to fold / dock / score a batch of inputs. :func:`map_parallel` is that
loop, once:

    from molforge.parallel import map_parallel
    structures = map_parallel(engine.predict, sequences, backend="process")

The backend is the one real decision:

- ``"process"`` — CPU-bound work that releases no GIL: subprocess engine
  wrappers (Vina, fpocket), NumPy-heavy analysis. One OS process per worker.
  ``func`` and each item must be picklable.
- ``"thread"`` — I/O-bound work: remote fetches (``fetch_many``), reading
  files. Cheap, shares memory, no pickling.
- ``"serial"`` — one at a time. The right choice for GPU engines, where
  running several models at once just fights over one device.

Results come back in input order. ``on_error="skip"`` drops the inputs that
raised (returning fewer results) instead of aborting the whole batch, which
is what you want when one bad input shouldn't sink a long run.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from molforge.core import Protein
    from molforge.docking import DockingEngine, DockingResult
    from molforge.wrappers.folding import FoldingEngine

__all__ = [
    "Backend",
    "OnError",
    "dock_many",
    "fold_many",
    "map_parallel",
    "run_many",
]

_T = TypeVar("_T")
_R = TypeVar("_R")

Backend = Literal["process", "thread", "serial"]
OnError = Literal["raise", "skip"]

_logger = logging.getLogger(__name__)


def _default_workers() -> int:
    return os.cpu_count() or 1


def _run_serial(func: Callable[[_T], _R], items: list[_T], on_error: OnError) -> list[_R]:
    out: list[_R] = []
    for i, item in enumerate(items):
        try:
            out.append(func(item))
        except Exception:
            if on_error == "raise":
                raise
            _logger.warning("map_parallel: item %d failed; skipping", i, exc_info=True)
    return out


def map_parallel(
    func: Callable[[_T], _R],
    items: Iterable[_T],
    *,
    workers: int | None = None,
    backend: Backend = "process",
    on_error: OnError = "raise",
) -> list[_R]:
    """Apply ``func`` to every item, in parallel, preserving input order.

    Args:
        func: A single-argument callable. For an engine method that takes
            options, bind them first with :func:`functools.partial` so what's
            left is one positional input.
        items: The inputs. Consumed once into a list.
        workers: Number of parallel workers. ``None`` uses ``os.cpu_count()``;
            ``1`` runs serially regardless of ``backend``.
        backend: ``"process"`` (default; CPU-bound, needs picklable
            ``func``/items), ``"thread"`` (I/O-bound), or ``"serial"``.
        on_error: ``"raise"`` (default) propagates the first failure;
            ``"skip"`` drops failing inputs and returns the rest, logging a
            warning per failure.

    Returns:
        The results in input order. With ``on_error="skip"`` the list is
        shorter than ``items`` by the number of failures.
    """
    work = list(items)
    if not work:
        return []

    n_workers = _default_workers() if workers is None else workers
    if backend == "serial" or n_workers <= 1 or len(work) == 1:
        return _run_serial(func, work, on_error)

    executor_cls = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
    results: list[_R] = []
    with executor_cls(max_workers=n_workers) as executor:
        # Submit in order; collecting futures in submission order keeps results
        # aligned to inputs even though the work runs concurrently.
        futures = [executor.submit(func, item) for item in work]
        for i, future in enumerate(futures):
            try:
                results.append(future.result())
            except Exception:
                if on_error == "raise":
                    raise
                _logger.warning("map_parallel: item %d failed; skipping", i, exc_info=True)
    return results


def _engine_backend(engine: object, backend: Backend | None) -> Backend:
    """Resolve the backend: an explicit override, else the engine's own
    ``parallelism`` hint, else ``"serial"`` (the safe default for GPU work)."""
    if backend is not None:
        return backend
    return cast("Backend", getattr(engine, "parallelism", "serial"))


def fold_many(
    engine: FoldingEngine,
    sequences: Sequence[str],
    *,
    workers: int | None = None,
    backend: Backend | None = None,
    on_error: OnError = "raise",
    **kwargs: object,
) -> list[Protein]:
    """Fold many sequences with a folding engine.

    Runs ``engine.predict(sequence, **kwargs)`` for each sequence. ``backend``
    defaults to the engine's ``parallelism`` hint (``"serial"`` for GPU
    engines, ``"process"`` for CPU ones). See :func:`map_parallel` for
    ``workers`` / ``on_error`` semantics.
    """
    # partial (not a closure) so the process backend can pickle it; the
    # **kwargs are engine options, never the sequence positional.
    func = partial(engine.predict, **kwargs) if kwargs else engine.predict  # type: ignore[arg-type]
    return map_parallel(
        func,
        list(sequences),
        workers=workers,
        backend=_engine_backend(engine, backend),
        on_error=on_error,
    )


def dock_many(
    engine: DockingEngine,
    receptor: Protein,
    ligands: Iterable[object],
    *,
    workers: int | None = None,
    backend: Backend | None = None,
    on_error: OnError = "raise",
    **kwargs: object,
) -> list[DockingResult]:
    """Dock many ligands against one receptor.

    Runs ``engine.dock(receptor, ligand, **kwargs)`` for each ligand — the
    common virtual-screening shape. ``backend`` defaults to the engine's
    ``parallelism`` hint (Vina and other CPU engines set ``"process"``).
    """
    func = partial(engine.dock, receptor, **kwargs)
    return map_parallel(
        func,
        list(ligands),
        workers=workers,
        backend=_engine_backend(engine, backend),
        on_error=on_error,
    )


def run_many(
    engine: object,
    items: Iterable[object],
    *,
    method: str,
    workers: int | None = None,
    backend: Backend | None = None,
    on_error: OnError = "raise",
    **kwargs: object,
) -> list[Any]:
    """Generic engine batch: run ``engine.<method>(item, **kwargs)`` over items.

    The escape hatch for modalities without a dedicated wrapper (e.g. an MD
    engine's ``run``): ``run_many(md_engine, systems, method="run")``. Each
    item is passed as the first positional argument. ``backend`` defaults to
    the engine's ``parallelism`` hint.
    """
    bound: Callable[..., Any] = getattr(engine, method)
    func = partial(bound, **kwargs) if kwargs else bound
    return map_parallel(
        func,
        list(items),
        workers=workers,
        backend=_engine_backend(engine, backend),
        on_error=on_error,
    )
