"""Running a drawn graph, one block at a time.

:func:`~molforge.reproducibility.replay` is all-or-nothing: it returns the
terminal output, and a pipeline that dies in its third step reports a
traceback and loses the two structures it already built. On a canvas that
is the wrong shape entirely — the blocks are *right there*, and which one
is running, which finished, and which one failed is the whole reason to
look at a diagram instead of a log.

This runs a :class:`~molforge.gui.graph.PipelineGraph` through
:func:`~molforge.reproducibility.replay_steps`, attributing each step back
to the node that drew it::

    run = RunRecord.start(graph)
    run.wait()
    [ (s.node_id, s.state) for s in run.steps ]
    # [('fold', 'succeeded'), ('dock', 'failed')]

It adds no execution logic of its own. The handlers, the engine
resolution, and the threading of one step's output into the next are
replay's, so a graph run here and the same ``pipeline.yaml`` replayed from
a script do the same thing — which is the property that makes the GUI
worth trusting. What this adds is *attribution* and *survival*: which node
each step belongs to, how long it took, and the partial results of a run
that didn't finish.

Runs happen on a worker thread because engines block for minutes to hours
and a dashboard cannot. One run at a time per record, no pool: these are
calls into CUDA, subprocesses pinned to every core, and tools that write
scratch files to fixed paths. Running two concurrently is a decision a
user should make by starting two, not one a dashboard should make for
them.
"""

from __future__ import annotations

import contextlib
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from molforge.gui.catalog import BlockSpec
    from molforge.gui.graph import PipelineGraph

__all__ = [
    "RunRecord",
    "StepRun",
    "describe_output",
]

#: Lifecycle of one block. ``cancelled`` is distinct from ``failed``: a run
#: someone stopped is not a pipeline that broke, and a UI that conflates
#: them sends people hunting for a bug they caused on purpose.
_PENDING = "pending"
_RUNNING = "running"
_SUCCEEDED = "succeeded"
_FAILED = "failed"
_CANCELLED = "cancelled"


@dataclass
class StepRun:
    """One block's turn: what it is, how it went, and what it produced.

    Attributes:
        node_id: The canvas node this step came from, so the editor can
            colour the right box.
        step: 1-indexed position in the chain.
        engine: Engine name, as recorded.
        operation: The replayed operation.
        state: One of ``pending`` / ``running`` / ``succeeded`` /
            ``failed`` / ``cancelled``.
        started / finished: Monotonic timestamps, ``0.0`` while unset.
        error: The failure, phrased for someone reading a toast.
        traceback: The full traceback, kept separately — a wrapper's own
            error message is usually the useful part, and burying it under
            forty frames in the UI helps nobody.
        output: A short description of what the step produced. Never the
            object itself: a run holds structures and trajectories, and a
            record that pinned them would keep every intermediate of every
            run alive for as long as the server does.
    """

    node_id: str
    step: int
    engine: str
    operation: str
    state: str = _PENDING
    started: float = 0.0
    finished: float = 0.0
    error: str = ""
    traceback: str = ""
    output: str = ""

    @property
    def elapsed(self) -> float:
        """Seconds taken, or so far. ``0.0`` before the step starts."""
        if not self.started:
            return 0.0
        return (self.finished or time.monotonic()) - self.started

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "step": self.step,
            "engine": self.engine,
            "operation": self.operation,
            "state": self.state,
            "elapsed": round(self.elapsed, 3),
            "error": self.error,
            "traceback": self.traceback,
            "output": self.output,
        }


@dataclass
class RunRecord:
    """One execution of a graph, watchable while it happens.

    Every read goes through a lock because the worker thread writes these
    fields while the server reads them to answer a poll.
    """

    id: str = field(default_factory=lambda: uuid4().hex[:12])
    steps: list[StepRun] = field(default_factory=list)
    state: str = _PENDING
    error: str = ""
    result: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)

    # ---------- starting ----------

    @classmethod
    def start(
        cls,
        graph: PipelineGraph,
        *,
        context: dict[str, Any] | None = None,
        catalog: Sequence[BlockSpec] | None = None,
        on_change: Callable[[RunRecord], None] | None = None,
    ) -> RunRecord:
        """Validate the graph, then run it on a worker thread.

        Validation happens here, on the caller's thread, so an unrunnable
        graph raises :class:`~molforge.gui.graph.GraphError` to whoever
        asked rather than arriving later as a failed run. This is the one
        place that requires every engine to be installed — saving a graph
        for another machine is fine, running it here is not.

        Args:
            graph: The graph to run.
            context: Extra inputs, merged over the ones the graph carries.
                Lets a caller supply a live object where the canvas only
                had a path.
            catalog: Blocks to resolve against; defaults to the live one.
            on_change: Called after every state transition, with the
                record. For pushing updates without polling.

        Returns:
            The record, already running.

        Raises:
            GraphError: If the graph can't run here.
        """
        graph.validate(catalog=catalog, require_available=True)
        manifest, graph_context = graph.to_manifest(catalog=catalog)
        merged = {**graph_context, **(context or {})}

        # Step N of the manifest is node N of the ordered chain; the
        # manifest has no node ids and the canvas has nothing else to
        # colour, so the mapping is made here, once, while both are known.
        node_ids = [node.id for node in graph.order()]

        record = cls(
            steps=[
                StepRun(
                    node_id=node_ids[i],
                    step=step.step,
                    engine=step.engine,
                    operation=step.operation,
                )
                for i, step in enumerate(manifest.steps)
            ]
        )
        record._thread = threading.Thread(
            target=record._run,
            args=(manifest, merged, on_change),
            name=f"molforge-run-{record.id}",
            daemon=True,
        )
        record.state = _RUNNING
        record._thread.start()
        return record

    # ---------- the worker ----------

    def _run(
        self,
        manifest: Any,
        context: dict[str, Any],
        on_change: Callable[[RunRecord], None] | None,
    ) -> None:
        from molforge.reproducibility import replay_steps

        def changed() -> None:
            # A listener that raises must not take the pipeline with it: a
            # dashboard disconnecting mid-run is routine, and the engines
            # are the part worth protecting.
            if on_change is not None:
                with contextlib.suppress(Exception):
                    on_change(self)

        index = 0
        self._mark(0, _RUNNING)
        changed()
        output: Any = None
        try:
            for _step, output in replay_steps(manifest, context=context):
                self._finish(index, _SUCCEEDED, output=describe_output(output))
                index += 1
                if self._cancel.is_set():
                    self._remaining(_CANCELLED)
                    self._settle(_CANCELLED, "cancelled")
                    changed()
                    return
                if index < len(self.steps):
                    self._mark(index, _RUNNING)
                changed()
        except Exception as e:  # every engine failure lands here
            # The wrapper's own message is the useful part: molforge's
            # errors carry install hints and the argument that was wrong.
            self._finish(
                index,
                _FAILED,
                error=f"{type(e).__name__}: {e}",
                tb=traceback.format_exc(),
            )
            self._remaining(_CANCELLED)
            self._settle(_FAILED, f"{type(e).__name__}: {e}")
            changed()
            return

        self._settle(_SUCCEEDED, "", result=describe_output(output))
        changed()

    # ---------- state, all under the lock ----------

    def _mark(self, index: int, state: str) -> None:
        with self._lock:
            if index < len(self.steps):
                self.steps[index].state = state
                self.steps[index].started = time.monotonic()

    def _finish(
        self, index: int, state: str, *, output: str = "", error: str = "", tb: str = ""
    ) -> None:
        with self._lock:
            if index >= len(self.steps):  # pragma: no cover - defensive
                return
            step = self.steps[index]
            step.state = state
            step.finished = time.monotonic()
            step.output = output
            step.error = error
            step.traceback = tb

    def _remaining(self, state: str) -> None:
        """Mark the steps that never got their turn.

        Leaving them ``pending`` would read as "still queued" on a run that
        has stopped, which is the one thing a status display must not say.
        """
        with self._lock:
            for step in self.steps:
                if step.state == _PENDING:
                    step.state = state

    def _settle(self, state: str, error: str, *, result: str = "") -> None:
        with self._lock:
            self.state = state
            self.error = error
            self.result = result

    # ---------- control and inspection ----------

    def cancel(self) -> None:
        """Ask the run to stop after the step in flight.

        It cannot interrupt one: a step is a subprocess or a CUDA call, and
        killing it mid-write is how a half-written output file gets mistaken
        for a finished one. A long fold will therefore keep going until it
        returns.
        """
        self._cancel.set()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the run ends. Returns whether it did."""
        if self._thread is None:  # pragma: no cover - a record that never started
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()

    @property
    def running(self) -> bool:
        return self.state == _RUNNING

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "state": self.state,
                "error": self.error,
                "result": self.result,
                "steps": [s.to_dict() for s in self.steps],
            }


def describe_output(obj: Any) -> str:
    """A one-line description of what a step produced.

    Deliberately a string rather than the object: a record holds every
    intermediate of a run, and keeping structures and trajectories alive
    for as long as the server runs is how a dashboard becomes the reason a
    machine is out of memory. Written to be read in a node's footer, so it
    leads with the size that tells you the step did what you meant —
    residue count for a structure, pose count for a docking result.
    """
    if obj is None:
        return ""
    if isinstance(obj, (str, int, float, bool)):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        if not obj:
            return "empty"
        return f"{len(obj)} × {type(obj[0]).__name__}"

    name = type(obj).__name__
    for attribute, label in _COUNTS:
        value = getattr(obj, attribute, None)
        if value is None:
            continue
        count = value if isinstance(value, int) else _length(value)
        if count is not None:
            return f"{name} ({count} {label})"
    return name


#: Counts worth leading with, most informative first. A structure's size
#: is what says the step folded what you meant rather than a fragment of
#: it; an alignment's depth is the number that decides the fold that
#: consumes it. Both are read off the result types molforge already
#: exposes (``Protein.n_residues``, ``MultipleSequenceAlignment.depth``)
#: rather than by reaching into their internals.
_COUNTS: tuple[tuple[str, str], ...] = (
    ("n_residues", "residues"),
    ("depth", "sequences"),
    ("poses", "poses"),
    ("n_atoms", "atoms"),
)


def _length(value: Any) -> int | None:
    """``len(value)`` where that means something, else ``None``."""
    try:
        return len(value)
    except TypeError:
        return None
