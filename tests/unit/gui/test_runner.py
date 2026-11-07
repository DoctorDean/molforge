"""Running a graph block by block.

These register fake engines in the plugin registry, so a run exercises the
real dispatch — replay's handlers, engine resolution, output threading —
without a model weight in sight. What is being tested is attribution and
survival: which node a step belongs to, and what a half-finished run
leaves behind.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from molforge.gui.catalog import BlockParameter, BlockPort, BlockSpec
from molforge.gui.graph import GraphEdge, GraphError, GraphNode, PipelineGraph
from molforge.gui.runner import RunRecord, describe_output

if TYPE_CHECKING:
    from collections.abc import Iterator


class _Folder:
    """Stands in for a folding engine; records that it ran."""

    name = "FakeFolder"

    def __init__(self, *, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def predict(self, sequence: str, **_kwargs: object) -> str:
        return f"structure({sequence})"


class _Docker:
    name = "FakeDocker"

    def __init__(self, *, scoring: str = "vina") -> None:
        self.scoring = scoring

    def dock(
        self, receptor: object, ligand: object, *, center: tuple[float, float, float], **_k: object
    ) -> str:
        return f"poses({receptor},{ligand},{center})"


class _Broken:
    """Fails the way a real engine does: a missing binary, with the hint."""

    name = "Broken"

    def __init__(self) -> None:
        pass

    def dock(self, receptor: object, ligand: object, **_kwargs: object) -> str:
        raise RuntimeError("gnina is not installed. Install it with: conda install gnina")


class _Slow:
    """Blocks until a test releases it, so a run can be observed mid-flight."""

    name = "Slow"
    released: ClassVar[threading.Event] = threading.Event()

    def __init__(self) -> None:
        pass

    def predict(self, sequence: str, **_kwargs: object) -> str:
        assert self.released.wait(timeout=10), "test did not release the slow engine"
        return "slow-structure"


def _block(
    block_id: str,
    *,
    inputs: tuple[BlockPort, ...] = (),
    output: str = "structure",
    parameters: tuple[BlockParameter, ...] = (),
) -> BlockSpec:
    engine, _, operation = block_id.partition(".")
    return BlockSpec(
        id=block_id,
        engine=engine,
        operation=operation,
        label=operation,
        category="test",
        inputs=inputs,
        output=BlockPort("output", output),
        parameters=parameters,
        available=True,
    )


CATALOG = (
    _block("FakeFolder.predict", inputs=(BlockPort("sequence", "sequence"),)),
    _block(
        "FakeDocker.dock",
        inputs=(BlockPort("receptor", "structure"), BlockPort("ligand", "ligand")),
        output="docking_result",
        parameters=(
            BlockParameter("center", "vector3", "tuple[float, float, float]", required=True),
        ),
    ),
    _block(
        "Broken.dock",
        inputs=(BlockPort("receptor", "structure"), BlockPort("ligand", "ligand")),
        output="docking_result",
    ),
    _block("Slow.predict", inputs=(BlockPort("sequence", "sequence"),)),
    _block("Absent.predict", inputs=(BlockPort("sequence", "sequence"),)),
)

UNAVAILABLE = BlockSpec(
    id="Absent.predict",
    engine="Absent",
    operation="predict",
    label="predict",
    category="test",
    inputs=(BlockPort("sequence", "sequence"),),
    output=BlockPort("output", "structure"),
    available=False,
    detail="not installed here",
)


@pytest.fixture(autouse=True)
def engines() -> Iterator[None]:
    from molforge import plugins

    plugins.register_engine("FakeFolder", _Folder)
    plugins.register_engine("FakeDocker", _Docker)
    plugins.register_engine("Broken", _Broken)
    plugins.register_engine("Slow", _Slow)
    _Slow.released = threading.Event()
    yield
    plugins.clear()


def _graph(second: str = "FakeDocker.dock", *, first: str = "FakeFolder.predict") -> PipelineGraph:
    """Fold a sequence, then dock against the structure it produced.

    The second block varies so a test can make the *downstream* step fail
    while the upstream one has already succeeded, which is the case
    step-by-step running exists for.
    """
    parameters = {"center": [1.0, 2.0, 3.0]} if second == "FakeDocker.dock" else {}
    return PipelineGraph(
        nodes=[
            GraphNode(id="fold", block_id=first, inputs={"sequence": "MKTV"}),
            GraphNode(
                id="dock",
                block_id=second,
                inputs={"ligand": "aspirin.sdf"},
                parameters=parameters,
            ),
        ],
        edges=[GraphEdge(source="fold", target="dock", target_port="receptor")],
    )


def _run(graph: PipelineGraph, **kwargs: Any) -> RunRecord:
    record = RunRecord.start(graph, catalog=CATALOG, **kwargs)
    assert record.wait(timeout=10), "run did not finish"
    return record


class TestSuccess:
    def test_every_step_succeeds_and_is_attributed_to_its_node(self) -> None:
        """The node id is the whole point: without it the editor knows a
        step finished but not which box to colour."""
        record = _run(_graph())
        assert record.state == "succeeded"
        assert [(s.node_id, s.state) for s in record.steps] == [
            ("fold", "succeeded"),
            ("dock", "succeeded"),
        ]

    def test_output_threads_from_one_step_to_the_next(self) -> None:
        """Dispatch is replay's, so the receptor must be the fold's output
        rather than anything the canvas supplied."""
        record = _run(_graph())
        assert "structure(MKTV)" in record.steps[1].output

    def test_the_terminal_output_is_described(self) -> None:
        record = _run(_graph())
        assert record.result.startswith("poses(")

    def test_steps_are_timed(self) -> None:
        record = _run(_graph())
        assert all(s.elapsed >= 0.0 for s in record.steps)
        assert all(s.finished for s in record.steps)


class TestFailure:
    def test_a_failing_step_is_marked_and_the_rest_are_not_left_pending(self) -> None:
        """A stopped run that still shows 'pending' steps reads as one
        that's still going, which is the one thing a status must not say."""
        record = _run(_graph("Broken.dock"))
        assert record.state == "failed"
        assert record.steps[0].state == "succeeded"
        assert record.steps[1].state == "failed"
        assert not [s for s in record.steps if s.state == "pending"]

    def test_the_engines_own_message_survives(self) -> None:
        """molforge's errors carry install hints; losing them to a generic
        'step 2 failed' throws away the useful part."""
        record = _run(_graph("Broken.dock"))
        assert "conda install gnina" in record.steps[1].error
        assert "conda install gnina" in record.error

    def test_the_traceback_is_kept_but_separate(self) -> None:
        record = _run(_graph("Broken.dock"))
        assert "RuntimeError" in record.steps[1].traceback
        assert record.steps[1].traceback != record.steps[1].error

    def test_work_already_done_is_not_lost(self) -> None:
        """The reason for running step-by-step at all: a pipeline that dies
        in step two still built something in step one."""
        record = _run(_graph("Broken.dock"))
        assert record.steps[0].output == "structure(MKTV)"


class TestValidation:
    def test_an_unrunnable_graph_raises_to_the_caller(self) -> None:
        """Not as a failed run: the caller asked for something impossible
        and should hear about it now."""
        graph = PipelineGraph(
            nodes=[GraphNode(id="x", block_id="Absent.predict", inputs={"sequence": "MKTV"})]
        )
        with pytest.raises(GraphError, match="isn't available"):
            RunRecord.start(graph, catalog=(*CATALOG[:-1], UNAVAILABLE))

    def test_a_missing_required_parameter_raises_before_anything_runs(self) -> None:
        graph = _graph()
        graph.node("dock").parameters.clear()
        with pytest.raises(GraphError, match="needs center"):
            RunRecord.start(graph, catalog=CATALOG)


class TestCancellation:
    def test_cancelling_stops_the_chain_and_says_so(self) -> None:
        """Cancelled is not failed: a run someone stopped is not a pipeline
        that broke, and conflating them sends people hunting a bug they
        caused on purpose."""
        record = RunRecord.start(_graph(first="Slow.predict"), catalog=CATALOG)
        record.cancel()
        _Slow.released.set()
        assert record.wait(timeout=10)

        assert record.state == "cancelled"
        assert record.steps[0].state == "succeeded"  # it was in flight, and finished
        assert record.steps[1].state == "cancelled"  # never got its turn

    def test_a_run_in_flight_reports_itself_as_running(self) -> None:
        graph = PipelineGraph(
            nodes=[GraphNode(id="slow", block_id="Slow.predict", inputs={"sequence": "MKTV"})]
        )
        record = RunRecord.start(graph, catalog=CATALOG)
        try:
            assert record.running
            assert record.to_dict()["steps"][0]["state"] == "running"
        finally:
            _Slow.released.set()
            record.wait(timeout=10)
        assert record.state == "succeeded"


class TestNotifications:
    def test_the_listener_sees_the_run_finish(self) -> None:
        seen: list[str] = []
        record = _run(_graph(), on_change=lambda r: seen.append(r.state))
        assert seen
        assert seen[-1] == record.state == "succeeded"

    def test_a_broken_listener_does_not_kill_the_run(self) -> None:
        """A dashboard that disconnects mid-run must not take the pipeline
        with it."""

        def explode(_record: RunRecord) -> None:
            raise RuntimeError("the websocket went away")

        record = _run(_graph(), on_change=explode)
        assert record.state == "succeeded"


class TestPayload:
    def test_it_serializes_for_the_wire(self) -> None:
        import json

        payload = json.loads(json.dumps(_run(_graph()).to_dict()))
        assert payload["state"] == "succeeded"
        assert [s["node_id"] for s in payload["steps"]] == ["fold", "dock"]

    def test_no_result_object_is_retained(self) -> None:
        """Holding every intermediate alive for the life of the server is
        how a dashboard becomes the reason a machine is out of memory."""
        record = _run(_graph())
        for step in record.steps:
            assert isinstance(step.output, str)


class TestDescribeOutput:
    def test_none_is_empty(self) -> None:
        assert describe_output(None) == ""

    def test_a_scalar_is_itself(self) -> None:
        assert describe_output("structure(MKTV)") == "structure(MKTV)"

    def test_a_sequence_reports_its_size(self) -> None:
        """An ensemble's size is the thing worth seeing: it says whether
        the step produced the sample count that was asked for."""
        assert describe_output(["a", "b", "c"]) == "3 × str"
        assert describe_output([]) == "empty"

    def test_a_structure_leads_with_its_residue_count(self) -> None:
        """The size is what says the step folded what you meant rather than
        a fragment of it. Read off the attribute Protein actually exposes."""

        class FakeProtein:
            n_residues = 214
            n_atoms = 1673

        assert describe_output(FakeProtein()) == "FakeProtein (214 residues)"

    def test_an_alignment_leads_with_its_depth(self) -> None:
        """Depth is the number that decides the fold that consumes it."""

        class FakeAlignment:
            depth = 1024

        assert describe_output(FakeAlignment()) == "FakeAlignment (1024 sequences)"

    def test_a_sized_attribute_is_counted(self) -> None:
        class FakeDocking:
            poses: ClassVar[list[object]] = [object(), object(), object()]

        assert describe_output(FakeDocking()) == "FakeDocking (3 poses)"

    def test_an_object_with_nothing_countable_still_describes_itself(self) -> None:
        class Opaque:
            pass

        assert describe_output(Opaque()) == "Opaque"
