"""Graph validation and its translation to a pipeline manifest.

These build against a fixed, synthetic catalog rather than the live one.
The live catalog depends on what happens to be installed — a machine with
no Vina would skip every docking case — and these are assertions about
wiring rules, which are the same everywhere.
"""

from __future__ import annotations

import pytest

from molforge.gui.catalog import BlockParameter, BlockPort, BlockSpec
from molforge.gui.graph import GraphEdge, GraphError, GraphNode, PipelineGraph


def _block(
    block_id: str,
    *,
    inputs: tuple[BlockPort, ...] = (),
    output: str = "structure",
    parameters: tuple[BlockParameter, ...] = (),
    available: bool = True,
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
        available=available,
        detail="" if available else "not installed here",
    )


#: A catalog with the shapes that matter: a source, a one-input step, a
#: two-input step with a required parameter, and an absent engine.
CATALOG = (
    _block("Folder.predict", inputs=(BlockPort("sequence", "sequence"),)),
    _block("Searcher.search_msa", inputs=(BlockPort("sequence", "sequence"),), output="alignment"),
    _block(
        "Docker.dock",
        inputs=(BlockPort("receptor", "structure"), BlockPort("ligand", "ligand")),
        output="docking_result",
        parameters=(
            BlockParameter("center", "vector3", "tuple[float, float, float]", required=True),
            BlockParameter("exhaustiveness", "int", "int", default=8),
        ),
    ),
    _block("Pockets.detect_pockets", inputs=(BlockPort("protein", "structure"),), output="pockets"),
    _block("Absent.predict", inputs=(BlockPort("sequence", "sequence"),), available=False),
    # Consumes and produces a structure, so a cycle through it is
    # type-valid and reaches the shape check rather than the port check.
    _block("Refiner.predict", inputs=(BlockPort("receptor", "structure"),)),
    # Takes a structure *and* a sequence, which is what makes an unwired
    # name collision reachable inside a single chain.
    _block(
        "Threader.predict",
        inputs=(BlockPort("receptor", "structure"), BlockPort("sequence", "sequence")),
    ),
)


def _fold(node_id: str = "fold", sequence: str = "MKTV") -> GraphNode:
    return GraphNode(id=node_id, block_id="Folder.predict", inputs={"sequence": sequence})


def _dock(node_id: str = "dock", **inputs: object) -> GraphNode:
    return GraphNode(
        id=node_id,
        block_id="Docker.dock",
        inputs={"ligand": "aspirin.sdf", **inputs},
        parameters={"center": [1.0, 2.0, 3.0]},
    )


def _chain() -> PipelineGraph:
    """The canonical pipeline: fold a sequence, dock a ligand against it."""
    return PipelineGraph(
        nodes=[_fold(), _dock()],
        edges=[GraphEdge(source="fold", target="dock", target_port="receptor")],
    )


class TestValidation:
    def test_a_well_formed_chain_passes(self) -> None:
        _chain().validate(catalog=CATALOG)

    def test_an_empty_canvas_is_refused(self) -> None:
        with pytest.raises(GraphError, match="canvas is empty"):
            PipelineGraph().validate(catalog=CATALOG)

    def test_unknown_block_names_the_node(self) -> None:
        graph = PipelineGraph(nodes=[GraphNode(id="x", block_id="Nope.predict")])
        with pytest.raises(GraphError, match="unknown block") as e:
            graph.validate(catalog=CATALOG)
        assert e.value.node_id == "x"

    def test_duplicate_node_ids_are_refused(self) -> None:
        """The editor addresses nodes by id; a duplicate would make one of
        them unreachable."""
        graph = PipelineGraph(nodes=[_fold("same"), _fold("same")])
        with pytest.raises(GraphError, match="share the id"):
            graph.validate(catalog=CATALOG)

    def test_mismatched_port_types_are_refused(self) -> None:
        """The error a typed canvas exists to make impossible to draw."""
        graph = PipelineGraph(
            nodes=[
                GraphNode(id="msa", block_id="Searcher.search_msa", inputs={"sequence": "MKTV"}),
                _dock(),
            ],
            edges=[GraphEdge(source="msa", target="dock", target_port="receptor")],
        )
        with pytest.raises(GraphError, match="can't wire a alignment") as e:
            graph.validate(catalog=CATALOG)
        assert e.value.node_id == "dock"

    def test_wiring_to_a_port_that_does_not_exist(self) -> None:
        graph = PipelineGraph(
            nodes=[_fold(), _dock()],
            edges=[GraphEdge(source="fold", target="dock", target_port="protein")],
        )
        with pytest.raises(GraphError, match="no input called 'protein'"):
            graph.validate(catalog=CATALOG)

    def test_two_wires_into_one_port(self) -> None:
        graph = PipelineGraph(
            nodes=[_fold("a"), _fold("b"), _dock()],
            edges=[
                GraphEdge(source="a", target="dock", target_port="receptor"),
                GraphEdge(source="b", target="dock", target_port="receptor"),
            ],
        )
        with pytest.raises(GraphError, match="takes one"):
            graph.validate(catalog=CATALOG)

    def test_an_edge_to_a_node_not_on_the_canvas(self) -> None:
        graph = PipelineGraph(
            nodes=[_fold()],
            edges=[GraphEdge(source="fold", target="ghost", target_port="receptor")],
        )
        with pytest.raises(GraphError, match="not a node on this canvas"):
            graph.validate(catalog=CATALOG)


class TestShape:
    def test_a_branch_is_refused_with_the_reason(self) -> None:
        """Legitimate to want, impossible to express as a manifest — so it
        is refused at the branching node, not silently serialized."""
        graph = PipelineGraph(
            nodes=[_fold(), _dock("dock1"), _dock("dock2")],
            edges=[
                GraphEdge(source="fold", target="dock1", target_port="receptor"),
                GraphEdge(source="fold", target="dock2", target_port="receptor"),
            ],
        )
        with pytest.raises(GraphError, match="single chain") as e:
            graph.validate(catalog=CATALOG)
        assert e.value.node_id == "fold"

    def test_two_disconnected_chains_are_refused(self) -> None:
        graph = PipelineGraph(nodes=[_fold("a"), _fold("b")])
        with pytest.raises(GraphError, match="unconnected chains"):
            graph.validate(catalog=CATALOG)

    def test_a_cycle_has_no_starting_step(self) -> None:
        graph = PipelineGraph(
            nodes=[
                GraphNode(id="a", block_id="Refiner.predict"),
                GraphNode(id="b", block_id="Refiner.predict"),
            ],
            edges=[
                GraphEdge(source="a", target="b", target_port="receptor"),
                GraphEdge(source="b", target="a", target_port="receptor"),
            ],
        )
        with pytest.raises(GraphError, match="loop"):
            graph.validate(catalog=CATALOG)

    def test_a_long_chain_orders_correctly(self) -> None:
        graph = PipelineGraph(
            nodes=[_fold(), GraphNode(id="pock", block_id="Pockets.detect_pockets")],
            edges=[GraphEdge(source="fold", target="pock", target_port="protein")],
        )
        manifest, _ = graph.to_manifest(catalog=CATALOG)
        assert [s.engine for s in manifest.steps] == ["Folder", "Pockets"]
        assert [s.step for s in manifest.steps] == [1, 2]


class TestInputs:
    def test_a_missing_required_input_is_refused(self) -> None:
        graph = PipelineGraph(nodes=[GraphNode(id="fold", block_id="Folder.predict")])
        with pytest.raises(GraphError, match="needs sequence") as e:
            graph.validate(catalog=CATALOG)
        assert e.value.node_id == "fold"

    def test_a_wired_input_need_not_be_typed_in(self) -> None:
        _chain().validate(catalog=CATALOG)

    def test_colliding_unwired_inputs_are_refused(self) -> None:
        """``replay``'s context is keyed by input name and shared across
        steps, so two typed-in ligands cannot be told apart. A real limit
        of the replay API, reported rather than silently resolved."""
        graph = PipelineGraph(
            nodes=[
                _fold(sequence="MKTV"),
                GraphNode(
                    id="thread",
                    block_id="Threader.predict",
                    inputs={"sequence": "A DIFFERENT SEQUENCE"},
                ),
            ],
            edges=[GraphEdge(source="fold", target="thread", target_port="receptor")],
        )
        with pytest.raises(GraphError, match="collide") as e:
            graph.validate(catalog=CATALOG)
        assert e.value.node_id == "thread"


class TestParameters:
    def test_a_required_parameter_must_be_set(self) -> None:
        """``center`` decides whether a run docks to the pocket or to empty
        space, and has no default to fall back on."""
        graph = _chain()
        graph.node("dock").parameters.pop("center")
        with pytest.raises(GraphError, match="needs center") as e:
            graph.validate(catalog=CATALOG)
        assert e.value.node_id == "dock"

    def test_unknown_parameters_are_dropped_not_passed_on(self) -> None:
        """A graph saved against an older molforge can carry a knob the
        wrapper has since dropped."""
        graph = _chain()
        graph.node("dock").parameters["removed_in_0_11"] = True
        manifest, _ = graph.to_manifest(catalog=CATALOG)
        assert "removed_in_0_11" not in manifest.steps[1].parameters
        assert manifest.steps[1].parameters["center"] == [1.0, 2.0, 3.0]


class TestAvailability:
    def test_saving_a_graph_for_another_machine_is_allowed(self) -> None:
        """A pipeline drawn on a laptop for a GPU cluster is correct, just
        not runnable here — refusing to save it would defeat the point of a
        portable manifest."""
        graph = PipelineGraph(
            nodes=[GraphNode(id="x", block_id="Absent.predict", inputs={"sequence": "MKTV"})]
        )
        graph.validate(catalog=CATALOG)
        manifest, _ = graph.to_manifest(catalog=CATALOG)
        assert manifest.steps[0].engine == "Absent"

    def test_running_requires_the_engine_to_be_here(self) -> None:
        graph = PipelineGraph(
            nodes=[GraphNode(id="x", block_id="Absent.predict", inputs={"sequence": "MKTV"})]
        )
        with pytest.raises(GraphError, match="isn't available here") as e:
            graph.validate(catalog=CATALOG, require_available=True)
        assert e.value.node_id == "x"


class TestManifest:
    def test_unwired_inputs_become_the_replay_context(self) -> None:
        _, context = _chain().to_manifest(catalog=CATALOG)
        assert context == {"sequence": "MKTV", "ligand": "aspirin.sdf"}

    def test_wired_inputs_are_not_in_the_context(self) -> None:
        """The receptor comes from the previous step; putting it in context
        would override the thing the pipeline computed."""
        _, context = _chain().to_manifest(catalog=CATALOG)
        assert "receptor" not in context

    def test_steps_record_the_operation_replay_dispatches_on(self) -> None:
        manifest, _ = _chain().to_manifest(catalog=CATALOG)
        assert [s.operation for s in manifest.steps] == ["predict", "dock"]

    def test_the_environment_block_claims_only_what_is_installed(self) -> None:
        """A drawn graph hasn't run, so there are no engine versions to
        report — only the weaker 'detected here' claim."""
        manifest, _ = _chain().to_manifest(catalog=CATALOG)
        assert manifest.environment["engines"] == {}
        assert "engines_detected" in manifest.environment
        assert manifest.environment["molforge_version"]

    def test_an_invalid_graph_does_not_serialize(self) -> None:
        """A saved manifest that cannot run is worse than a refusal: it
        looks like a working pipeline."""
        graph = PipelineGraph(nodes=[GraphNode(id="fold", block_id="Folder.predict")])
        with pytest.raises(GraphError):
            graph.to_manifest(catalog=CATALOG)


class TestRoundTrip:
    def test_graph_survives_its_own_format(self) -> None:
        graph = _chain()
        restored = PipelineGraph.from_dict(graph.to_dict())
        assert restored.to_dict() == graph.to_dict()

    def test_a_manifest_reopens_as_the_same_chain(self) -> None:
        manifest, _ = _chain().to_manifest(catalog=CATALOG)
        reopened = PipelineGraph.from_manifest(manifest, catalog=CATALOG)
        assert [n.block_id for n in reopened.nodes] == ["Folder.predict", "Docker.dock"]
        assert [(e.source, e.target, e.target_port) for e in reopened.edges] == [
            ("step-1", "step-2", "receptor")
        ]

    def test_a_reopened_manifest_still_validates(self) -> None:
        manifest, _ = _chain().to_manifest(catalog=CATALOG)
        PipelineGraph.from_manifest(manifest, catalog=CATALOG).validate(catalog=CATALOG)

    def test_reopening_lays_the_chain_out(self) -> None:
        """A manifest carries no positions — it describes what ran, not
        where someone dragged a box."""
        manifest, _ = _chain().to_manifest(catalog=CATALOG)
        positions = [
            n.position for n in PipelineGraph.from_manifest(manifest, catalog=CATALOG).nodes
        ]
        assert positions[0][0] < positions[1][0]
        assert len(set(positions)) == len(positions)

    def test_the_threaded_input_becomes_a_wire_not_a_literal(self) -> None:
        """The receptor was computed by the previous step; reopening must
        show that as an arrow, not as a value someone typed."""
        manifest, _ = _chain().to_manifest(catalog=CATALOG)
        reopened = PipelineGraph.from_manifest(manifest, catalog=CATALOG)
        assert "receptor" not in reopened.node("step-2").inputs
        assert reopened.node("step-2").inputs["ligand"] == "aspirin.sdf"

    def test_a_step_with_no_block_here_is_reported(self) -> None:
        import dataclasses

        manifest, _ = _chain().to_manifest(catalog=CATALOG)
        manifest.steps[1] = dataclasses.replace(manifest.steps[1], engine="SomethingElse")
        with pytest.raises(GraphError, match="no block for"):
            PipelineGraph.from_manifest(manifest, catalog=CATALOG)

    def test_a_malformed_payload_is_a_graph_error(self) -> None:
        """The server hands this straight to a client; a KeyError from
        inside a dataclass would surface as a 500."""
        with pytest.raises(GraphError, match="malformed"):
            PipelineGraph.from_dict({"nodes": [{"block_id": "Folder.predict"}]})
