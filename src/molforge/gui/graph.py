"""The canvas's graph, and its translation to a ``pipeline.yaml``.

What the editor draws is a graph of blocks and wires. What molforge
already has is :class:`~molforge.reproducibility.PipelineManifest` — the
citable artifact a methods section points at, and the thing
:func:`~molforge.reproducibility.replay` re-executes. This module is the
translation between them, in both directions::

    from molforge.gui.graph import PipelineGraph

    graph = PipelineGraph.from_dict(payload)
    graph.validate()                 # refuse a bad wiring, with the node to blame
    manifest, context = graph.to_manifest()
    emit_pipeline_manifest(manifest, "pipeline.yaml")

A graph saved from the canvas is therefore an ordinary manifest: runnable
with ``replay()`` by someone who has never opened the GUI, diffable, and
citable. That constraint — the editor's save format is the library's
existing format, not a format of its own — is what keeps the GUI from
becoming a second way to describe a pipeline that only it understands.

Two consequences of translating *to* a manifest, both of which the editor
has to surface rather than hide:

**Chains, not trees.** ``replay()`` threads each step's output into the
next, so a manifest is a linear sequence. A canvas can draw a fold feeding
three different docking blocks; that is a legitimate thing to want and it
is not a pipeline manifest. :meth:`PipelineGraph.validate` rejects it here,
with the node to blame, rather than letting it serialize into something
that replays as one arbitrary branch.

**Unwired inputs become ``context``.** A docking block's ligand usually
isn't produced by another block — it's a file or a SMILES the user typed.
Those become the ``context`` dict ``replay()`` takes. ``context`` is keyed
by input *name* and shared across every step, so two blocks that both need
an unwired ``ligand`` cannot be distinguished; that is a real limitation of
the replay API and is reported as such instead of silently resolving to
whichever one was written last.

Node positions are deliberately *not* part of the manifest. A manifest
describes what ran, and where someone dragged a box on a canvas is not
that. Loading one lays the chain out left-to-right instead, which also
means a ``pipeline.yaml`` from a colleague's script opens as a diagram
with no GUI-specific metadata in it at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from molforge.gui.catalog import build_catalog

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from molforge.gui.catalog import BlockSpec
    from molforge.reproducibility import PipelineManifest

__all__ = [
    "GraphEdge",
    "GraphError",
    "GraphNode",
    "PipelineGraph",
]

#: Horizontal spacing between laid-out nodes, in canvas units. Only used
#: when importing a manifest, which carries no positions.
_LAYOUT_STEP_X = 320
_LAYOUT_Y = 160


class GraphError(ValueError):
    """A graph that cannot be run, and the node responsible.

    Attributes:
        node_id: The node to highlight, or ``""`` for a whole-graph
            problem (no nodes, several disconnected chains).
    """

    def __init__(self, message: str, *, node_id: str = "") -> None:
        super().__init__(message)
        self.node_id = node_id


@dataclass
class GraphNode:
    """One block placed on the canvas.

    Attributes:
        id: Canvas-unique identifier. Distinct from ``block_id`` because a
            graph can hold two Vina blocks docking different ligands.
        block_id: Which catalog block this is (``"Vina.dock"``).
        parameters: Values for the block's editable settings. Only those
            the catalog declares are kept — a stale key from an older
            graph is dropped rather than passed to a constructor that will
            reject it.
        inputs: Literal values for ports that aren't wired (a sequence
            typed in, a path to a ligand).
        position: ``(x, y)`` on the canvas. Round-trips through the
            editor's own save format, never into a manifest.
    """

    id: str
    block_id: str
    parameters: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)
    position: tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "block_id": self.block_id,
            "parameters": dict(self.parameters),
            "inputs": dict(self.inputs),
            "position": list(self.position),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GraphNode:
        position = data.get("position") or (0.0, 0.0)
        return cls(
            id=str(data["id"]),
            block_id=str(data["block_id"]),
            parameters=dict(data.get("parameters") or {}),
            inputs=dict(data.get("inputs") or {}),
            position=(float(position[0]), float(position[1])),
        )


@dataclass
class GraphEdge:
    """A wire from one block's output into another block's input port.

    Attributes:
        source: Node id the value comes from.
        target: Node id it goes to.
        target_port: Which input port it lands on — required, because a
            docking block has two and "the wire goes to Vina" doesn't say
            whether it's the receptor or the ligand.
    """

    source: str
    target: str
    target_port: str

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "target": self.target, "target_port": self.target_port}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GraphEdge:
        return cls(
            source=str(data["source"]),
            target=str(data["target"]),
            target_port=str(data["target_port"]),
        )


@dataclass
class PipelineGraph:
    """A drawn pipeline: blocks, wires, and the checks that make it runnable."""

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)

    # ---------- serialization (the editor's own format) ----------

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PipelineGraph:
        """Rebuild from the editor's payload.

        Raises:
            GraphError: If a node or edge is missing a field it cannot be
                defaulted into. A malformed payload is reported as a graph
                problem rather than a ``KeyError`` from inside a dataclass.
        """
        try:
            nodes = [GraphNode.from_dict(n) for n in data.get("nodes") or ()]
            edges = [GraphEdge.from_dict(e) for e in data.get("edges") or ()]
        except (KeyError, TypeError, ValueError, IndexError) as e:
            raise GraphError(f"malformed graph payload: {e}") from e
        return cls(nodes=nodes, edges=edges)

    # ---------- lookup ----------

    def node(self, node_id: str) -> GraphNode:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise GraphError(f"no node with id {node_id!r}", node_id=node_id)

    # ---------- validation ----------

    def validate(
        self,
        *,
        catalog: Sequence[BlockSpec] | None = None,
        require_available: bool = False,
    ) -> None:
        """Check the graph can be linearized into a runnable manifest.

        Every failure names the node to blame so the editor can highlight
        it; the messages are written to be read in a toast, not a log.

        Args:
            catalog: Blocks to validate against. Defaults to the live
                catalog; passed explicitly by tests and by a server that
                built it once per request.
            require_available: Also require every engine to be installed
                here. Off by default, because drawing and saving a pipeline
                for a machine other than this one is a thing a portable
                ``pipeline.yaml`` exists to allow — a graph built on a
                laptop for a GPU cluster is correct, just not runnable
                yet. :meth:`run` turns it on.

        Raises:
            GraphError: On an unknown block, a duplicate node id, a wire
                between mismatched port types, a wire to a port that
                doesn't exist, a branch or a cycle, a disconnected graph,
                a missing required input, or two unwired inputs that would
                collide in ``replay``'s shared context.
        """
        blocks = {b.id: b for b in (build_catalog() if catalog is None else catalog)}
        if not self.nodes:
            raise GraphError("the canvas is empty — add a block to run something.")

        self._check_nodes(blocks, require_available=require_available)
        self._check_edges(blocks)
        self._check_shape()
        self._check_inputs(blocks)
        self._check_parameters(blocks)

    def _check_nodes(
        self, blocks: Mapping[str, BlockSpec], *, require_available: bool = False
    ) -> None:
        seen: set[str] = set()
        for node in self.nodes:
            if node.id in seen:
                raise GraphError(f"two nodes share the id {node.id!r}", node_id=node.id)
            seen.add(node.id)
            block = blocks.get(node.block_id)
            if block is None:
                raise GraphError(
                    f"unknown block {node.block_id!r}. It may come from a newer "
                    "molforge, or from an engine this install doesn't register.",
                    node_id=node.id,
                )
            if require_available and not block.available:
                raise GraphError(
                    f"{block.engine} isn't available here: {block.detail}",
                    node_id=node.id,
                )

    def _check_edges(self, blocks: Mapping[str, BlockSpec]) -> None:
        wired: set[tuple[str, str]] = set()
        ids = {n.id for n in self.nodes}
        for edge in self.edges:
            for end, label in ((edge.source, "source"), (edge.target, "target")):
                if end not in ids:
                    raise GraphError(f"wire {label} {end!r} is not a node on this canvas")
            target = self.node(edge.target)
            block = blocks[target.block_id]
            port = next((p for p in block.inputs if p.name == edge.target_port), None)
            if port is None:
                names = [p.name for p in block.inputs] or ["(none)"]
                raise GraphError(
                    f"{block.engine} has no input called {edge.target_port!r} "
                    f"— it takes {', '.join(names)}.",
                    node_id=edge.target,
                )
            if (edge.target, edge.target_port) in wired:
                raise GraphError(
                    f"two wires feed {block.engine}'s {port.name}; it takes one.",
                    node_id=edge.target,
                )
            wired.add((edge.target, edge.target_port))

            produced = blocks[self.node(edge.source).block_id].output.type
            if produced != port.type:
                raise GraphError(
                    f"can't wire a {produced} into {block.engine}'s {port.name}, "
                    f"which takes a {port.type}.",
                    node_id=edge.target,
                )

    def _check_shape(self) -> None:
        """Reject anything that isn't a single chain.

        A manifest is a sequence, so a branch, a merge of two chains, or a
        second disconnected chain cannot be represented. Each is a sensible
        thing to draw and an unrepresentable thing to save, which is worth
        saying plainly at the node where it happens.
        """
        outgoing: dict[str, int] = dict.fromkeys((n.id for n in self.nodes), 0)
        incoming: dict[str, int] = dict.fromkeys((n.id for n in self.nodes), 0)
        for edge in self.edges:
            outgoing[edge.source] += 1
            incoming[edge.target] += 1

        for node_id, count in outgoing.items():
            if count > 1:
                raise GraphError(
                    "a pipeline manifest is a single chain, and this block feeds "
                    f"{count} others. Run them as separate pipelines, or chain them.",
                    node_id=node_id,
                )

        self._order()  # raises on a cycle or a disconnected graph

    def _order(self) -> list[GraphNode]:
        """The nodes in execution order, oldest first.

        Walks from the one node with no input wire. A graph with no such
        node is a cycle; a walk that doesn't reach every node means there
        is more than one chain on the canvas.
        """
        by_source = {e.source: e for e in self.edges}
        wired_targets = {e.target for e in self.edges}
        roots = [n for n in self.nodes if n.id not in wired_targets]
        if not roots:
            raise GraphError("these blocks form a loop, so there's no step to start from.")
        if len(roots) > 1:
            raise GraphError(
                f"this canvas holds {len(roots)} unconnected chains. A pipeline is "
                "one chain — wire them together, or save them separately.",
                node_id=roots[1].id,
            )

        order: list[GraphNode] = []
        seen: set[str] = set()
        current: GraphNode | None = roots[0]
        while current is not None:
            if current.id in seen:  # pragma: no cover - the root check catches cycles
                raise GraphError("these blocks form a loop.", node_id=current.id)
            seen.add(current.id)
            order.append(current)
            edge = by_source.get(current.id)
            current = self.node(edge.target) if edge is not None else None

        if len(order) != len(self.nodes):
            stranded = next(n for n in self.nodes if n.id not in seen)
            raise GraphError(
                "this block isn't connected to the rest of the pipeline.",
                node_id=stranded.id,
            )
        return order

    def _check_inputs(self, blocks: Mapping[str, BlockSpec]) -> None:
        """Every required input is either wired or typed in, and no two
        unwired inputs share a name."""
        wired = {(e.target, e.target_port) for e in self.edges}
        supplied: dict[str, str] = {}
        for node in self._order():
            block = blocks[node.block_id]
            for port in block.inputs:
                if (node.id, port.name) in wired:
                    continue
                value = node.inputs.get(port.name)
                if value in (None, "") and port.required:
                    raise GraphError(
                        f"{block.engine} needs {port.name} — wire a block into it "
                        "or enter a value.",
                        node_id=node.id,
                    )
                if value in (None, ""):
                    continue
                if port.name in supplied and supplied[port.name] != node.id:
                    raise GraphError(
                        f"two blocks each supply their own {port.name}. Replay shares "
                        "one value per input name across the pipeline, so they would "
                        "collide — wire one of them instead.",
                        node_id=node.id,
                    )
                supplied[port.name] = node.id

    def _check_parameters(self, blocks: Mapping[str, BlockSpec]) -> None:
        """Parameters with no default have to be set.

        Docking's ``center`` is the whole reason this check exists: it has
        no default, it decides whether the run docks to the pocket or to
        empty space, and a graph missing it would otherwise serialize
        happily and fail at the engine.
        """
        for node in self.nodes:
            block = blocks[node.block_id]
            for param in block.parameters:
                if not param.required:
                    continue
                if node.parameters.get(param.name) in (None, ""):
                    raise GraphError(
                        f"{block.engine} needs {param.name} set — it has no default.",
                        node_id=node.id,
                    )

    # ---------- translation ----------

    def to_manifest(
        self, *, catalog: Sequence[BlockSpec] | None = None
    ) -> tuple[PipelineManifest, dict[str, Any]]:
        """Linearize into a manifest and the ``context`` needed to replay it.

        Validates first: a manifest that can't run is worse than a refusal,
        because it looks like a saved pipeline.

        Returns:
            ``(manifest, context)``. The manifest is the citable artifact;
            the context carries the unwired inputs — the ligand file, the
            sequence someone typed — that the manifest records as literals
            but ``replay()`` resolves by name.
        """
        from molforge.reproducibility import PipelineManifest, PipelineStep

        specs = tuple(build_catalog() if catalog is None else catalog)
        self.validate(catalog=specs)
        blocks = {b.id: b for b in specs}

        wired = {(e.target, e.target_port) for e in self.edges}
        steps: list[PipelineStep] = []
        context: dict[str, Any] = {}

        for position, node in enumerate(self._order(), start=1):
            block = blocks[node.block_id]
            inputs = {
                port.name: node.inputs[port.name]
                for port in block.inputs
                if (node.id, port.name) not in wired
                and node.inputs.get(port.name) not in (None, "")
            }
            context.update(inputs)
            steps.append(
                PipelineStep(
                    step=position,
                    engine=block.engine,
                    operation=block.operation,
                    engine_version=block.version,
                    inputs=inputs,
                    parameters=self._parameters(block, node),
                )
            )

        manifest = PipelineManifest(environment=_environment(), steps=steps)
        return manifest, context

    @staticmethod
    def _parameters(block: BlockSpec, node: GraphNode) -> dict[str, Any]:
        """The node's settings, filtered to what the block declares.

        A graph saved against an older molforge can carry a parameter the
        wrapper has since dropped. Replay's ``_construct`` would filter it
        anyway, but doing it here means the manifest records what will
        actually be used rather than what the canvas happened to hold.
        """
        declared = {p.name for p in block.parameters}
        return {k: v for k, v in node.parameters.items() if k in declared}

    # ---------- import ----------

    @classmethod
    def from_manifest(
        cls, manifest: PipelineManifest, *, catalog: Sequence[BlockSpec] | None = None
    ) -> PipelineGraph:
        """Open a ``pipeline.yaml`` as a diagram.

        The manifest carries no positions, so the chain is laid out
        left-to-right. That is the point: a colleague's script emits an
        ordinary manifest, and it opens here as a graph without anyone
        having added editor metadata to it.

        Raises:
            GraphError: If a step names an engine-and-operation pair this
                install has no block for — a manifest from a newer molforge,
                or one whose engine isn't registered here.
        """
        specs = tuple(build_catalog() if catalog is None else catalog)
        by_id = {b.id: b for b in specs}

        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        for index, step in enumerate(manifest.steps):
            block_id = f"{step.engine}.{step.operation}"
            block = by_id.get(block_id)
            if block is None:
                raise GraphError(
                    f"step {step.step} ran {step.engine} {step.operation!r}, which this "
                    "install has no block for. The engine may not be registered here."
                )
            node_id = f"step-{step.step}"
            nodes.append(
                GraphNode(
                    id=node_id,
                    block_id=block_id,
                    parameters=dict(step.parameters),
                    inputs=dict(step.inputs),
                    position=(float(index * _LAYOUT_STEP_X), float(_LAYOUT_Y)),
                )
            )
            if index:
                previous = nodes[index - 1]
                port = _threaded_port(block, by_id[previous.block_id])
                if port is not None:
                    edges.append(GraphEdge(source=previous.id, target=node_id, target_port=port))
                    nodes[index].inputs.pop(port, None)
        return cls(nodes=nodes, edges=edges)


def _threaded_port(block: BlockSpec, upstream: BlockSpec) -> str | None:
    """Which of ``block``'s ports the previous step's output lands on.

    Replay threads the output into the step's primary input; on the canvas
    that has to become an actual wire to a named port. Chosen by port type
    so a docking step's receptor is wired and its ligand is left as the
    literal the manifest recorded — which is how the run actually happened.
    """
    for port in block.inputs:
        if port.type == upstream.output.type:
            return port.name
    return None


def _environment() -> dict[str, Any]:
    """The environment block for a manifest the editor is about to save.

    A manifest emitted from a *drawn* graph describes a pipeline that has
    not run yet, so there are no engine versions to report — only what is
    installed here, which is the weaker ``engines_detected`` claim the
    reproducibility module already distinguishes.
    """
    import platform
    import sys

    from molforge import __version__, engine_versions

    detected = {
        name: backend.version
        for name, backend in engine_versions(probe_executables=False).items()
        if backend.version and backend.category != "runtime"
    }
    return {
        "molforge_version": __version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "engines": {},
        "engines_detected": detected,
        "note": f"drawn in the molforge editor on {sys.platform}; not yet run",
    }


def layout(nodes: Iterable[GraphNode]) -> list[GraphNode]:
    """Lay nodes out left-to-right, for a graph that arrived without positions."""
    return [
        GraphNode(
            id=node.id,
            block_id=node.block_id,
            parameters=node.parameters,
            inputs=node.inputs,
            position=(float(index * _LAYOUT_STEP_X), float(_LAYOUT_Y)),
        )
        for index, node in enumerate(nodes)
    ]
