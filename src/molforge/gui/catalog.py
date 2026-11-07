"""The block catalog: every engine molforge can drive, as a GUI block.

A node-graph editor needs to know what blocks exist, what each one can be
wired to, and which knobs it exposes. That description already exists in
the library — as the engine registry :func:`~molforge.reproducibility.replay`
resolves against, and as the signatures of the wrapper classes themselves.
This module derives the catalog from those two sources rather than
restating them::

    from molforge.gui.catalog import build_catalog

    for block in build_catalog():
        print(block.id, block.output.type, block.available)

Nothing here is a hand-maintained list. A new wrapper that registers itself
and records an operation replay knows appears as a block with no edit here,
and — more to the point — a wrapper whose signature changes cannot leave a
stale parameter behind in the GUI. The catalog is wrong only when the
library is.

Three things each block carries:

**Ports.** A block's input and output *port types* are semantic, not Python
types: ``structure``, ``sequence``, ``ligand``, ``alignment``,
``docking_result``, ``pockets``, ``designs``. They come from the
operation's annotations, so the editor can refuse a wire from a docking
result into a folding block — the kind of mistake that is obvious in a
diagram and invisible in a script.

**Parameters.** Constructor arguments, each with the widget kind its
annotation implies: a ``Literal`` becomes a dropdown with its own options,
a ``bool`` a checkbox, ``tuple[float, float, float]`` the three-field box
coordinate editor every docking engine needs. Annotations are read as
strings (every wrapper module uses ``from __future__ import annotations``),
which is deliberate: the catalog never imports torch to find out that
ESMFold takes a ``device``.

**Availability.** Whether the backend is installed right now, from
:func:`molforge.engine_versions`. A block for an absent engine is still
listed — knowing molforge can drive Gnina is worth more than hiding it —
but flagged, so the editor can grey it out and say why instead of letting
someone wire up a graph that cannot run.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

__all__ = [
    "BlockParameter",
    "BlockPort",
    "BlockSpec",
    "build_catalog",
    "catalog_to_dict",
]


# ---------------------------------------------------------------------------
# Port types
# ---------------------------------------------------------------------------

#: Semantic port type per result annotation. Keyed by the bare annotation
#: text a wrapper writes, so the mapping is checkable against the source.
#: ``list[Protein]`` is a separate type from ``Protein`` on purpose: an
#: ensemble and a single structure are not interchangeable downstream, and
#: silently feeding ten samples to a docking block that wants one receptor
#: is exactly the error a typed canvas should catch.
_RETURN_PORTS: dict[str, str] = {
    "Protein": "structure",
    "list[Protein]": "structure_ensemble",
    "DockingResult": "docking_result",
    "MultipleSequenceAlignment": "alignment",
    "list[Pocket]": "pockets",
    "list[DesignedSequence]": "designs",
    "Molecule": "ligand",
}

#: Semantic port type per *input* parameter name. Input annotations are
#: unions wide enough to be useless as a wire type — docking's ``receptor``
#: is ``Protein | str | PathLike[str]``, which describes how forgiving the
#: wrapper is, not what a graph should hand it. The parameter name is the
#: stronger signal, and the set of names is small and stable because it is
#: the vocabulary the replay handlers already resolve by.
_INPUT_PORTS: dict[str, str] = {
    "sequence": "sequence",
    "backbone": "structure",
    "receptor": "structure",
    "protein": "structure",
    "ligand": "ligand",
    "spec": "complex_spec",
    "complex_spec": "complex_spec",
    "pdb_id": "pdb_id",
}


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Operation:
    """One replayable operation and the method that performs it.

    ``operation`` is the string a wrapper records in provenance and replay
    dispatches on; ``method`` is what to call. They differ where the
    recorded name describes the step and the method name describes the
    call (``search_msa`` / ``search``), and a block has to carry both: the
    editor shows the method, the manifest records the operation.
    """

    operation: str
    method: str
    label: str


#: Operations a block can represent, in the order a pipeline tends to run
#: them. Deliberately a subset of the replay handlers: every entry here
#: must have a handler, which :mod:`tests` asserts, but an operation whose
#: inputs a canvas cannot express yet (``predict_affinity`` needs a built
#: complex spec) is left out until the editor can build one.
_OPERATIONS: tuple[_Operation, ...] = (
    _Operation("fetch", "__call__", "Fetch structure"),
    _Operation("search_msa", "search", "Search homologues"),
    _Operation("predict", "predict", "Predict structure"),
    _Operation("predict_samples", "predict_samples", "Predict ensemble"),
    _Operation("detect_pockets", "__call__", "Detect pockets"),
    _Operation("dock", "dock", "Dock ligand"),
    _Operation("generate", "generate", "Generate sequences"),
)


# ---------------------------------------------------------------------------
# Catalog types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockPort:
    """One input or output socket on a block.

    Attributes:
        name: The argument the wired value is passed as — the key a graph
            puts in ``replay``'s ``context``, which is why it matches the
            parameter name rather than being a display label.
        type: Semantic port type (``"structure"``, ``"ligand"``, ...).
            Two ports connect only when their types match.
        required: Whether the operation refuses to run without it.
    """

    name: str
    type: str
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "type": self.type, "required": self.required}


@dataclass(frozen=True)
class BlockParameter:
    """One editable setting on a block, with the widget its type implies.

    Attributes:
        name: Constructor keyword.
        kind: Widget hint — ``"choice"``, ``"bool"``, ``"int"``,
            ``"float"``, ``"vector3"``, ``"path"``, or ``"text"``.
        annotation: The raw annotation text, kept so the editor can show
            what the library actually declares when the widget hint is a
            simplification of it.
        default: The wrapper's own default, or ``None``.
        choices: Allowed values for ``kind == "choice"``.
        optional: Whether ``None`` is a legal value — an unset knob the
            engine fills in itself, which is a different state from a knob
            set to its default and worth distinguishing in a UI.
    """

    name: str
    kind: str
    annotation: str
    default: Any = None
    choices: tuple[str, ...] = ()
    optional: bool = False

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "annotation": self.annotation,
            "default": self.default,
            "optional": self.optional,
        }
        if self.choices:
            out["choices"] = list(self.choices)
        return out


@dataclass(frozen=True)
class BlockSpec:
    """One block: an engine, one of its operations, and how to wire it.

    Attributes:
        id: Stable identifier, ``"<engine>.<operation>"``. The engine name
            is the registry's, so a block id round-trips to the thing that
            will run it.
        engine: Registry name, as recorded in provenance.
        operation: The recorded operation replay dispatches on.
        label: Human-readable action for the block header.
        category: Which part of molforge drives it, from
            :func:`molforge.engine_versions`; ``"io"`` for the sources that
            aren't engines.
        inputs: Input ports, in call order.
        output: What the block produces.
        parameters: Editable constructor settings.
        available: Whether the backend is installed right now.
        detail: Why it isn't, when it isn't.
    """

    id: str
    engine: str
    operation: str
    label: str
    category: str
    inputs: tuple[BlockPort, ...]
    output: BlockPort
    parameters: tuple[BlockParameter, ...] = ()
    available: bool = True
    detail: str = ""
    version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "engine": self.engine,
            "operation": self.operation,
            "label": self.label,
            "category": self.category,
            "inputs": [p.to_dict() for p in self.inputs],
            "output": self.output.to_dict(),
            "parameters": [p.to_dict() for p in self.parameters],
            "available": self.available,
            "detail": self.detail,
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------

_LITERAL_RE = re.compile(r"Literal\[(.*)\]", re.DOTALL)
_LITERAL_MEMBER_RE = re.compile(r"""['"]([^'"]*)['"]""")


def _strip_none(annotation: str) -> tuple[str, bool]:
    """Split ``X | None`` into ``("X", True)``. Returns ``(annotation, False)``
    when nothing is optional."""
    parts = [p.strip() for p in annotation.split("|")]
    if "None" not in parts:
        return annotation.strip(), False
    remaining = [p for p in parts if p != "None"]
    return " | ".join(remaining), True


def _widget_kind(annotation: str) -> tuple[str, tuple[str, ...]]:
    """Map an annotation to a widget hint and its choices.

    Unions that aren't just ``X | None`` collapse to the widest useful
    widget rather than being rejected: ``TemplatePolicy | str`` is a text
    field, because a GUI that refused to show the knob would be worse than
    one that shows it as free text.
    """
    literal = _LITERAL_RE.search(annotation)
    if literal is not None:
        return "choice", tuple(_LITERAL_MEMBER_RE.findall(literal.group(1)))
    if annotation == "bool":
        return "bool", ()
    if annotation == "int":
        return "int", ()
    if annotation == "float":
        return "float", ()
    if annotation.replace(" ", "").startswith("tuple[float,float,float]"):
        return "vector3", ()
    if "PathLike" in annotation or "Path" in annotation:
        return "path", ()
    return "text", ()


def _parameters(factory: Callable[..., Any]) -> tuple[BlockParameter, ...]:
    """Read a constructor's keyword arguments as editable parameters.

    Annotations are read as text, never resolved: resolving ESMFold's would
    import torch, and the catalog has to be cheap enough to build on every
    page load. Positional parameters are skipped — on a class they are
    ``self``, and on the plain functions (``fetch``, the pocket detectors)
    they are the block's input port, handled separately.
    """
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):  # pragma: no cover - C callables
        return ()

    out: list[BlockParameter] = []
    for name, param in signature.parameters.items():
        if name == "self" or param.kind is not inspect.Parameter.KEYWORD_ONLY:
            continue
        if name.startswith("_"):  # **_kwargs sinks and private escape hatches
            continue
        annotation = "" if param.annotation is inspect.Parameter.empty else str(param.annotation)
        bare, optional = _strip_none(annotation)
        kind, choices = _widget_kind(bare)
        default = None if param.default is inspect.Parameter.empty else param.default
        if isinstance(default, tuple):  # box sizes: JSON has no tuple
            default = list(default)
        out.append(
            BlockParameter(
                name=name,
                kind=kind,
                annotation=annotation,
                default=default,
                choices=choices,
                optional=optional,
            )
        )
    return tuple(out)


def _ports(method: Callable[..., Any]) -> tuple[tuple[BlockPort, ...], BlockPort]:
    """Derive a method's input ports and output port.

    An operation whose return annotation isn't a known result type gets an
    ``"unknown"`` output rather than being dropped: a block that appears
    with a port the editor won't wire is a visible, reportable gap, where a
    block that silently vanishes from the palette is not.
    """
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):  # pragma: no cover - C callables
        return (), BlockPort("output", "unknown")

    inputs: list[BlockPort] = []
    for name, param in signature.parameters.items():
        if name == "self" or param.kind is inspect.Parameter.KEYWORD_ONLY:
            continue
        if name.startswith("_") or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        port_type = _INPUT_PORTS.get(name)
        if port_type is None:
            continue
        inputs.append(BlockPort(name, port_type, required=param.default is inspect.Parameter.empty))

    returns = signature.return_annotation
    annotation = "" if returns is inspect.Signature.empty else str(returns)
    return tuple(inputs), BlockPort("output", _RETURN_PORTS.get(annotation, "unknown"))


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


@dataclass
class _Environment:
    """Installed-backend lookup, built once per catalog build."""

    categories: dict[str, str] = field(default_factory=dict)
    available: dict[str, bool] = field(default_factory=dict)
    versions: dict[str, str] = field(default_factory=dict)
    details: dict[str, str] = field(default_factory=dict)

    @classmethod
    def probe(cls) -> _Environment:
        """Sweep the environment without spawning anything.

        ``probe_executables=False`` keeps the catalog free of subprocesses:
        a dashboard rebuilds this on every page load, and running
        ``gnina --version`` each time to populate a palette would be a poor
        trade for a version string the block doesn't need.
        """
        from molforge import engine_versions

        env = cls()
        for name, backend in engine_versions(probe_executables=False).items():
            env.categories[name] = backend.category
            env.available[name] = backend.available
            env.versions[name] = backend.version
            env.details[name] = backend.detail
        return env


def _registered_engines() -> dict[str, Callable[..., Any]]:
    """Every engine replay can resolve, by registry name."""
    from molforge import plugins
    from molforge.reproducibility import _register_builtin_engines

    _register_builtin_engines()
    return {name: plugins.get("engine", name) for name in plugins.available("engine")}


def _operations_for(factory: Callable[..., Any]) -> Iterable[tuple[_Operation, Any]]:
    """Yield each catalogued operation ``factory`` actually implements.

    A registered entry is either an engine class or a plain function
    (``fetch``, the pocket detectors). For a class, an operation applies
    when the class itself defines the method — inherited ``predict_samples``
    would otherwise claim every folding engine can return an ensemble, when
    only Chai-1 does. For a function, the one operation that names it
    applies, and the callable *is* the method.
    """
    if inspect.isclass(factory):
        for op in _OPERATIONS:
            if op.method in vars(factory):
                yield op, getattr(factory, op.method)
        return
    for op in _OPERATIONS:
        if op.method == "__call__" and _function_operation(factory) == op.operation:
            yield op, factory


def _function_operation(factory: Callable[..., Any]) -> str:
    """The operation a registered plain function performs.

    Read from the function's own name rather than a table keyed by registry
    name, because the registry name is the *engine* (``fpocket``,
    ``p2rank``) while the operation is what it does, and both detectors do
    the same thing.
    """
    name = getattr(factory, "__name__", "")
    if name.startswith("detect_pockets"):
        return "detect_pockets"
    if name == "fetch":
        return "fetch"
    return ""


def build_catalog(*, include_unavailable: bool = True) -> tuple[BlockSpec, ...]:
    """Build the block catalog from the engine registry.

    Args:
        include_unavailable: Keep blocks whose backend isn't installed,
            flagged ``available=False``. ``False`` drops them, for a
            palette that should only offer what can run here.

    Returns:
        One :class:`BlockSpec` per engine-and-operation pair, ordered by
        operation (roughly pipeline order) then engine name, so the palette
        groups by what a block *does* rather than by which library it came
        from.
    """
    env = _Environment.probe()
    order = {op.operation: i for i, op in enumerate(_OPERATIONS)}

    blocks: list[BlockSpec] = []
    for engine, factory in _registered_engines().items():
        for op, method in _operations_for(factory):
            inputs, output = _ports(method)
            available = env.available.get(engine, True)
            if not available and not include_unavailable:
                continue
            blocks.append(
                BlockSpec(
                    id=f"{engine}.{op.operation}",
                    engine=engine,
                    operation=op.operation,
                    label=op.label,
                    category=env.categories.get(engine, "io"),
                    inputs=inputs,
                    output=output,
                    parameters=_parameters(factory),
                    available=available,
                    detail="" if available else env.details.get(engine, ""),
                    version=env.versions.get(engine, ""),
                )
            )

    blocks.sort(key=lambda b: (order.get(b.operation, len(order)), b.engine))
    return tuple(blocks)


def catalog_to_dict(blocks: Iterable[BlockSpec] | None = None) -> dict[str, Any]:
    """The catalog as the JSON payload the editor consumes.

    ``port_types`` ships alongside the blocks so the frontend validates
    wires against the same vocabulary the backend derived them from,
    instead of keeping its own copy to fall out of step.
    """
    specs = tuple(build_catalog()) if blocks is None else tuple(blocks)
    return {
        "blocks": [b.to_dict() for b in specs],
        "port_types": sorted(
            {b.output.type for b in specs} | {p.type for b in specs for p in b.inputs}
        ),
    }
