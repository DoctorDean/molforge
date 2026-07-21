"""Emit a citable ``pipeline.yaml`` from a molforge output's provenance.

Most papers in this space don't ship reproducible code. molforge already
records *what produced an output* — every engine wrapper attaches a
:class:`~molforge.core.provenance.Provenance` (engine, version, parameters,
inputs, and a pointer to the step it consumed) to
``result.metadata["provenance"]``. This module turns that chain into a
single, human-readable manifest — the artifact a methods section can point
at:

    from molforge.reproducibility import emit_pipeline

    folded = esmfold.predict(sequence)
    docked = vina.dock(folded, ligand)
    emit_pipeline(docked, "pipeline.yaml")

The resulting file linearizes the provenance chain into ordered steps and
adds a consolidated environment block (molforge / Python / platform
versions and the engine versions that ran)::

    molforge_pipeline: 1
    generated: "2026-07-15T12:00:00+00:00"
    environment:
      molforge_version: "0.6.0"
      python_version: "3.12.13"
      platform: "macOS-14.3-arm64"
      engines: {ESMFold: "1.0.3", Vina: "1.2.5"}
    steps:
      - step: 1
        engine: ESMFold
        engine_version: "1.0.3"
        inputs: {sequence: "MKT..."}
        parameters: {num_recycles: 4}
      - step: 2
        engine: Vina
        ...
    output: {type: DockingResult}

The in-memory :class:`PipelineManifest` and its ``to_dict`` / ``to_json``
forms need no third-party dependency. Reading and writing the ``.yaml``
form needs PyYAML — an opt-in extra (``pip install "molforge[repro]"``) so
molforge's core stays numpy-only.

Replay
------

:func:`replay` re-executes a manifest's chain, threading each step's output
into the next::

    from molforge.reproducibility import load_pipeline, replay

    manifest = load_pipeline("pipeline.yaml")
    output = replay(manifest, context={"ligand": "aspirin.sdf"})

It resolves each step's engine from the registry (molforge's own wrappers
plus anything under :mod:`molforge.plugins`), reconstructs the call with a
per-*operation* **replay handler** (``molforge`` ships ``predict`` /
``dock``), and runs it. Handlers own the reconstruction, so the fragile
"which recorded input is upstream vs. a literal" wiring is contained per
operation rather than guessed globally.

Replay is inherently partial: engines must be installed, GPU steps need the
hardware (replay orchestrates, it doesn't provide compute), and inputs that
aren't literals (a docking receptor is really the previous step's output; a
ligand may be a path that no longer exists) come from the previous step or a
supplied ``context``. An unresolvable input, an unknown engine, or an
operation with no registered handler raises a clear :class:`ReplayError`.
Register a handler for a custom operation with :func:`register_replay_handler`.

A manifest is single-output and linear (provenance has one parent pointer);
merging several outputs' chains is a future extension.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from molforge.core import metadata_keys as mk
from molforge.core.provenance import Provenance

if TYPE_CHECKING:
    import os
    from collections.abc import Callable, Iterator

__all__ = [
    "PipelineManifest",
    "PipelineStep",
    "ReplayError",
    "emit_pipeline",
    "load_pipeline",
    "pipeline_manifest",
    "register_replay_handler",
    "replay",
]

#: On-disk schema version, emitted as the ``molforge_pipeline`` key. Bump
#: when the manifest shape changes incompatibly; :meth:`PipelineManifest.from_dict`
#: stays tolerant of older shapes.
SCHEMA_VERSION = 1

_YAML_INSTALL_HINT = (
    'Reading or writing pipeline YAML needs PyYAML. Install it with: pip install "molforge[repro]"'
)


def _generated_timestamp() -> str:
    """ISO-8601 UTC timestamp of manifest emission, second precision.

    A free function so tests can monkeypatch it for deterministic output,
    mirroring :func:`molforge.core.provenance._utc_timestamp`.
    """
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()


def _molforge_version() -> str:
    """The running molforge version, or ``"unknown"``."""
    try:
        import molforge

        return str(getattr(molforge, "__version__", "unknown"))
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class PipelineStep:
    """One step of a pipeline — a single provenance entry, linearized.

    Attributes:
        step: 1-indexed position in the pipeline (1 = oldest / originating).
        engine: Producer name (engine name or molforge function path).
        operation: The engine method that produced the output
            (``"predict"`` / ``"dock"`` / ...); ``""`` when unrecorded.
        engine_version: Producer version, ``""`` when not exposed.
        timestamp: ISO-8601 UTC time the step ran.
        inputs: Input identifiers (sequence, paths, hashes).
        parameters: Engine arguments that drove the step.
    """

    step: int
    engine: str
    operation: str = ""
    engine_version: str = ""
    timestamp: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain, ordered dict for serialization."""
        return {
            "step": self.step,
            "engine": self.engine,
            "operation": self.operation,
            "engine_version": self.engine_version,
            "timestamp": self.timestamp,
            "inputs": dict(self.inputs),
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineStep:
        """Rebuild from :meth:`to_dict` output; tolerant of missing keys."""
        if "engine" not in data or "step" not in data:
            raise ValueError("pipeline step missing required 'step'/'engine' key")
        return cls(
            step=int(data["step"]),
            engine=str(data["engine"]),
            operation=str(data.get("operation", "")),
            engine_version=str(data.get("engine_version", "")),
            timestamp=str(data.get("timestamp", "")),
            inputs=dict(data.get("inputs") or {}),
            parameters=dict(data.get("parameters") or {}),
        )


@dataclass(frozen=True)
class PipelineManifest:
    """A citable description of the workflow that produced an output.

    Attributes:
        environment: molforge / Python / platform versions plus an
            ``engines`` map of the engine versions that ran.
        steps: The pipeline steps, oldest-first.
        generated: ISO-8601 UTC time the manifest was emitted.
        output: A short descriptor of the terminal output
            (``{"type": ..., "name": ...}``).
        schema_version: The on-disk schema version.
    """

    environment: dict[str, Any]
    steps: list[PipelineStep]
    generated: str = ""
    output: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[PipelineStep]:
        return iter(self.steps)

    def describe(self) -> str:
        """A compact human-readable summary, one line per step."""
        lines = [
            f"pipeline ({len(self.steps)} step{'s' if len(self.steps) != 1 else ''}) — "
            f"molforge {self.environment.get('molforge_version', '?')}"
        ]
        for s in self.steps:
            ver = f" v{s.engine_version}" if s.engine_version else ""
            lines.append(f"  {s.step}. {s.engine}{ver}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to the on-disk dict shape (ordered, JSON/YAML-native)."""
        return {
            "molforge_pipeline": self.schema_version,
            "generated": self.generated,
            "environment": dict(self.environment),
            "steps": [s.to_dict() for s in self.steps],
            "output": dict(self.output),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineManifest:
        """Rebuild from :meth:`to_dict` output; tolerant of missing keys."""
        return cls(
            environment=dict(data.get("environment") or {}),
            steps=[PipelineStep.from_dict(s) for s in data.get("steps", [])],
            generated=str(data.get("generated", "")),
            output=dict(data.get("output") or {}),
            schema_version=int(data.get("molforge_pipeline", SCHEMA_VERSION)),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize to JSON text. No third-party dependency."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> PipelineManifest:
        """Deserialize from JSON text."""
        return cls.from_dict(json.loads(text))

    def to_yaml(self) -> str:
        """Serialize to YAML text. Requires the ``repro`` extra (PyYAML)."""
        yaml = _load_yaml()
        # sort_keys=False preserves our deliberate ordering (schema, env,
        # steps, output); block style keeps the artifact readable/diffable.
        return str(
            yaml.safe_dump(
                self.to_dict(),
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
            )
        )

    @classmethod
    def from_yaml(cls, text: str) -> PipelineManifest:
        """Deserialize from YAML text. Requires the ``repro`` extra (PyYAML)."""
        yaml = _load_yaml()
        return cls.from_dict(yaml.safe_load(text))


def pipeline_manifest(obj: Provenance | object) -> PipelineManifest:
    """Build a :class:`PipelineManifest` from an output or a provenance.

    Args:
        obj: A :class:`~molforge.core.provenance.Provenance`, or any molforge
            output carrying one at ``metadata["provenance"]`` (a
            :class:`~molforge.core.Protein`, ``DockingResult``, ``Pose``,
            ``DesignedSequence``, ...).

    Returns:
        A manifest with the provenance chain linearized oldest-first and the
        environment consolidated.

    Raises:
        ValueError: If no provenance can be found on ``obj``.
    """
    provenance = _extract_provenance(obj)
    chain = provenance.chain()  # oldest-first
    steps = [
        PipelineStep(
            step=i + 1,
            engine=p.engine,
            operation=p.operation,
            engine_version=p.engine_version,
            timestamp=p.timestamp,
            inputs=dict(p.inputs),
            parameters=dict(p.parameters),
        )
        for i, p in enumerate(chain)
    ]
    return PipelineManifest(
        environment=_capture_environment(provenance),
        steps=steps,
        generated=_generated_timestamp(),
        output=_describe_output(obj),
    )


def emit_pipeline(
    obj: Provenance | object,
    path: str | os.PathLike[str],
    *,
    fmt: str = "yaml",
) -> PipelineManifest:
    """Write a ``pipeline.yaml`` (or ``.json``) describing how ``obj`` was made.

    Args:
        obj: An output carrying provenance, or a
            :class:`~molforge.core.provenance.Provenance`.
        path: Destination file path.
        fmt: ``"yaml"`` (default; needs the ``repro`` extra) or ``"json"``
            (no extra).

    Returns:
        The :class:`PipelineManifest` that was written (handy for inspection).

    Raises:
        ValueError: If no provenance is found, or ``fmt`` is unrecognized.
        ImportError: If ``fmt="yaml"`` and PyYAML isn't installed.
    """
    manifest = pipeline_manifest(obj)
    if fmt == "yaml":
        text = manifest.to_yaml()
    elif fmt == "json":
        text = manifest.to_json()
    else:
        raise ValueError(f"unknown fmt {fmt!r}; expected 'yaml' or 'json'.")
    Path(path).write_text(text, encoding="utf-8")
    return manifest


def load_pipeline(path: str | os.PathLike[str]) -> PipelineManifest:
    """Load a manifest from a ``.yaml`` / ``.json`` file.

    The format is chosen by suffix: ``.json`` is parsed as JSON (no extra);
    anything else is parsed as YAML (needs the ``repro`` extra). Since YAML
    is a superset of JSON, a ``.yaml`` loader also reads JSON content.

    Raises:
        ImportError: If a YAML file is loaded without PyYAML installed.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".json":
        return PipelineManifest.from_json(text)
    return PipelineManifest.from_yaml(text)


# ---------- internals ----------


def _load_yaml() -> Any:
    """Import PyYAML or raise a clean, install-hint ImportError."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError(_YAML_INSTALL_HINT) from e
    return yaml


def _extract_provenance(obj: Provenance | object) -> Provenance:
    """Pull a :class:`Provenance` from ``obj`` (itself, or its metadata)."""
    if isinstance(obj, Provenance):
        return obj
    metadata = getattr(obj, "metadata", None)
    if isinstance(metadata, dict):
        prov = metadata.get(mk.PROVENANCE)
        if isinstance(prov, Provenance):
            return prov
    raise ValueError(
        f"no provenance found on {type(obj).__name__}; pipeline emission needs an "
        "output produced by a molforge engine (which attaches a Provenance to "
        "metadata['provenance']) or a Provenance instance directly."
    )


def _capture_environment(provenance: Provenance) -> dict[str, Any]:
    """Consolidate the environment: molforge / Python / platform + engines.

    The per-engine versions are collected across the whole chain; the
    molforge version is taken from the terminal step (falling back to the
    live version) since that's the one that assembled the final output.
    """
    import platform as _platform

    engines: dict[str, str] = {}
    for step in provenance.chain():
        if step.engine_version:
            engines[step.engine] = step.engine_version
    return {
        "molforge_version": provenance.molforge_version or _molforge_version(),
        "python_version": _platform.python_version(),
        "platform": _platform.platform(),
        "engines": engines,
    }


def _describe_output(obj: Provenance | object) -> dict[str, Any]:
    """A short descriptor of the terminal output object."""
    if isinstance(obj, Provenance):
        return {}
    out: dict[str, Any] = {"type": type(obj).__name__}
    name = getattr(obj, "name", None)
    if isinstance(name, str) and name:
        out["name"] = name
    return out


# ======================================================================
# Replay
# ======================================================================


class ReplayError(RuntimeError):
    """Raised when a manifest can't be replayed — a missing engine, an
    operation with no handler, or an input that can't be resolved."""


#: operation name -> handler. A handler reconstructs and runs one step:
#: ``handler(engine_factory, step, upstream_output, context) -> output``.
_REPLAY_HANDLERS: dict[str, Callable[..., Any]] = {}


def register_replay_handler(
    operation: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator registering a replay handler for ``operation``.

    A handler has the signature ``(engine_factory, step, upstream_output,
    context) -> output`` — it reconstructs the engine from ``step.parameters``
    and calls the right method, using ``upstream_output`` (the previous
    step's result) and ``context`` (user-supplied inputs) as needed::

        @register_replay_handler("dock")
        def _dock(factory, step, upstream, context): ...
    """

    def decorator(handler: Callable[..., Any]) -> Callable[..., Any]:
        _REPLAY_HANDLERS[operation] = handler
        return handler

    return decorator


def replay(
    source: PipelineManifest | Provenance | object, *, context: dict[str, Any] | None = None
) -> Any:
    """Re-execute a pipeline's chain, returning the terminal output.

    Args:
        source: A :class:`PipelineManifest` (e.g. from :func:`load_pipeline`),
            a :class:`~molforge.core.provenance.Provenance`, or any output
            carrying one.
        context: Objects to resolve recorded inputs that aren't literals —
            keyed by the input name (``{"ligand": "aspirin.sdf"}``). A step's
            primary input is threaded from the previous step automatically;
            ``context`` covers the rest.

    Returns:
        The output of the final step.

    Raises:
        ReplayError: If a step has no recorded operation, its engine can't be
            resolved, no handler is registered for its operation, or a
            required input can't be resolved.
    """
    manifest = source if isinstance(source, PipelineManifest) else pipeline_manifest(source)
    ctx = context or {}
    if not manifest.steps:
        raise ReplayError("manifest has no steps to replay.")

    output: Any = None
    for step in manifest.steps:  # oldest-first
        if not step.operation:
            raise ReplayError(
                f"step {step.step} ({step.engine}) has no recorded operation, so "
                "replay can't know which method to call. It predates the operation "
                "field or was produced by an engine that doesn't record one."
            )
        handler = _REPLAY_HANDLERS.get(step.operation)
        if handler is None:
            raise ReplayError(
                f"no replay handler registered for operation {step.operation!r} "
                f"(step {step.step}, engine {step.engine}). Register one with "
                "register_replay_handler()."
            )
        factory = _resolve_engine(step.engine)
        output = handler(factory, step, output, ctx)
    return output


# ---------- engine + input resolution ----------


def _resolve_engine(name: str) -> Callable[..., Any]:
    """Resolve an engine name to its factory (class), via the registry."""
    from molforge import plugins

    _register_builtin_engines()
    try:
        return plugins.get("engine", name)  # type: ignore[no-any-return]
    except KeyError as e:
        available = plugins.available("engine")
        raise ReplayError(
            f"no engine registered as {name!r}; can't replay this step. "
            f"Registered engines: {sorted(available)}. Install the engine or "
            "register it via molforge.plugins.register_engine()."
        ) from e


def _register_builtin_engines() -> None:
    """Register molforge's built-in folding / docking engines by name.

    Idempotent and cheap (importing an engine class doesn't import its heavy
    deps — those are lazy). Run on every resolve so a cleared registry
    (e.g. in tests) is repopulated.
    """
    from molforge import plugins
    from molforge.docking import DockingEngine
    from molforge.wrappers import docking, folding
    from molforge.wrappers.folding import FoldingEngine

    for module, base in ((folding, FoldingEngine), (docking, DockingEngine)):
        for attr in getattr(module, "__all__", []):
            obj = getattr(module, attr, None)
            if isinstance(obj, type) and issubclass(obj, base) and obj not in (base,):
                plugins.register_engine(obj.name, obj)


def _construct(factory: Callable[..., Any], parameters: dict[str, Any]) -> Any:
    """Instantiate ``factory`` from recorded parameters.

    Filters ``parameters`` to the constructor's accepted keyword arguments,
    so call-level parameters recorded in provenance (e.g. Boltz's
    ``affinity_binder``) don't break construction.
    """
    import inspect

    try:
        accepted = set(inspect.signature(factory).parameters)
        kwargs = {k: v for k, v in parameters.items() if k in accepted}
    except (TypeError, ValueError):
        kwargs = dict(parameters)
    return factory(**kwargs)


def _resolve_input(step: PipelineStep, key: str, context: dict[str, Any], *, what: str) -> Any:
    """Resolve an input named ``key`` for ``step`` — context first, else the
    recorded literal (a sequence string, a path). Raises if neither exists."""
    if key in context:
        return context[key]
    if key in step.inputs and step.inputs[key] is not None:
        return step.inputs[key]
    raise ReplayError(
        f"step {step.step} ({step.engine}) needs {what} but it isn't a recorded "
        f"literal — pass it via context={{{key!r}: ...}}."
    )


# ---------- built-in operation handlers ----------


@register_replay_handler("predict")
def _replay_predict(
    factory: Callable[..., Any],
    step: PipelineStep,
    upstream: Any,
    context: dict[str, Any],
) -> Any:
    """Replay a folding ``predict(sequence)`` step. (``upstream`` unused —
    a fold is a chain root.)"""
    engine = _construct(factory, step.parameters)
    sequence = _resolve_input(step, "sequence", context, what="a sequence")
    return engine.predict(sequence)


@register_replay_handler("dock")
def _replay_dock(
    factory: Callable[..., Any],
    step: PipelineStep,
    upstream: Any,
    context: dict[str, Any],
) -> Any:
    """Replay a docking ``dock(receptor, ligand)`` step.

    The receptor is the previous step's output (a folded structure) when
    there is one; otherwise it must come from ``context``. The ligand comes
    from ``context`` or the recorded literal.
    """
    engine = _construct(factory, step.parameters)
    receptor = (
        upstream
        if upstream is not None
        else _resolve_input(step, "receptor", context, what="a receptor")
    )
    ligand = _resolve_input(step, "ligand", context, what="a ligand")
    return engine.dock(receptor, ligand)
