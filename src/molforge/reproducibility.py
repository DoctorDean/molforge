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
      engines_detected: {fpocket: "4.1"}   # only when a step recorded none
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

Engines that shell out to a native binary (fpocket, GROMACS, gnina) have no
version to record when they run, so the environment block falls back to
:func:`molforge.engine_versions` — the registry of every backend molforge
can drive and what it finds installed. It is reported under its own
``engines_detected`` key because it is the weaker claim: what is installed
when the manifest is written, which is not necessarily what ran. Call
:func:`engine_versions` directly to capture the whole environment up front,
before a long run, which is the sturdier habit.

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

Aggregates
----------

A :class:`PipelineManifest` describes one output. Plenty of results aren't
one output: a consensus structure stands on several folds, a ranking on a
screen of thousands. Emitting one manifest per contributing prediction
answers the question in a form nobody can read, and restates the shared
upstream work — the target preparation, the MSA — once per prediction.

:func:`aggregate_manifest` folds them into one
:class:`AggregateManifest`. Every distinct computation appears once, keyed
by :meth:`~molforge.core.provenance.Provenance.content_id`, carrying a
count of how many outputs descend from it::

    from molforge.reproducibility import aggregate_manifest

    manifest = aggregate_manifest(ensemble.members)
    print(manifest.describe())
    # aggregate (5 outputs, 6 unique steps from 10, 4 deduplicated) - molforge 0.8.0
    #   shared:
    #     MMseqs2 -> 5 outputs  [6623ea6703b3]

Provenance itself stays linear — one parent pointer, so
:meth:`~molforge.core.provenance.Provenance.chain` keeps meaning what it
says and cache keys keep their shape. The many-to-one structure lives in
the aggregate instead, which is where it belongs: it is a property of the
*question being asked of a set of outputs*, not of any one of them.

:meth:`PipelineManifest.aggregate` does the same from manifests rather
than live objects, so a directory of ``pipeline.yaml`` files off a cluster
run folds together without the outputs still being in memory — which is
the usual situation by the time anyone asks how a screen was produced.
Both paths produce identical step ids, so the two are interchangeable.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from molforge.core import metadata_keys as mk
from molforge.core.provenance import Provenance
from molforge.versions import BackendVersion, engine_versions

if TYPE_CHECKING:
    import os
    from collections.abc import Callable, Iterable, Iterator, Sequence

__all__ = [
    "AggregateManifest",
    "AggregateStep",
    "BackendVersion",
    "PipelineManifest",
    "PipelineStep",
    "ReplayError",
    "aggregate_manifest",
    "emit_aggregate",
    "emit_pipeline",
    "engine_versions",
    "load_aggregate",
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
        environment: molforge / Python / platform versions, an
            ``engines`` map of the engine versions that ran, and — when
            a step's wrapper recorded none — an ``engines_detected`` map
            of what :func:`molforge.engine_versions` finds installed now.
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

    @classmethod
    def aggregate(cls, manifests: Iterable[PipelineManifest]) -> AggregateManifest:
        """Fold several single-output manifests into one aggregate.

        The counterpart to building an aggregate from live objects with
        :func:`aggregate_manifest`, for when the outputs are long gone and
        all you have is the manifests — a directory of ``pipeline.yaml``
        files off a cluster run, say, which is the usual situation by the
        time someone asks how a screen was produced.

        A manifest records no parent pointers, but its steps are ordered
        oldest-first, which says the same thing; the chain is rebuilt from
        that ordering, so a manifest read off disk deduplicates against a
        live object on exactly the same terms.

        Args:
            manifests: The manifests to fold together.

        Returns:
            An :class:`AggregateManifest` in which every distinct step
            appears once.

        Raises:
            ValueError: If ``manifests`` is empty, or any of them has no
                steps.

        Example:
            >>> from pathlib import Path
            >>> from molforge.reproducibility import PipelineManifest, load_pipeline
            >>> runs = [load_pipeline(p) for p in Path("runs").glob("*.yaml")]  # doctest: +SKIP
            >>> combined = PipelineManifest.aggregate(runs)                     # doctest: +SKIP
        """
        materialised = list(manifests)
        if not materialised:
            raise ValueError(
                "PipelineManifest.aggregate() needs at least one manifest; an "
                "aggregate of nothing has no provenance to describe."
            )
        chains = [_provenance_from_steps(m.steps).chain() for m in materialised]
        return _assemble_aggregate(chains, [dict(m.output) for m in materialised])


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


# ======================================================================
# Aggregate manifests
# ======================================================================

#: On-disk schema version for an aggregate manifest, emitted as the
#: ``molforge_aggregate`` key. Independent of :data:`SCHEMA_VERSION`; the
#: two shapes evolve separately.
AGGREGATE_SCHEMA_VERSION = 1

#: Step ids in an aggregate manifest are truncated content ids. Full
#: 64-char digests are correct but make a manifest with a thousand steps
#: unreadable, so the shortest of these lengths that keeps every id
#: distinct is used.
_ID_LENGTHS = (12, 16, 64)


@dataclass(frozen=True)
class AggregateStep:
    """One *distinct* computation in an :class:`AggregateManifest`.

    A step that a thousand predictions shared appears here once, with
    :attr:`contributes_to` recording that it fed a thousand of them.

    There is no single timestamp, because a deduplicated step may have
    run many times; :attr:`first_run` keeps the earliest seen, which is
    what dates the work.

    Attributes:
        id: Short, manifest-unique identifier — a truncated
            :meth:`~molforge.core.provenance.Provenance.content_id`.
        engine: Producer name.
        operation: The method that produced the output; ``""`` when
            unrecorded.
        engine_version: Producer version; ``""`` when not exposed.
        inputs: Input identifiers.
        parameters: Arguments that drove the step.
        parent: :attr:`id` of the step this one consumed, or ``None``
            for an originating step.
        contributes_to: How many of the manifest's outputs descend from
            this step. ``1`` means it was unique to one output; higher
            means it was shared, and is the whole reason to aggregate.
        first_run: Earliest ISO-8601 timestamp seen for this step.
    """

    id: str
    engine: str
    operation: str = ""
    engine_version: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    parent: str | None = None
    contributes_to: int = 1
    first_run: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain, ordered dict for serialization."""
        return {
            "id": self.id,
            "engine": self.engine,
            "operation": self.operation,
            "engine_version": self.engine_version,
            "parent": self.parent,
            "contributes_to": self.contributes_to,
            "first_run": self.first_run,
            "inputs": dict(self.inputs),
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AggregateStep:
        """Rebuild from :meth:`to_dict` output; tolerant of missing keys."""
        if "id" not in data or "engine" not in data:
            raise ValueError("aggregate step missing required 'id'/'engine' key")
        parent = data.get("parent")
        return cls(
            id=str(data["id"]),
            engine=str(data["engine"]),
            operation=str(data.get("operation", "")),
            engine_version=str(data.get("engine_version", "")),
            inputs=dict(data.get("inputs") or {}),
            parameters=dict(data.get("parameters") or {}),
            parent=str(parent) if parent else None,
            contributes_to=int(data.get("contributes_to", 1)),
            first_run=str(data.get("first_run", "")),
        )


@dataclass(frozen=True)
class AggregateManifest:
    """A citable description of the work behind *many* outputs at once.

    :class:`PipelineManifest` answers "how was this one thing made".
    Plenty of results aren't one thing: a consensus structure stands on
    five folds, a ranking on a screen of thousands. Emitting one manifest
    per prediction answers the question in a form nobody can read, and
    restates the shared upstream work — the target preparation, the MSA —
    once per prediction.

    This aggregates them. Every distinct computation appears once, keyed
    by :meth:`~molforge.core.provenance.Provenance.content_id`, with a
    count of how many outputs descend from it. A thousand predictions off
    one MSA yield one MSA step marked ``contributes_to: 1000``, and the
    manifest is a page rather than a directory.

    Attributes:
        environment: molforge / Python / platform versions and engine
            versions, consolidated across every contributing output.
        steps: The distinct steps, ordered so a step's :attr:`parent`
            always precedes it.
        outputs: One descriptor per aggregated output — its type, name
            where it has one, and the ``step`` id it terminates at.
        generated: ISO-8601 UTC time the manifest was emitted.
        schema_version: The on-disk schema version.
    """

    environment: dict[str, Any]
    steps: list[AggregateStep]
    outputs: list[dict[str, Any]]
    generated: str = ""
    schema_version: int = AGGREGATE_SCHEMA_VERSION

    def __len__(self) -> int:
        """The number of distinct steps."""
        return len(self.steps)

    def __iter__(self) -> Iterator[AggregateStep]:
        return iter(self.steps)

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

    @property
    def n_outputs(self) -> int:
        """How many outputs this manifest describes."""
        return len(self.outputs)

    @property
    def total_steps(self) -> int:
        """Chain positions before deduplication.

        The sum of every output's chain length — what a directory of
        per-output manifests would have spelled out.
        """
        return sum(step.contributes_to for step in self.steps)

    @property
    def shared_steps(self) -> list[AggregateStep]:
        """Steps more than one output descends from, most-shared first."""
        shared = [s for s in self.steps if s.contributes_to > 1]
        return sorted(shared, key=lambda s: (-s.contributes_to, s.id))

    def roots(self) -> list[AggregateStep]:
        """The originating steps — those consuming nothing in this manifest."""
        return [s for s in self.steps if s.parent is None]

    def step_by_id(self, step_id: str) -> AggregateStep:
        """Look a step up by :attr:`AggregateStep.id`.

        Raises:
            KeyError: If no step has that id.
        """
        for step in self.steps:
            if step.id == step_id:
                return step
        raise KeyError(f"no step with id {step_id!r} in this manifest")

    def summary(self) -> dict[str, int]:
        """The headline counts, as emitted under the ``summary`` key."""
        return {
            "outputs": self.n_outputs,
            "unique_steps": len(self.steps),
            "total_steps": self.total_steps,
            "shared_steps": len(self.shared_steps),
        }

    def describe(self) -> str:
        """A compact human-readable summary.

        Leads with the counts, then the shared steps — the ones worth
        looking at, since a step only many outputs depend on is the one
        whose settings matter most.
        """
        counts = self.summary()
        saved = counts["total_steps"] - counts["unique_steps"]
        lines = [
            f"aggregate ({counts['outputs']} output"
            f"{'s' if counts['outputs'] != 1 else ''}, "
            f"{counts['unique_steps']} unique step"
            f"{'s' if counts['unique_steps'] != 1 else ''} "
            f"from {counts['total_steps']}, {saved} deduplicated) — "
            f"molforge {self.environment.get('molforge_version', '?')}"
        ]
        shared = self.shared_steps
        if shared:
            lines.append("  shared:")
            for step in shared:
                ver = f" v{step.engine_version}" if step.engine_version else ""
                lines.append(
                    f"    {step.engine}{ver} -> {step.contributes_to} outputs  [{step.id}]"
                )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert to the on-disk dict shape (ordered, JSON/YAML-native)."""
        return {
            "molforge_aggregate": self.schema_version,
            "generated": self.generated,
            "environment": dict(self.environment),
            "summary": self.summary(),
            "outputs": [dict(o) for o in self.outputs],
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AggregateManifest:
        """Rebuild from :meth:`to_dict` output; tolerant of missing keys.

        The ``summary`` block is not read back — it is derived from the
        steps, so trusting a stored copy would let a hand-edited file
        report counts its own steps contradict.
        """
        return cls(
            environment=dict(data.get("environment") or {}),
            steps=[AggregateStep.from_dict(s) for s in data.get("steps", [])],
            outputs=[dict(o) for o in data.get("outputs", [])],
            generated=str(data.get("generated", "")),
            schema_version=int(data.get("molforge_aggregate", AGGREGATE_SCHEMA_VERSION)),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize to JSON text. No third-party dependency."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> AggregateManifest:
        """Deserialize from JSON text."""
        return cls.from_dict(json.loads(text))

    def to_yaml(self) -> str:
        """Serialize to YAML text. Requires the ``repro`` extra (PyYAML)."""
        yaml = _load_yaml()
        return str(yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False))

    @classmethod
    def from_yaml(cls, text: str) -> AggregateManifest:
        """Deserialize from YAML text. Requires the ``repro`` extra."""
        yaml = _load_yaml()
        return cls.from_dict(yaml.safe_load(text))


def aggregate_manifest(objs: Iterable[Provenance | object]) -> AggregateManifest:
    """Build an :class:`AggregateManifest` from many outputs.

    Args:
        objs: Outputs carrying provenance (:class:`~molforge.core.Protein`,
            ``DockingResult``, ``Pose``, ``DesignedSequence``, ...) or
            :class:`~molforge.core.provenance.Provenance` instances. Any
            mix is fine.

    Returns:
        A manifest in which every distinct computation appears once.

    Raises:
        ValueError: If ``objs`` is empty, or if any element carries no
            provenance.

    Example:
        >>> from molforge.reproducibility import aggregate_manifest
        >>> manifest = aggregate_manifest(ensemble.members)   # doctest: +SKIP
        >>> print(manifest.describe())                        # doctest: +SKIP
        aggregate (5 outputs, 6 unique steps from 10, 4 deduplicated) — molforge 0.8.0
          shared:
            ESMFold v1.0.3 -> 5 outputs  [a1b2c3d4e5f6]
    """
    materialised = list(objs)
    if not materialised:
        raise ValueError(
            "aggregate_manifest() needs at least one output; an aggregate of "
            "nothing has no provenance to describe."
        )
    chains = [_extract_provenance(obj).chain() for obj in materialised]
    descriptors = [_describe_output(obj) for obj in materialised]
    return _assemble_aggregate(chains, descriptors)


def emit_aggregate(
    objs: Iterable[Provenance | object],
    path: str | os.PathLike[str],
    *,
    fmt: str = "yaml",
) -> AggregateManifest:
    """Write an ``aggregate.yaml`` (or ``.json``) spanning many outputs.

    The many-output counterpart of :func:`emit_pipeline`.

    Args:
        objs: Outputs carrying provenance.
        path: Destination file path.
        fmt: ``"yaml"`` (default; needs the ``repro`` extra) or ``"json"``.

    Returns:
        The :class:`AggregateManifest` that was written.

    Raises:
        ValueError: If ``objs`` is empty, provenance is missing, or ``fmt``
            is unrecognized.
        ImportError: If ``fmt="yaml"`` and PyYAML isn't installed.
    """
    manifest = aggregate_manifest(objs)
    if fmt == "yaml":
        text = manifest.to_yaml()
    elif fmt == "json":
        text = manifest.to_json()
    else:
        raise ValueError(f"unknown fmt {fmt!r}; expected 'yaml' or 'json'.")
    Path(path).write_text(text, encoding="utf-8")
    return manifest


def load_aggregate(path: str | os.PathLike[str]) -> AggregateManifest:
    """Load an aggregate manifest from a ``.yaml`` / ``.json`` file.

    Format is chosen by suffix, as in :func:`load_pipeline`.

    Raises:
        ImportError: If a YAML file is loaded without PyYAML installed.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".json":
        return AggregateManifest.from_json(text)
    return AggregateManifest.from_yaml(text)


# ---------- aggregate internals ----------


def _assemble_aggregate(
    chains: list[list[Provenance]], descriptors: list[dict[str, Any]]
) -> AggregateManifest:
    """Deduplicate a set of provenance chains into one manifest.

    Each chain arrives oldest-first. A node's
    :meth:`~molforge.core.provenance.Provenance.content_id` covers its
    whole ancestry, so equal ids really are the same work reached the same
    way — two chains that diverge at step 3 share ids for 1 and 2 and
    differ from 3 on, which is exactly the merge we want.

    Walking oldest-first and recording each id the first time it appears
    also gives the ordering for free: a parent is always seen before its
    child, so :attr:`AggregateManifest.steps` needs no topological sort.
    """
    full_ids = [[node.content_id() for node in chain] for chain in chains]
    shorten = _id_shortener([i for ids in full_ids for i in ids])

    accumulated: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for chain, ids in zip(chains, full_ids, strict=True):
        parent: str | None = None
        for node, full_id in zip(chain, ids, strict=True):
            step_id = shorten(full_id)
            existing = accumulated.get(step_id)
            if existing is None:
                accumulated[step_id] = {
                    "id": step_id,
                    "engine": node.engine,
                    "operation": node.operation,
                    "engine_version": node.engine_version,
                    "inputs": dict(node.inputs),
                    "parameters": dict(node.parameters),
                    "parent": parent,
                    "contributes_to": 1,
                    "first_run": node.timestamp,
                }
                order.append(step_id)
            else:
                existing["contributes_to"] += 1
                # Earliest timestamp dates the work. An empty one sorts
                # before any real timestamp, so guard against it.
                if node.timestamp and (
                    not existing["first_run"] or node.timestamp < existing["first_run"]
                ):
                    existing["first_run"] = node.timestamp
            parent = step_id

    steps = [AggregateStep(**accumulated[step_id]) for step_id in order]
    return AggregateManifest(
        environment=_merge_environments(chains),
        steps=steps,
        outputs=[
            {**descriptor, "step": shorten(ids[-1]) if ids else None}
            for descriptor, ids in zip(descriptors, full_ids, strict=True)
        ],
        generated=_generated_timestamp(),
    )


def _id_shortener(full_ids: Iterable[str]) -> Callable[[str], str]:
    """Return a function truncating content ids to a manifest-wide length.

    Full digests are unreadable in bulk, so the shortest length that keeps
    every id in *this* manifest distinct is used. The check is what makes
    truncation safe: a prefix collision would silently merge two unrelated
    steps, which is a worse manifest than a verbose one.
    """
    distinct = set(full_ids)
    for length in _ID_LENGTHS:
        if len({i[:length] for i in distinct}) == len(distinct):
            return lambda full_id, _length=length: str(full_id[:_length])  # type: ignore[misc]
    # Unreachable: the last entry is the full digest length.
    return lambda full_id: full_id


def _merge_environments(chains: list[list[Provenance]]) -> dict[str, Any]:
    """Consolidate one environment block across every contributing chain.

    molforge / Python / platform come from the first chain's terminal
    step; engine versions are unioned, because an aggregate legitimately
    spans engines no single output touched. A version that two outputs
    disagree about is recorded as the set of values seen rather than
    silently resolved — disagreement is a real finding about the run, not
    a formatting problem.
    """
    if not chains:  # pragma: no cover - callers guarantee non-empty
        return {}
    base = _capture_environment(chains[0][-1])
    engines: dict[str, set[str]] = {}
    for chain in chains:
        for node in chain:
            if node.engine_version:
                engines.setdefault(node.engine, set()).add(node.engine_version)
    base["engines"] = {
        engine: (next(iter(versions)) if len(versions) == 1 else sorted(versions))
        for engine, versions in sorted(engines.items())
    }
    return base


def _provenance_from_steps(steps: Sequence[PipelineStep]) -> Provenance:
    """Rebuild a provenance chain from a linearized manifest's steps.

    A :class:`PipelineManifest` records no parent pointers, but it is
    ordered oldest-first, which says the same thing: step *i* consumed
    step *i-1*. Rebuilding the chain as :class:`Provenance` means
    aggregation has exactly one code path, and manifests read back off
    disk deduplicate against live objects on the same terms.

    Raises:
        ValueError: If ``steps`` is empty.
    """
    if not steps:
        raise ValueError("cannot rebuild a provenance chain from a manifest with no steps")
    node: Provenance | None = None
    for step in steps:
        node = Provenance(
            engine=step.engine,
            operation=step.operation,
            engine_version=step.engine_version,
            timestamp=step.timestamp,
            parameters=dict(step.parameters),
            inputs=dict(step.inputs),
            parent=node,
        )
    assert node is not None  # non-empty steps guarantee this
    return node


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

    Steps whose wrapper recorded no version — every wrapper that shells
    out to a native binary — get a second pass against
    :func:`molforge.engine_versions`, reported under a separate
    ``engines_detected`` key. Separate because it is a weaker claim:
    ``engines`` is what ran, ``engines_detected`` is what is installed
    *now*, and they disagree if the tool was upgraded since.
    """
    import platform as _platform

    engines: dict[str, str] = {}
    unrecorded: list[str] = []
    for step in provenance.chain():
        if step.engine_version:
            engines[step.engine] = step.engine_version
        elif step.engine not in unrecorded:
            unrecorded.append(step.engine)

    environment: dict[str, Any] = {
        "molforge_version": provenance.molforge_version or _molforge_version(),
        "python_version": _platform.python_version(),
        "platform": _platform.platform(),
        "engines": engines,
    }
    detected = _detect_versions(name for name in unrecorded if name not in engines)
    if detected:
        environment["engines_detected"] = detected
    return environment


def _detect_versions(engine_names: Iterable[str]) -> dict[str, str]:
    """Ask the backend registry about engines the chain left blank.

    Returns only the ones it can actually answer for, so the manifest
    gains numbers where they exist and stays silent where they don't.
    """
    registry = engine_versions()
    found: dict[str, str] = {}
    for name in engine_names:
        # Multi-step wrappers namespace their operations ("GROMACS.minimize").
        backend = registry.get(name) or registry.get(name.split(".", 1)[0])
        if backend is None or not backend.version:
            continue
        # "runtime" rows are shared libraries, not engines; molforge's own
        # version is already reported as molforge_version, and matching
        # "molforge.io.fetch" against it would just restate that.
        if backend.category == "runtime":
            continue
        found[name] = backend.version
    return found


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
