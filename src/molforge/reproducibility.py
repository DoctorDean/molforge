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

Scope (v1): **emit and inspect**. Replaying a manifest — re-executing the
steps — is deliberately out of scope: provenance records the engine and
its parameters but not the *operation* (predict vs dock vs generate) or
resolvable input objects, so replay needs a provenance-schema extension and
an engine registry. That's the documented next step, not part of this
module. A manifest is also single-output and linear (provenance has one
parent pointer); merging several outputs' chains is a future extension.
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
    from collections.abc import Iterator

__all__ = [
    "PipelineManifest",
    "PipelineStep",
    "emit_pipeline",
    "load_pipeline",
    "pipeline_manifest",
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
        engine_version: Producer version, ``""`` when not exposed.
        timestamp: ISO-8601 UTC time the step ran.
        inputs: Input identifiers (sequence, paths, hashes).
        parameters: Engine arguments that drove the step.
    """

    step: int
    engine: str
    engine_version: str = ""
    timestamp: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain, ordered dict for serialization."""
        return {
            "step": self.step,
            "engine": self.engine,
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
