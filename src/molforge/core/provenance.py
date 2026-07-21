"""First-class provenance tracking for molforge outputs.

A molforge user running ``fold -> dock -> md`` ends up with a
:class:`Trajectory` that depends on a docking pose that depends on a
folded structure that depends on a sequence. Reconstructing that
chain — *what engine produced this with what arguments from what
inputs* — is the job of the :class:`Provenance` dataclass introduced
here.

Why this exists
---------------

Each engine wrapper already writes some of this information into
``protein.metadata`` in an ad-hoc way: ESMFold sets
``metadata["engine"] = "ESMFold"``, RFdiffusion sets
``metadata["source_args"]`` to a string. The keys are scattered, the
shapes disagree across engines, and there is no concept of a *parent*
output, so chains of operations are not traceable. This module
canonicalises the shape and gives it a parent pointer so a single
:class:`Provenance` walks back through the whole chain of operations
that produced an output.

Design
------

- **In-memory only (for now).** The provenance object lives at
  ``protein.metadata["provenance"]``. molforge's existing PDB / mmCIF
  writers preserve only six known metadata keys (see the mmCIF audit
  in ``c3a012e``); ``"provenance"`` is not one of them. Persisting
  provenance to disk needs a sidecar format and is deliberately not
  in scope for this module — the helpers below give explicit
  ``to_dict`` / ``from_dict`` conversions so callers can serialise
  manually.

- **Frozen dataclass.** Once attached to an output, mutating provenance
  would corrupt the audit trail. The dataclass is frozen; updates
  produce a new instance via :meth:`replace`.

- **Composable parent.** ``parent`` is itself a :class:`Provenance`
  (or ``None``). Walking the chain reconstructs the whole history;
  see :meth:`chain` for the convenience traversal.

- **Stable JSON shape.** ``to_dict`` emits a plain-Python ``dict`` with
  JSON-serialisable values only; ``from_dict`` is the inverse. This
  is the contract for any downstream code that wants to persist or
  transmit provenance.

Concretely
----------

A typical wrapper attaches provenance like this::

    from molforge.core.provenance import Provenance
    from molforge.core import metadata_keys as mk

    prov = Provenance.from_engine(
        engine="Vina",
        engine_version="1.2.3",
        parameters={"exhaustiveness": 8, "n_poses": 10},
        inputs={"receptor": str(receptor_path),
                "ligand":   str(ligand_path)},
        parent=receptor_protein.metadata.get(mk.PROVENANCE),
    )
    result.metadata[mk.PROVENANCE] = prov

A user inspecting a downstream output traces the chain::

    prov = result.metadata[mk.PROVENANCE]
    for step in prov.chain():  # oldest first
        print(step.engine, step.engine_version, step.timestamp)

Wrapper adoption
----------------

This module *provides* provenance; wrapper adoption (each engine
calling :meth:`Provenance.from_engine` instead of writing ad-hoc keys)
is intentionally a separate commit. The existing
``metadata["engine"]`` convention continues to work; adoption adds the
canonical key alongside it without removing the existing one.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator


__all__ = ["Provenance"]


def _utc_timestamp() -> str:
    """ISO-8601 timestamp in UTC, second precision.

    Kept as a free function so tests can monkeypatch it for
    deterministic output. Second precision is sufficient for
    audit-trail purposes; nanosecond-precision timestamps would just
    make Provenance objects spuriously different between rapid
    repeated calls.
    """
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()


def _molforge_version() -> str:
    """Return the running molforge version, or ``"unknown"`` if it
    cannot be read.

    Lazily imported to avoid circular imports during package init.
    """
    try:
        import molforge

        return str(getattr(molforge, "__version__", "unknown"))
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class Provenance:
    """A record of *what produced this output*.

    Attributes:
        engine: Name of the producer — typically an engine name
            (``"ESMFold"``, ``"Vina"``) or a molforge function path
            (``"molforge.prep.prepare_for_md"``).
        operation: The engine method that produced the output
            (``"predict"``, ``"dock"``, ``"generate"``, ...). ``""`` when
            unrecorded. Consumed by :func:`molforge.reproducibility.replay`
            to know which method to re-invoke.
        engine_version: Version string of the engine itself. ``""``
            when the engine doesn't expose one. Distinct from
            :attr:`molforge_version`.
        molforge_version: The molforge version that ran this step.
            Auto-filled by :meth:`from_engine` from ``molforge.__version__``.
        timestamp: ISO-8601 UTC timestamp of when the step ran.
            Auto-filled by :meth:`from_engine`.
        parameters: Engine-specific arguments that drove this step
            (``{"exhaustiveness": 8, "pH": 7.4, ...}``). Values must
            be JSON-serialisable.
        inputs: Identifiers for the input data. For folding:
            ``{"sequence": "..."}`` or ``{"sequence_hash": "..."}``.
            For docking: ``{"receptor": <path>, "ligand": <path>}``.
            For MD: ``{"system": <path>}``. Values must be
            JSON-serialisable.
        parent: The provenance of the input that this step *consumed*,
            forming a chain back to the original input. ``None`` for
            terminal steps (e.g. reading a PDB from disk).

    The dataclass is frozen — once attached to an output, the audit
    trail can't be mutated. Use :meth:`replace` for derived copies.
    """

    engine: str
    operation: str = ""
    engine_version: str = ""
    molforge_version: str = ""
    timestamp: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)
    parent: Provenance | None = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_engine(
        cls,
        engine: str,
        *,
        operation: str = "",
        engine_version: str = "",
        parameters: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        parent: Provenance | None = None,
    ) -> Provenance:
        """Build a :class:`Provenance` with auto-filled metadata.

        Auto-fills :attr:`molforge_version` from ``molforge.__version__``
        and :attr:`timestamp` from the current UTC time. Wrappers should
        prefer this over the bare constructor so those two fields are
        consistent across the package.

        Args:
            engine: Producer name (engine name or function path).
            operation: The engine method that produced the output
                (``"predict"`` / ``"dock"`` / ...). Enables replay.
            engine_version: Producer version. Pass ``""`` if the engine
                doesn't expose one.
            parameters: Engine-specific arguments. Must be
                JSON-serialisable; copied into a fresh dict so the
                caller's mutations don't affect the stored value.
            inputs: Input identifiers. Same JSON constraint.
            parent: Provenance of the input this step consumed.

        Returns:
            A frozen :class:`Provenance`. Validate JSON-serialisability
            of ``parameters`` and ``inputs`` eagerly so a malformed
            value fails here rather than at later serialisation time.
        """
        # Defensive copies — the caller may keep mutating their dict.
        params = dict(parameters) if parameters is not None else {}
        ins = dict(inputs) if inputs is not None else {}

        # Eager JSON-serialisability check. Without this, a wrapper
        # could attach a Path or a NumPy array; to_dict would crash
        # much later, far from the wrapper that caused it. We
        # intentionally use strict json.dumps (no ``default=``) so the
        # contract is "values must be JSON-native" — callers convert
        # paths to strings, NumPy arrays to lists, etc., themselves.
        try:
            json.dumps(params)
            json.dumps(ins)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Provenance parameters/inputs must be JSON-serialisable: {e}") from e

        return cls(
            engine=engine,
            operation=operation,
            engine_version=engine_version,
            molforge_version=_molforge_version(),
            timestamp=_utc_timestamp(),
            parameters=params,
            inputs=ins,
            parent=parent,
        )

    def replace(self, **changes: Any) -> Provenance:
        """Return a copy with the given fields replaced.

        Frozen dataclasses don't support attribute assignment;
        ``replace`` is the supported way to derive a modified copy.
        Useful for tools that need to amend a provenance entry
        (e.g. injecting a parent that wasn't known at construction
        time)::

            prov2 = prov.replace(parent=upstream_prov)
        """
        return replace(self, **changes)

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def walk(self) -> Iterator[Provenance]:
        """Yield this step then each ancestor, newest first.

        ``walk()`` is the "stack" view: the current step, then what it
        consumed, then what *that* consumed. Use :meth:`chain` for the
        oldest-first view that reads naturally as "fold -> dock -> md".
        """
        cur: Provenance | None = self
        while cur is not None:
            yield cur
            cur = cur.parent

    def chain(self) -> list[Provenance]:
        """Return the provenance chain oldest-first.

        The first element is the *originating* step (the deepest
        ancestor with ``parent=None``); the last element is ``self``.
        Suitable for printing as a left-to-right pipeline::

            for step in prov.chain():
                print(step.engine)
            # -> ESMFold
            # -> Vina
            # -> OpenMM
        """
        return list(reversed(list(self.walk())))

    @property
    def depth(self) -> int:
        """Number of steps in the chain (this step + ancestors).

        Useful for assertions ("we expected a 3-step pipeline"). A
        terminal Provenance has depth 1.
        """
        return sum(1 for _ in self.walk())

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serialisable plain dict.

        The shape is::

            {
                "engine": str,
                "operation": str,
                "engine_version": str,
                "molforge_version": str,
                "timestamp": str,        # ISO-8601 UTC
                "parameters": dict,
                "inputs": dict,
                "parent": dict | None,   # recursively
            }

        This is the on-disk format — when provenance gets serialised
        (sidecar JSON, database row, etc.) this dict is the source of
        truth. :meth:`from_dict` is the inverse.
        """
        return {
            "engine": self.engine,
            "operation": self.operation,
            "engine_version": self.engine_version,
            "molforge_version": self.molforge_version,
            "timestamp": self.timestamp,
            "parameters": dict(self.parameters),
            "inputs": dict(self.inputs),
            "parent": self.parent.to_dict() if self.parent is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Provenance:
        """Reconstruct a :class:`Provenance` from :meth:`to_dict` output.

        Tolerant of missing keys (treats them as defaults) so an older
        on-disk shape continues to load after fields are added.

        Args:
            data: A dict shaped like :meth:`to_dict`'s output.

        Returns:
            A :class:`Provenance` rebuilt from the dict. The ``parent``
            key, if present and non-``None``, is recursively rebuilt.

        Raises:
            ValueError: If ``data`` lacks the required ``engine`` key.
        """
        if "engine" not in data:
            raise ValueError("Provenance dict missing required 'engine' key")
        parent_data = data.get("parent")
        parent = cls.from_dict(parent_data) if parent_data else None
        return cls(
            engine=str(data["engine"]),
            operation=str(data.get("operation", "")),
            engine_version=str(data.get("engine_version", "")),
            molforge_version=str(data.get("molforge_version", "")),
            timestamp=str(data.get("timestamp", "")),
            parameters=dict(data.get("parameters") or {}),
            inputs=dict(data.get("inputs") or {}),
            parent=parent,
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise to JSON text. Convenience wrapper around
        :meth:`to_dict` and :func:`json.dumps`. Values are guaranteed
        JSON-native because :meth:`from_engine` validated them at
        construction time, so no ``default=`` coercion is needed."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> Provenance:
        """Deserialise from JSON text. Inverse of :meth:`to_json`."""
        return cls.from_dict(json.loads(text))
