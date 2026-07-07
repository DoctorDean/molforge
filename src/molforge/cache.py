"""Result caching for molforge engines.

Engines (folding, docking, generative) take minutes-to-hours per
call. Recomputing identical inputs is wasteful, and molforge's
:class:`Provenance` makes it trivial to detect when a recomputation
is identical: hash the engine + parameters + inputs + parent chain,
look up the result.

Integration on the engine side is a few lines per method::

    provenance = self._build_provenance(...)
    cache = get_default_cache()
    cached = cache.get(provenance, "protein")
    if cached is not None:
        return cached
    result = self._actually_compute(...)
    cache.put(provenance, result, "protein")
    return result

What gets cached:
    - :class:`molforge.core.Protein` from folding wrappers.
    - ``list[DesignedSequence]`` from generative wrappers.
    - :class:`molforge.docking.DockingResult` from docking wrappers
      (Vina, Gnina, DiffDock). Extra result types register via
      :func:`register_serializer`.

What deliberately doesn't get cached:
    - :class:`molforge.md.Trajectory`. Multi-GB per simulation; users
      who want this should use the upstream MD framework's
      checkpointing.

Cache location:
    - Default: ``~/.cache/molforge/`` (XDG convention).
    - Overridable via ``MOLFORGE_CACHE_DIR``.
    - Disable globally with ``MOLFORGE_CACHE=disabled``.

Cache layout:
    One subdirectory per entry, named by SHA-256 of the canonical
    key. Each entry holds:

    - ``type``: text file with the type tag
    - ``meta.json``: Protein name + metadata (with arrays + Provenance
      replaced by markers)
    - ``structure.cif`` (for Protein only): the AtomArray as mmCIF
    - ``payload.json`` (for DesignedSequence list): the design list
    - ``receptor.cif`` + ``pose_{i}.cif`` (for DockingResult): the
      receptor and each pose ligand as mmCIF, with scalar pose fields
      and metadata in ``payload.json``
    - ``arrays.npz``: numpy arrays from metadata when present

Safety:
    - Corrupted entries are treated as misses and logged; never crash.
    - Molforge major.minor version is part of the key — version
      upgrades invalidate transparently.
    - Timestamps are excluded from the key — different runs of the
      same computation share a slot.
    - Writes go to a ``.tmp`` directory and rename atomically.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

import molforge
from molforge.core.provenance import Provenance

if TYPE_CHECKING:
    from molforge.core import Protein
    from molforge.docking import DockingResult, Pose
    from molforge.freeenergy import FreeEnergyResult
    from molforge.generative import DesignedSequence


__all__ = [
    "CACHE_DIR_ENV",
    "CACHE_DISABLED_ENV",
    "Cache",
    "cache_key",
    "default_cache_dir",
    "get_default_cache",
    "register_serializer",
]


# ---------------------------------------------------------------------
# Env vars
# ---------------------------------------------------------------------

#: Set to ``"disabled"``/``"0"``/``"false"`` to disable the cache.
CACHE_DISABLED_ENV = "MOLFORGE_CACHE"

#: Override the default cache directory.
CACHE_DIR_ENV = "MOLFORGE_CACHE_DIR"

_DISABLED_VALUES = frozenset({"disabled", "0", "false", "off", "no"})
_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------


def _provenance_for_key(provenance: Provenance) -> dict[str, Any]:
    """Strip timestamps from Provenance for deterministic hashing.

    Walks the parent chain so timestamps anywhere up the chain are
    stripped. Everything else (engine, engine_version, parameters,
    inputs, parent chain) participates in the key.
    """
    out: dict[str, Any] = {
        "engine": provenance.engine,
        "engine_version": provenance.engine_version,
        "parameters": dict(provenance.parameters),
        "inputs": dict(provenance.inputs),
    }
    if provenance.parent is not None:
        out["parent"] = _provenance_for_key(provenance.parent)
    return out


def cache_key(provenance: Provenance) -> str:
    """Return the canonical 64-char hex cache key for a Provenance.

    Mixes the molforge major.minor version into the hash so version
    upgrades invalidate transparently. Timestamps are excluded so
    two runs of the same computation share a slot.
    """
    payload = {
        "molforge_version": _major_minor(),
        "provenance": _provenance_for_key(provenance),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _major_minor() -> str:
    parts = molforge.__version__.split(".")
    if len(parts) < 2:
        return molforge.__version__
    return f"{parts[0]}.{parts[1]}"


# ---------------------------------------------------------------------
# Cache directory
# ---------------------------------------------------------------------


def default_cache_dir() -> Path:
    """Resolve the default cache directory.

    Order: ``$MOLFORGE_CACHE_DIR`` → ``$XDG_CACHE_HOME/molforge`` →
    ``~/.cache/molforge``.
    """
    if env := os.environ.get(CACHE_DIR_ENV):
        return Path(env).expanduser()
    if xdg := os.environ.get("XDG_CACHE_HOME"):
        return Path(xdg).expanduser() / "molforge"
    return Path.home() / ".cache" / "molforge"


def _env_disabled() -> bool:
    val = os.environ.get(CACHE_DISABLED_ENV, "").strip().lower()
    return val in _DISABLED_VALUES


# ---------------------------------------------------------------------
# Serializer registry
# ---------------------------------------------------------------------


Serializer = Callable[[Any, Path], None]
Deserializer = Callable[[Path], Any]

_SERIALIZERS: dict[str, tuple[Serializer, Deserializer]] = {}


def register_serializer(type_tag: str, serializer: Serializer, deserializer: Deserializer) -> None:
    """Register a serializer/deserializer for a result type.

    The ``type_tag`` is stored alongside the entry and dispatched on
    lookup, so changing it invalidates previously-cached entries of
    the same shape. Pick a stable string.

    Re-registering overwrites; tests rely on that for cleanup.
    """
    _SERIALIZERS[type_tag] = (serializer, deserializer)


def _get_serializer(type_tag: str) -> tuple[Serializer, Deserializer]:
    if type_tag not in _SERIALIZERS:
        raise ValueError(
            f"No serializer registered for cache type {type_tag!r}. "
            f"Known types: {sorted(_SERIALIZERS.keys())}"
        )
    return _SERIALIZERS[type_tag]


# ---------------------------------------------------------------------
# Cache class
# ---------------------------------------------------------------------


class Cache:
    """File-system-backed cache for engine results.

    Args:
        directory: Cache directory. ``None`` uses :func:`default_cache_dir`.
        enabled: Master switch. ``None`` defers to
            ``$MOLFORGE_CACHE`` env var.
    """

    def __init__(
        self,
        directory: Path | str | None = None,
        *,
        enabled: bool | None = None,
    ) -> None:
        self.directory = Path(directory) if directory else default_cache_dir()
        if enabled is None:
            enabled = not _env_disabled()
        self.enabled = enabled

    def get(self, provenance: Provenance, type_tag: str) -> Any | None:
        """Look up a cached result. Returns ``None`` on miss or any error."""
        if not self.enabled:
            return None
        key = cache_key(provenance)
        entry = self.directory / key
        if not entry.is_dir():
            return None

        type_file = entry / "type"
        if not type_file.is_file():
            _logger.warning("Cache entry %s missing type file; miss", key[:12])
            return None
        try:
            stored_type = type_file.read_text(encoding="utf-8").strip()
        except OSError as e:
            _logger.warning("Cache entry %s unreadable: %s", key[:12], e)
            return None
        if stored_type != type_tag:
            _logger.warning(
                "Cache entry %s has type %r but caller wants %r; miss",
                key[:12],
                stored_type,
                type_tag,
            )
            return None

        try:
            _, deserializer = _get_serializer(type_tag)
        except ValueError as e:
            _logger.warning("Cache entry %s deserializer missing: %s", key[:12], e)
            return None
        try:
            return deserializer(entry)
        except Exception as e:
            _logger.warning("Cache entry %s failed to deserialize (%s); miss", key[:12], e)
            return None

    def put(self, provenance: Provenance, result: Any, type_tag: str) -> None:
        """Store a result. Errors are logged, never propagate."""
        if not self.enabled:
            return
        try:
            serializer, _ = _get_serializer(type_tag)
        except ValueError as e:
            _logger.warning("Cache.put failed: %s", e)
            return

        key = cache_key(provenance)
        entry = self.directory / key
        tmp = entry.with_name(entry.name + ".tmp")
        try:
            tmp.mkdir(parents=True, exist_ok=True)
            (tmp / "type").write_text(type_tag, encoding="utf-8")
            serializer(result, tmp)
        except Exception as e:
            _logger.warning("Cache write failed for %s (%s); cleaning up", key[:12], e)
            shutil.rmtree(tmp, ignore_errors=True)
            return

        if entry.exists():
            shutil.rmtree(tmp, ignore_errors=True)
            return
        try:
            tmp.rename(entry)
        except OSError as e:
            _logger.warning("Cache rename failed for %s (%s)", key[:12], e)
            shutil.rmtree(tmp, ignore_errors=True)

    def contains(self, provenance: Provenance) -> bool:
        """True if a cache entry exists for this Provenance."""
        if not self.enabled:
            return False
        return (self.directory / cache_key(provenance)).is_dir()

    def path_for(self, provenance: Provenance) -> Path:
        """On-disk path for an entry. May not exist (cache miss)."""
        return self.directory / cache_key(provenance)

    def clear(self) -> int:
        """Delete every entry. Only removes hex-named directories
        (defensive — never touches anything else in the cache dir)."""
        if not self.directory.is_dir():
            return 0
        n = 0
        for child in self.directory.iterdir():
            if child.is_dir() and _looks_like_cache_entry(child.name):
                shutil.rmtree(child, ignore_errors=True)
                n += 1
        return n

    def __repr__(self) -> str:
        state = "enabled" if self.enabled else "disabled"
        return f"Cache(directory={self.directory!r}, {state})"


def _looks_like_cache_entry(name: str) -> bool:
    return len(name) == 64 and all(c in "0123456789abcdef" for c in name)


# ---------------------------------------------------------------------
# Default singleton
# ---------------------------------------------------------------------

_default_cache: Cache | None = None


def get_default_cache() -> Cache:
    """Return a process-wide :class:`Cache` rooted at
    :func:`default_cache_dir`.

    Singleton — engines call this each time they want to consult the
    cache. ``MOLFORGE_CACHE=disabled`` is honoured at first-call time.
    """
    global _default_cache
    if _default_cache is None:
        _default_cache = Cache()
    return _default_cache


def _reset_default_cache_for_testing() -> None:
    """Test-only: clear the singleton so env-var changes take effect."""
    global _default_cache
    _default_cache = None


# =====================================================================
# Built-in serializers
# =====================================================================


_PROVENANCE_MARKER = "__provenance__"
_ARRAY_MARKER = "__array_ref__"
_COMPLEX_SPEC_MARKER = "__complex_spec__"


def _split_arrays(metadata: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate JSON-safe values from numpy arrays.

    Numpy arrays are pulled into a dict for ``np.savez``. Provenance
    objects are stored as dicts with a marker so we can rebuild
    them on read. ComplexSpec values get the same marker treatment.
    """
    safe: dict[str, Any] = {}
    arrays: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, np.ndarray):
            arrays[key] = value
            safe[key] = {_ARRAY_MARKER: key}
        elif isinstance(value, Provenance):
            safe[key] = {_PROVENANCE_MARKER: value.to_dict()}
        else:
            safe[key] = _make_json_safe(value)
    return safe, arrays


def _restore_metadata(safe: dict[str, Any]) -> dict[str, Any]:
    """Inverse of :func:`_split_arrays`'s safe payload."""
    # Lazy import — avoid molforge.folding at module import time.
    from molforge.folding import ComplexSpec, Entity

    out: dict[str, Any] = {}
    for key, value in safe.items():
        if isinstance(value, dict) and _PROVENANCE_MARKER in value:
            out[key] = Provenance.from_dict(value[_PROVENANCE_MARKER])
        elif isinstance(value, dict) and _COMPLEX_SPEC_MARKER in value:
            payload = value[_COMPLEX_SPEC_MARKER]
            entities = tuple(
                Entity(**{k: v for k, v in e.items() if k != "chain_ids"})
                for e in payload["entities"]
            )
            out[key] = ComplexSpec(entities=entities)
        elif isinstance(value, dict) and _ARRAY_MARKER in value:
            # Skip; caller layers arrays in from the npz.
            continue
        else:
            out[key] = value
    return out


def _make_json_safe(value: Any) -> Any:
    """Best-effort conversion of arbitrary metadata to JSON-safe shapes."""
    # Lazy import to avoid a top-level dependency on molforge.folding.
    from molforge.folding import ComplexSpec

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, ComplexSpec):
        # Serialize ComplexSpec via its entities. Discard auto-assigned
        # chain IDs (they'll be re-derived on the rehydrated spec) so
        # round-trip preserves the user-facing shape.
        return {
            _COMPLEX_SPEC_MARKER: {
                "entities": [
                    {
                        "kind": e.kind,
                        "sequence": e.sequence,
                        "smiles": e.smiles,
                        "ccd": e.ccd,
                        "chain_id": e.chain_id,
                        "copies": e.copies,
                        "name": e.name,
                    }
                    for e in value.entities
                ],
            }
        }
    warnings.warn(
        f"Cache: value of type {type(value).__name__} not JSON-serializable; storing repr only",
        stacklevel=3,
    )
    return repr(value)


# ---------------------------------------------------------------------
# Protein serializer
# ---------------------------------------------------------------------


def _serialize_protein(protein: Protein, entry: Path) -> None:
    """Write a Protein to ``entry`` as mmCIF + metadata JSON + npz."""
    from molforge.io.mmcif import write_cif_string

    (entry / "structure.cif").write_text(write_cif_string(protein), encoding="utf-8")
    metadata_safe, arrays = _split_arrays(protein.metadata)
    payload = {"name": protein.name, "metadata": metadata_safe}
    (entry / "meta.json").write_text(json.dumps(payload), encoding="utf-8")
    if arrays:
        np.savez(entry / "arrays.npz", **arrays)


def _deserialize_protein(entry: Path) -> Protein:
    """Inverse of :func:`_serialize_protein`."""
    from molforge.io.mmcif import read_cif_string

    cif_text = (entry / "structure.cif").read_text(encoding="utf-8")
    protein = read_cif_string(cif_text)

    payload = json.loads((entry / "meta.json").read_text(encoding="utf-8"))
    metadata = _restore_metadata(payload["metadata"])

    arrays_path = entry / "arrays.npz"
    if arrays_path.is_file():
        with np.load(arrays_path, allow_pickle=False) as npz:
            for key in npz.files:
                metadata[key] = npz[key]

    if payload["name"]:
        protein.name = payload["name"]
    protein.metadata.update(metadata)
    return protein


# ---------------------------------------------------------------------
# DesignedSequence list serializer
# ---------------------------------------------------------------------


def _serialize_designed_sequences(designs: list[DesignedSequence], entry: Path) -> None:
    """Write a list of DesignedSequence as JSON + npz for any arrays."""
    items: list[dict[str, Any]] = []
    all_arrays: dict[str, Any] = {}
    for i, d in enumerate(designs):
        metadata_safe, arrays = _split_arrays(d.metadata)
        items.append(
            {
                "sequence": d.sequence,
                "score": float(d.score),
                "recovery": None if d.recovery is None else float(d.recovery),
                "metadata": metadata_safe,
            }
        )
        for k, v in arrays.items():
            all_arrays[f"{i}__{k}"] = v

    (entry / "payload.json").write_text(json.dumps({"designs": items}), encoding="utf-8")
    if all_arrays:
        np.savez(entry / "arrays.npz", **all_arrays)


def _deserialize_designed_sequences(entry: Path) -> list[DesignedSequence]:
    """Inverse of :func:`_serialize_designed_sequences`."""
    from molforge.generative import DesignedSequence

    payload = json.loads((entry / "payload.json").read_text(encoding="utf-8"))
    arrays_by_index: dict[int, dict[str, Any]] = {}
    arrays_path = entry / "arrays.npz"
    if arrays_path.is_file():
        with np.load(arrays_path, allow_pickle=False) as npz:
            for key in npz.files:
                idx_str, _, real_key = key.partition("__")
                arrays_by_index.setdefault(int(idx_str), {})[real_key] = npz[key]

    out: list[DesignedSequence] = []
    for i, item in enumerate(payload["designs"]):
        metadata = _restore_metadata(item["metadata"])
        metadata.update(arrays_by_index.get(i, {}))
        out.append(
            DesignedSequence(
                sequence=item["sequence"],
                score=item["score"],
                recovery=item.get("recovery"),
                metadata=metadata,
            )
        )
    return out


# ---------------------------------------------------------------------
# DockingResult serializer
# ---------------------------------------------------------------------


def _serialize_protein_member(
    protein: Protein,
    entry: Path,
    *,
    cif_name: str,
    array_slot: str,
    arrays_out: dict[str, Any],
) -> dict[str, Any]:
    """Serialize one Protein embedded in a larger result.

    Writes the structure to ``cif_name`` and returns a JSON-safe
    ``{"name", "metadata"}`` payload. Any numpy arrays in the
    protein's metadata are funnelled into the shared ``arrays_out``
    dict under ``f"{array_slot}__{key}"`` so a single ``arrays.npz``
    holds every array in the entry (one npz per entry, keyed by slot).
    """
    from molforge.io.mmcif import write_cif_string

    (entry / cif_name).write_text(write_cif_string(protein), encoding="utf-8")
    metadata_safe, arrays = _split_arrays(protein.metadata)
    for key, value in arrays.items():
        arrays_out[f"{array_slot}__{key}"] = value
    return {"name": protein.name, "metadata": metadata_safe}


def _deserialize_protein_member(
    entry: Path,
    payload: dict[str, Any],
    *,
    cif_name: str,
    slot_arrays: dict[str, Any],
) -> Protein:
    """Inverse of :func:`_serialize_protein_member`.

    ``slot_arrays`` is the per-slot array dict already partitioned out
    of the entry's ``arrays.npz`` (keys are the real metadata keys,
    with the slot prefix stripped).
    """
    from molforge.io.mmcif import read_cif_string

    protein = read_cif_string((entry / cif_name).read_text(encoding="utf-8"))
    metadata = _restore_metadata(payload["metadata"])
    metadata.update(slot_arrays)
    if payload["name"]:
        protein.name = payload["name"]
    protein.metadata.update(metadata)
    return protein


def _serialize_docking_result(result: DockingResult, entry: Path) -> None:
    """Write a DockingResult as per-member mmCIF + JSON + npz.

    Layout inside ``entry``:

    - ``receptor.cif`` — the receptor structure (omitted when the
      result has no receptor).
    - ``pose_{i}.cif`` — each pose's ligand structure.
    - ``payload.json`` — engine name, scalar pose fields
      (score / rank / rmsd bounds), and every member's name + metadata
      with arrays and Provenance replaced by markers.
    - ``arrays.npz`` — all numpy arrays from any member's metadata,
      namespaced by slot (``receptor``, ``ligand{i}``, ``pose{i}``,
      ``result``) so they survive the round-trip without colliding.

    The result-level Provenance (under ``metadata["provenance"]``)
    round-trips through :func:`_split_arrays` like any other
    Provenance value — that's what keys the cache, so it must come
    back as a real :class:`Provenance`.
    """
    all_arrays: dict[str, Any] = {}

    receptor_payload: dict[str, Any] | None = None
    if result.receptor is not None:
        receptor_payload = _serialize_protein_member(
            result.receptor,
            entry,
            cif_name="receptor.cif",
            array_slot="receptor",
            arrays_out=all_arrays,
        )

    pose_payloads: list[dict[str, Any]] = []
    for i, pose in enumerate(result.poses):
        ligand_payload = _serialize_protein_member(
            pose.ligand,
            entry,
            cif_name=f"pose_{i}.cif",
            array_slot=f"ligand{i}",
            arrays_out=all_arrays,
        )
        pose_metadata_safe, pose_arrays = _split_arrays(pose.metadata)
        for key, value in pose_arrays.items():
            all_arrays[f"pose{i}__{key}"] = value
        pose_payloads.append(
            {
                "score": float(pose.score),
                "rank": int(pose.rank),
                "rmsd_lb": None if pose.rmsd_lb is None else float(pose.rmsd_lb),
                "rmsd_ub": None if pose.rmsd_ub is None else float(pose.rmsd_ub),
                "ligand": ligand_payload,
                "metadata": pose_metadata_safe,
            }
        )

    result_metadata_safe, result_arrays = _split_arrays(result.metadata)
    for key, value in result_arrays.items():
        all_arrays[f"result__{key}"] = value

    payload = {
        "engine": result.engine,
        "receptor": receptor_payload,
        "poses": pose_payloads,
        "metadata": result_metadata_safe,
    }
    (entry / "payload.json").write_text(json.dumps(payload), encoding="utf-8")
    if all_arrays:
        np.savez(entry / "arrays.npz", **all_arrays)


def _deserialize_docking_result(entry: Path) -> DockingResult:
    """Inverse of :func:`_serialize_docking_result`."""
    from molforge.docking import DockingResult, Pose

    payload = json.loads((entry / "payload.json").read_text(encoding="utf-8"))

    # Partition the shared npz back into per-slot array dicts.
    arrays_by_slot: dict[str, dict[str, Any]] = {}
    arrays_path = entry / "arrays.npz"
    if arrays_path.is_file():
        with np.load(arrays_path, allow_pickle=False) as npz:
            for key in npz.files:
                slot, _, real_key = key.partition("__")
                arrays_by_slot.setdefault(slot, {})[real_key] = npz[key]

    receptor: Protein | None = None
    receptor_payload = payload.get("receptor")
    if receptor_payload is not None:
        receptor = _deserialize_protein_member(
            entry,
            receptor_payload,
            cif_name="receptor.cif",
            slot_arrays=arrays_by_slot.get("receptor", {}),
        )

    poses: list[Pose] = []
    for i, pose_payload in enumerate(payload["poses"]):
        ligand = _deserialize_protein_member(
            entry,
            pose_payload["ligand"],
            cif_name=f"pose_{i}.cif",
            slot_arrays=arrays_by_slot.get(f"ligand{i}", {}),
        )
        pose_metadata = _restore_metadata(pose_payload["metadata"])
        pose_metadata.update(arrays_by_slot.get(f"pose{i}", {}))
        poses.append(
            Pose(
                ligand=ligand,
                score=pose_payload["score"],
                rank=pose_payload["rank"],
                rmsd_lb=pose_payload.get("rmsd_lb"),
                rmsd_ub=pose_payload.get("rmsd_ub"),
                metadata=pose_metadata,
            )
        )

    result_metadata = _restore_metadata(payload["metadata"])
    result_metadata.update(arrays_by_slot.get("result", {}))

    return DockingResult(
        poses=poses,
        receptor=receptor,
        engine=payload["engine"],
        metadata=result_metadata,
    )


# ---------------------------------------------------------------------
# Free-energy serializer
# ---------------------------------------------------------------------


def _serialize_free_energy_result(result: FreeEnergyResult, entry: Path) -> None:
    """Write a FreeEnergyResult as JSON + npz.

    ``payload.json`` holds the scalars (delta_g, uncertainty, method),
    the component breakdown (or ``null``), the top-level Provenance as a
    dict (it keys the cache, so it must come back real), and the
    JSON-safe metadata. ``arrays.npz`` holds the optional convergence
    trace plus any numpy arrays from metadata, namespaced by slot.
    """
    components = None
    if result.components is not None:
        c = result.components
        components = {
            "vdw": float(c.vdw),
            "electrostatic": float(c.electrostatic),
            "polar_solvation": float(c.polar_solvation),
            "nonpolar_solvation": float(c.nonpolar_solvation),
            "entropy": None if c.entropy is None else float(c.entropy),
        }

    metadata_safe, metadata_arrays = _split_arrays(result.metadata)

    decomposition = None
    if result.decomposition is not None:
        decomposition = [
            {
                "residue": c.residue,
                "total": float(c.total),
                "uncertainty": float(c.uncertainty),
                "internal": float(c.internal),
                "vdw": float(c.vdw),
                "electrostatic": float(c.electrostatic),
                "polar_solvation": float(c.polar_solvation),
                "nonpolar_solvation": float(c.nonpolar_solvation),
            }
            for c in result.decomposition.residues
        ]

    payload = {
        "delta_g": float(result.delta_g),
        "uncertainty": float(result.uncertainty),
        "method": result.method,
        "components": components,
        "provenance": None if result.provenance is None else result.provenance.to_dict(),
        "metadata": metadata_safe,
        "decomposition": decomposition,
    }
    (entry / "payload.json").write_text(json.dumps(payload), encoding="utf-8")

    arrays: dict[str, Any] = {f"meta__{k}": v for k, v in metadata_arrays.items()}
    if result.convergence is not None:
        arrays["convergence"] = np.asarray(result.convergence)
    if arrays:
        np.savez(entry / "arrays.npz", **arrays)


def _deserialize_free_energy_result(entry: Path) -> FreeEnergyResult:
    """Inverse of :func:`_serialize_free_energy_result`."""
    from molforge.freeenergy import (
        Decomposition,
        FreeEnergyComponents,
        FreeEnergyResult,
        ResidueContribution,
    )

    payload = json.loads((entry / "payload.json").read_text(encoding="utf-8"))

    convergence = None
    meta_arrays: dict[str, Any] = {}
    arrays_path = entry / "arrays.npz"
    if arrays_path.is_file():
        with np.load(arrays_path, allow_pickle=False) as npz:
            for key in npz.files:
                if key == "convergence":
                    convergence = npz[key]
                else:
                    _, _, real_key = key.partition("__")
                    meta_arrays[real_key] = npz[key]

    components = None
    if payload["components"] is not None:
        components = FreeEnergyComponents(**payload["components"])

    provenance = None
    if payload["provenance"] is not None:
        provenance = Provenance.from_dict(payload["provenance"])

    metadata = _restore_metadata(payload["metadata"])
    metadata.update(meta_arrays)

    decomposition = None
    if payload.get("decomposition") is not None:
        decomposition = Decomposition(
            [ResidueContribution(**row) for row in payload["decomposition"]]
        )

    return FreeEnergyResult(
        delta_g=payload["delta_g"],
        uncertainty=payload["uncertainty"],
        method=payload["method"],
        components=components,
        convergence=convergence,
        provenance=provenance,
        metadata=metadata,
        decomposition=decomposition,
    )


# ---------------------------------------------------------------------
# Register built-ins
# ---------------------------------------------------------------------

register_serializer("protein", _serialize_protein, _deserialize_protein)
register_serializer(
    "designed_sequences",
    _serialize_designed_sequences,
    _deserialize_designed_sequences,
)
register_serializer(
    "docking_result",
    _serialize_docking_result,
    _deserialize_docking_result,
)
register_serializer(
    "free_energy_result",
    _serialize_free_energy_result,
    _deserialize_free_energy_result,
)
