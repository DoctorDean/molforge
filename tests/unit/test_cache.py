"""Tests for :mod:`molforge.cache`.

The cache lives between the engine and the underlying compute. Each
test constructs its own :class:`Cache` rooted at ``tmp_path`` (or in
some cases temporarily overrides the env var). The autouse
``_isolate_cache`` fixture in ``tests/conftest.py`` redirects the
*default* cache for every test in the suite; these tests don't rely
on the default — they construct their own.

What we test:

- Cache key determinism (timestamps stripped; parent chain matters)
- Key changes when params/inputs change
- Molforge version is part of the key
- Round-trip Protein (atom array + metadata + numpy arrays + Provenance)
- Round-trip DesignedSequence list
- ComplexSpec round-trips through metadata
- Corruption recovery (missing files, wrong type tag) → miss, not crash
- Env-var control (disabled, dir override)
- ``clear()`` only removes hex-named entries
- ``contains()`` / ``path_for()``
- Disabled cache: ``get`` returns None, ``put`` is a no-op
- Concurrent writes: last-write-wins, no partial entries
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import molforge
from molforge.cache import (
    CACHE_DIR_ENV,
    CACHE_DISABLED_ENV,
    Cache,
    _reset_default_cache_for_testing,
    cache_key,
    default_cache_dir,
    get_default_cache,
)
from molforge.core import AtomArray, Protein
from molforge.core import metadata_keys as mk
from molforge.core.provenance import Provenance
from molforge.folding import ComplexSpec
from molforge.generative import DesignedSequence

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _make_provenance(
    engine: str = "ESMFold",
    parameters: dict | None = None,
    inputs: dict | None = None,
    parent: Provenance | None = None,
) -> Provenance:
    return Provenance.from_engine(
        engine=engine,
        parameters=parameters or {"model": "v1"},
        inputs=inputs or {"sequence": "MKQH"},
        parent=parent,
    )


def _make_protein(name: str = "test", n_atoms: int = 3) -> Protein:
    arr = AtomArray(n_atoms)
    arr.coords[:] = np.array([[i * 1.5, 0.0, 0.0] for i in range(n_atoms)], dtype=np.float32)
    arr.element[:] = ["N", "C", "C"][:n_atoms]
    arr.atom_name[:] = ["N", "CA", "C"][:n_atoms]
    arr.residue_name[:] = "ALA"
    arr.residue_id[:] = 1
    arr.chain_id[:] = "A"
    arr.b_factor[:] = 80.0
    return Protein(arr, name=name)


# ---------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------


class TestCacheKeyDeterminism:
    def test_same_inputs_produce_same_key(self) -> None:
        k1 = cache_key(_make_provenance())
        k2 = cache_key(_make_provenance())
        assert k1 == k2

    def test_timestamps_excluded(self) -> None:
        """Two Provenance objects with the same data but different
        timestamps must produce the same key — different runs of the
        same computation share a cache slot."""
        p1 = _make_provenance()
        # Hand-build a second with an obviously-different timestamp.
        p2 = Provenance(
            engine=p1.engine,
            engine_version=p1.engine_version,
            molforge_version=p1.molforge_version,
            timestamp="1999-01-01T00:00:00Z",
            parameters=dict(p1.parameters),
            inputs=dict(p1.inputs),
        )
        assert cache_key(p1) == cache_key(p2)

    def test_key_is_64_char_hex(self) -> None:
        key = cache_key(_make_provenance())
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)


class TestCacheKeyChanges:
    def test_different_engine(self) -> None:
        k1 = cache_key(_make_provenance(engine="ESMFold"))
        k2 = cache_key(_make_provenance(engine="Boltz"))
        assert k1 != k2

    def test_different_parameters(self) -> None:
        k1 = cache_key(_make_provenance(parameters={"model": "v1"}))
        k2 = cache_key(_make_provenance(parameters={"model": "v2"}))
        assert k1 != k2

    def test_different_inputs(self) -> None:
        k1 = cache_key(_make_provenance(inputs={"sequence": "MKQH"}))
        k2 = cache_key(_make_provenance(inputs={"sequence": "HISH"}))
        assert k1 != k2

    def test_parent_chain_matters(self) -> None:
        parent = _make_provenance(engine="RFdiffusion")
        k1 = cache_key(_make_provenance(parent=None))
        k2 = cache_key(_make_provenance(parent=parent))
        assert k1 != k2

    def test_molforge_version_part_of_key(self) -> None:
        """A change in molforge's major.minor invalidates the key.
        We can't easily test a real version bump in isolation, but
        we can verify the version is reflected in the hash by
        monkey-patching :data:`molforge.__version__`."""
        k1 = cache_key(_make_provenance())
        original_version = molforge.__version__
        try:
            molforge.__version__ = "999.0.0"
            k2 = cache_key(_make_provenance())
            assert k1 != k2
        finally:
            molforge.__version__ = original_version


# ---------------------------------------------------------------------
# Default cache dir
# ---------------------------------------------------------------------


class TestDefaultCacheDir:
    def test_env_var_overrides(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path / "explicit"))
        assert default_cache_dir() == tmp_path / "explicit"

    def test_xdg_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(CACHE_DIR_ENV, raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        assert default_cache_dir() == tmp_path / "xdg" / "molforge"

    def test_default_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(CACHE_DIR_ENV, raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        assert default_cache_dir() == Path.home() / ".cache" / "molforge"


# ---------------------------------------------------------------------
# Round-trip: Protein
# ---------------------------------------------------------------------


class TestProteinRoundTrip:
    def test_basic_protein(self, tmp_path: Path) -> None:
        cache = Cache(directory=tmp_path)
        prov = _make_provenance()
        protein = _make_protein()

        assert cache.get(prov, "protein") is None  # miss
        cache.put(prov, protein, "protein")
        restored = cache.get(prov, "protein")

        assert restored is not None
        assert restored.name == "test"
        assert restored.atom_array.n_atoms == 3

    def test_numpy_arrays_in_metadata(self, tmp_path: Path) -> None:
        cache = Cache(directory=tmp_path)
        prov = _make_provenance()
        protein = _make_protein()
        protein.metadata[mk.CONFIDENCE_PER_RESIDUE] = np.array([80.0, 75.0, 90.0], dtype=np.float32)
        protein.metadata[mk.MEAN_CONFIDENCE] = 81.67

        cache.put(prov, protein, "protein")
        restored = cache.get(prov, "protein")
        np.testing.assert_array_equal(
            restored.metadata[mk.CONFIDENCE_PER_RESIDUE], [80.0, 75.0, 90.0]
        )
        assert restored.metadata[mk.MEAN_CONFIDENCE] == pytest.approx(81.67)

    def test_provenance_rebuilt(self, tmp_path: Path) -> None:
        """Provenance in metadata survives round-trip as a real
        Provenance instance (not as a dict)."""
        cache = Cache(directory=tmp_path)
        prov = _make_provenance(engine="ESMFold", parameters={"model": "v1"})
        protein = _make_protein()
        protein.metadata[mk.PROVENANCE] = prov

        cache.put(prov, protein, "protein")
        restored = cache.get(prov, "protein")
        restored_prov = restored.metadata[mk.PROVENANCE]
        assert isinstance(restored_prov, Provenance)
        assert restored_prov.engine == "ESMFold"
        assert restored_prov.parameters == {"model": "v1"}

    def test_complex_spec_in_metadata(self, tmp_path: Path) -> None:
        """ComplexSpec values in metadata round-trip as ComplexSpec
        instances, not as repr strings."""
        cache = Cache(directory=tmp_path)
        prov = _make_provenance(engine="Boltz")
        protein = _make_protein()
        spec = ComplexSpec.protein_ligand(protein_sequence="MKQH", ligand_smiles="CCO")
        protein.metadata["complex_spec"] = spec

        cache.put(prov, protein, "protein")
        restored = cache.get(prov, "protein")
        rs = restored.metadata["complex_spec"]
        assert isinstance(rs, ComplexSpec)
        assert rs == spec  # ComplexSpec is a frozen dataclass, __eq__ works


# ---------------------------------------------------------------------
# Round-trip: DesignedSequence list
# ---------------------------------------------------------------------


class TestDesignedSequenceRoundTrip:
    def test_basic_list(self, tmp_path: Path) -> None:
        cache = Cache(directory=tmp_path)
        prov = _make_provenance(engine="ProteinMPNN")
        designs = [
            DesignedSequence(
                sequence="MKQH",
                score=1.5,
                recovery=0.75,
                metadata={"engine": "ProteinMPNN", "seed": 42},
            ),
            DesignedSequence(
                sequence="HISH",
                score=1.7,
                recovery=None,  # exercise the None-recovery path
                metadata={"engine": "ProteinMPNN"},
            ),
        ]

        cache.put(prov, designs, "designed_sequences")
        restored = cache.get(prov, "designed_sequences")

        assert restored is not None
        assert len(restored) == 2
        assert restored[0].sequence == "MKQH"
        assert restored[0].score == 1.5
        assert restored[0].recovery == 0.75
        assert restored[0].metadata["seed"] == 42
        assert restored[1].recovery is None

    def test_arrays_per_design(self, tmp_path: Path) -> None:
        """Per-position numpy arrays in DesignedSequence metadata
        survive round-trip; each design's arrays come back distinct."""
        cache = Cache(directory=tmp_path)
        prov = _make_provenance(engine="ProteinMPNN")
        designs = [
            DesignedSequence(
                sequence="MKQH",
                score=1.5,
                metadata={"per_pos_score": np.array([0.5, 0.6, 0.7, 0.8])},
            ),
            DesignedSequence(
                sequence="HISH",
                score=1.7,
                metadata={"per_pos_score": np.array([0.1, 0.2, 0.3, 0.4])},
            ),
        ]
        cache.put(prov, designs, "designed_sequences")
        restored = cache.get(prov, "designed_sequences")
        np.testing.assert_array_equal(restored[0].metadata["per_pos_score"], [0.5, 0.6, 0.7, 0.8])
        np.testing.assert_array_equal(restored[1].metadata["per_pos_score"], [0.1, 0.2, 0.3, 0.4])


# ---------------------------------------------------------------------
# Corruption recovery
# ---------------------------------------------------------------------


class TestCorruption:
    def test_missing_type_file_is_miss(self, tmp_path: Path) -> None:
        cache = Cache(directory=tmp_path)
        prov = _make_provenance()
        cache.put(prov, _make_protein(), "protein")
        # Corrupt: delete the type file.
        (cache.path_for(prov) / "type").unlink()
        # Lookup must be a miss, not a crash.
        assert cache.get(prov, "protein") is None

    def test_wrong_type_tag_is_miss(self, tmp_path: Path) -> None:
        """An entry stored as 'protein' but requested as
        'designed_sequences' is a miss — guards against hash
        collisions."""
        cache = Cache(directory=tmp_path)
        prov = _make_provenance()
        cache.put(prov, _make_protein(), "protein")
        assert cache.get(prov, "designed_sequences") is None

    def test_missing_payload_file_is_miss(self, tmp_path: Path) -> None:
        cache = Cache(directory=tmp_path)
        prov = _make_provenance()
        cache.put(prov, _make_protein(), "protein")
        # Corrupt: delete the structure file.
        (cache.path_for(prov) / "structure.cif").unlink()
        assert cache.get(prov, "protein") is None

    def test_garbage_json_is_miss(self, tmp_path: Path) -> None:
        cache = Cache(directory=tmp_path)
        prov = _make_provenance()
        cache.put(prov, _make_protein(), "protein")
        (cache.path_for(prov) / "meta.json").write_text("{not json")
        assert cache.get(prov, "protein") is None


# ---------------------------------------------------------------------
# Disabled cache
# ---------------------------------------------------------------------


class TestDisabled:
    def test_explicit_disabled(self, tmp_path: Path) -> None:
        cache = Cache(directory=tmp_path, enabled=False)
        prov = _make_provenance()
        # Put is a no-op.
        cache.put(prov, _make_protein(), "protein")
        # Nothing was written.
        assert not (tmp_path / cache_key(prov)).exists()
        # Get returns None.
        assert cache.get(prov, "protein") is None
        # contains() respects disabled flag too.
        assert cache.contains(prov) is False

    def test_env_var_disabled(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(CACHE_DISABLED_ENV, "disabled")
        cache = Cache(directory=tmp_path)
        assert cache.enabled is False

    @pytest.mark.parametrize("value", ["disabled", "0", "false", "FALSE", "off", "no"])
    def test_env_var_disabled_values(
        self,
        value: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(CACHE_DISABLED_ENV, value)
        cache = Cache(directory=tmp_path)
        assert cache.enabled is False

    def test_env_var_enabled_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(CACHE_DISABLED_ENV, raising=False)
        cache = Cache(directory=tmp_path)
        assert cache.enabled is True


# ---------------------------------------------------------------------
# Clear / contains / path_for
# ---------------------------------------------------------------------


class TestClearContainsPath:
    def test_contains(self, tmp_path: Path) -> None:
        cache = Cache(directory=tmp_path)
        prov = _make_provenance()
        assert cache.contains(prov) is False
        cache.put(prov, _make_protein(), "protein")
        assert cache.contains(prov) is True

    def test_path_for_returns_predictable_path(self, tmp_path: Path) -> None:
        cache = Cache(directory=tmp_path)
        prov = _make_provenance()
        path = cache.path_for(prov)
        assert path.parent == tmp_path
        assert path.name == cache_key(prov)

    def test_clear_removes_only_hex_dirs(self, tmp_path: Path) -> None:
        """clear() must not touch arbitrary files in the cache dir —
        defensive against a user pointing the cache at the wrong
        directory."""
        cache = Cache(directory=tmp_path)
        # Cache entry.
        cache.put(_make_provenance(), _make_protein(), "protein")
        # An unrelated file the user shouldn't lose.
        (tmp_path / "important_notes.txt").write_text("don't delete me")
        # An unrelated directory.
        (tmp_path / "other_subdir").mkdir()
        (tmp_path / "other_subdir" / "data.txt").write_text("preserve me too")

        n = cache.clear()
        assert n == 1
        assert (tmp_path / "important_notes.txt").exists()
        assert (tmp_path / "other_subdir" / "data.txt").exists()

    def test_clear_empty_cache(self, tmp_path: Path) -> None:
        cache = Cache(directory=tmp_path)
        assert cache.clear() == 0

    def test_clear_nonexistent_directory(self, tmp_path: Path) -> None:
        cache = Cache(directory=tmp_path / "does_not_exist")
        assert cache.clear() == 0


# ---------------------------------------------------------------------
# Default singleton
# ---------------------------------------------------------------------


class TestDefaultSingleton:
    def test_returns_same_object(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path))
        _reset_default_cache_for_testing()
        c1 = get_default_cache()
        c2 = get_default_cache()
        assert c1 is c2

    def test_respects_env_var(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path / "explicit"))
        _reset_default_cache_for_testing()
        c = get_default_cache()
        assert c.directory == tmp_path / "explicit"


# ---------------------------------------------------------------------
# Atomicity
# ---------------------------------------------------------------------


class TestAtomicity:
    def test_failed_put_leaves_no_partial_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the serializer raises midway, no partial entry remains
        in the cache directory."""
        from molforge.cache import register_serializer

        # Register a serializer that always fails.
        def _broken_serializer(value, entry):
            (entry / "started.txt").write_text("partial")
            raise RuntimeError("boom")

        def _broken_deserializer(entry):
            raise RuntimeError("never called")

        register_serializer("test_broken", _broken_serializer, _broken_deserializer)

        cache = Cache(directory=tmp_path)
        prov = _make_provenance()
        cache.put(prov, "anything", "test_broken")

        # No entry, no temp directory left behind.
        contents = list(tmp_path.iterdir())
        assert contents == [], f"Expected empty cache dir, found {contents}"


# ---------------------------------------------------------------------
# Source inspection
# ---------------------------------------------------------------------


class TestSourceInspection:
    def test_disabled_values_set(self) -> None:
        """The set of strings that disable the cache must include
        the documented values."""
        from molforge.cache import _DISABLED_VALUES

        for value in {"disabled", "0", "false", "off", "no"}:
            assert value in _DISABLED_VALUES

    def test_protein_and_designed_sequences_registered(self) -> None:
        """The two built-in serializers are registered at import time."""
        from molforge.cache import _SERIALIZERS

        assert "protein" in _SERIALIZERS
        assert "designed_sequences" in _SERIALIZERS
