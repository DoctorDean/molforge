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
- Round-trip DockingResult (poses + ligand/receptor structures +
  per-pose arrays + Provenance; empty + no-receptor edges)
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
from molforge.docking import DockingResult, Pose
from molforge.folding import ComplexSpec
from molforge.freeenergy import (
    Decomposition,
    FreeEnergyComponents,
    FreeEnergyResult,
    ResidueContribution,
)
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


def _make_ligand(name: str = "lig", n_atoms: int = 3) -> Protein:
    """A small HETATM 'ligand' Protein, the shape a docking pose holds."""
    arr = AtomArray(n_atoms)
    arr.coords[:] = np.array([[i * 1.4, 0.0, 0.0] for i in range(n_atoms)], dtype=np.float32)
    arr.element[:] = ["C", "O", "N"][:n_atoms]
    arr.atom_name[:] = ["C1", "O1", "N1"][:n_atoms]
    arr.residue_name[:] = "LIG"
    arr.residue_id[:] = 1
    arr.chain_id[:] = "X"
    arr.record_type[:] = "HETATM"
    arr.entity_type[:] = "ligand"
    return Protein(arr, name=name)


def _make_docking_result(
    *,
    engine: str = "Vina",
    n_poses: int = 2,
    with_receptor: bool = True,
    provenance: Provenance | None = None,
) -> DockingResult:
    poses = [
        Pose(
            ligand=_make_ligand(f"pose{i}"),
            score=-8.0 + i,
            rank=i,
            rmsd_lb=0.0 if i == 0 else float(i),
            rmsd_ub=0.0 if i == 0 else float(i) + 1.0,
            metadata={"engine": engine},
        )
        for i in range(n_poses)
    ]
    metadata: dict = {"center": [10.0, 5.0, -2.0]}
    if provenance is not None:
        metadata[mk.PROVENANCE] = provenance
    return DockingResult(
        poses=poses,
        receptor=_make_protein("receptor") if with_receptor else None,
        engine=engine,
        metadata=metadata,
    )


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
# Round-trip: DockingResult
# ---------------------------------------------------------------------


class TestDockingResultRoundTrip:
    def test_basic_result(self, tmp_path: Path) -> None:
        cache = Cache(directory=tmp_path)
        prov = _make_provenance(engine="Vina")
        result = _make_docking_result(engine="Vina", n_poses=3)

        assert cache.get(prov, "docking_result") is None  # miss
        cache.put(prov, result, "docking_result")
        restored = cache.get(prov, "docking_result")

        assert restored is not None
        assert restored.engine == "Vina"
        assert len(restored) == 3
        # Poses keep order + scalar fields.
        assert restored.best.rank == 0
        assert restored.best.score == pytest.approx(-8.0)
        assert restored.poses[1].rmsd_lb == pytest.approx(1.0)
        assert restored.poses[1].rmsd_ub == pytest.approx(2.0)
        # Ligand structures survive as Proteins.
        assert restored.poses[0].ligand.name == "pose0"
        assert restored.poses[0].ligand.atom_array.n_atoms == 3
        assert list(restored.poses[0].ligand.atom_array.element) == ["C", "O", "N"]

    def test_receptor_round_trips(self, tmp_path: Path) -> None:
        cache = Cache(directory=tmp_path)
        prov = _make_provenance(engine="Gnina")
        result = _make_docking_result(engine="Gnina", with_receptor=True)

        cache.put(prov, result, "docking_result")
        restored = cache.get(prov, "docking_result")

        assert restored.receptor is not None
        assert restored.receptor.name == "receptor"
        assert restored.receptor.atom_array.n_atoms == 3

    def test_no_receptor(self, tmp_path: Path) -> None:
        cache = Cache(directory=tmp_path)
        prov = _make_provenance(engine="DiffDock")
        result = _make_docking_result(engine="DiffDock", with_receptor=False)

        cache.put(prov, result, "docking_result")
        restored = cache.get(prov, "docking_result")

        assert restored.receptor is None
        assert len(restored) == 2

    def test_empty_result(self, tmp_path: Path) -> None:
        """A DockingResult with no poses and no receptor round-trips."""
        cache = Cache(directory=tmp_path)
        prov = _make_provenance(engine="Vina")
        result = DockingResult(poses=[], receptor=None, engine="Vina", metadata={})

        cache.put(prov, result, "docking_result")
        restored = cache.get(prov, "docking_result")

        assert restored is not None
        assert restored.engine == "Vina"
        assert len(restored) == 0
        assert restored.receptor is None

    def test_provenance_rebuilt(self, tmp_path: Path) -> None:
        """Result-level Provenance survives as a real Provenance (the
        thing the cache keys on), not a dict."""
        cache = Cache(directory=tmp_path)
        prov = _make_provenance(
            engine="Vina",
            parameters={"exhaustiveness": 8, "n_poses": 9},
        )
        result = _make_docking_result(engine="Vina", provenance=prov)

        cache.put(prov, result, "docking_result")
        restored = cache.get(prov, "docking_result")

        restored_prov = restored.metadata[mk.PROVENANCE]
        assert isinstance(restored_prov, Provenance)
        assert restored_prov.engine == "Vina"
        assert restored_prov.parameters == {"exhaustiveness": 8, "n_poses": 9}

    def test_result_metadata_preserved(self, tmp_path: Path) -> None:
        cache = Cache(directory=tmp_path)
        prov = _make_provenance(engine="Gnina")
        result = _make_docking_result(engine="Gnina")
        result.metadata["cnn_scoring"] = "rescore"

        cache.put(prov, result, "docking_result")
        restored = cache.get(prov, "docking_result")

        assert restored.metadata["center"] == [10.0, 5.0, -2.0]
        assert restored.metadata["cnn_scoring"] == "rescore"

    def test_none_in_pose_metadata(self, tmp_path: Path) -> None:
        """Gnina-style poses carry None scores for absent CNN keys;
        None must round-trip as None, not the string 'None'."""
        cache = Cache(directory=tmp_path)
        prov = _make_provenance(engine="Gnina")
        result = DockingResult(
            poses=[
                Pose(
                    ligand=_make_ligand("p0"),
                    score=0.85,
                    rank=0,
                    metadata={"cnn_score": 0.85, "cnn_affinity": None},
                )
            ],
            receptor=None,
            engine="Gnina",
        )

        cache.put(prov, result, "docking_result")
        restored = cache.get(prov, "docking_result")

        assert restored.poses[0].metadata["cnn_score"] == pytest.approx(0.85)
        assert restored.poses[0].metadata["cnn_affinity"] is None

    def test_arrays_per_pose_distinct(self, tmp_path: Path) -> None:
        """Per-pose numpy arrays in pose metadata survive and stay
        distinct across poses (no cross-pose array bleed)."""
        cache = Cache(directory=tmp_path)
        prov = _make_provenance(engine="Vina")
        result = DockingResult(
            poses=[
                Pose(
                    ligand=_make_ligand("p0"),
                    score=-8.4,
                    rank=0,
                    metadata={"per_atom": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
                ),
                Pose(
                    ligand=_make_ligand("p1"),
                    score=-7.9,
                    rank=1,
                    metadata={"per_atom": np.array([4.0, 5.0, 6.0], dtype=np.float32)},
                ),
            ],
            receptor=None,
            engine="Vina",
        )

        cache.put(prov, result, "docking_result")
        restored = cache.get(prov, "docking_result")

        np.testing.assert_array_equal(restored.poses[0].metadata["per_atom"], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(restored.poses[1].metadata["per_atom"], [4.0, 5.0, 6.0])

    def test_arrays_in_ligand_metadata(self, tmp_path: Path) -> None:
        """Arrays attached to a pose's *ligand* Protein metadata (not
        the Pose.metadata) also round-trip per-pose."""
        cache = Cache(directory=tmp_path)
        prov = _make_provenance(engine="Vina")
        lig0 = _make_ligand("p0")
        lig0.metadata["charges"] = np.array([-0.4, 0.1, 0.3], dtype=np.float32)
        lig1 = _make_ligand("p1")
        lig1.metadata["charges"] = np.array([0.2, -0.2, 0.0], dtype=np.float32)
        result = DockingResult(
            poses=[
                Pose(ligand=lig0, score=-8.4, rank=0),
                Pose(ligand=lig1, score=-7.9, rank=1),
            ],
            receptor=None,
            engine="Vina",
        )

        cache.put(prov, result, "docking_result")
        restored = cache.get(prov, "docking_result")

        np.testing.assert_allclose(
            restored.poses[0].ligand.metadata["charges"], [-0.4, 0.1, 0.3], rtol=1e-6
        )
        np.testing.assert_allclose(
            restored.poses[1].ligand.metadata["charges"], [0.2, -0.2, 0.0], rtol=1e-6
        )

    def test_corrupt_payload_is_miss(self, tmp_path: Path) -> None:
        """A docking entry with garbage payload.json is a miss, not a crash."""
        cache = Cache(directory=tmp_path)
        prov = _make_provenance(engine="Vina")
        result = _make_docking_result()
        cache.put(prov, result, "docking_result")

        (cache.path_for(prov) / "payload.json").write_text("{not json", encoding="utf-8")
        assert cache.get(prov, "docking_result") is None


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


class TestFreeEnergyResultRoundTrip:
    def test_basic_result(self, tmp_path: Path) -> None:
        cache = Cache(directory=tmp_path)
        prov = _make_provenance(engine="AmberMMGBSA.run")
        result = FreeEnergyResult(delta_g=-21.0, uncertainty=0.7, method="MM/GBSA")

        assert cache.get(prov, "free_energy_result") is None  # miss
        cache.put(prov, result, "free_energy_result")
        restored = cache.get(prov, "free_energy_result")

        assert restored is not None
        assert restored.delta_g == pytest.approx(-21.0)
        assert restored.uncertainty == pytest.approx(0.7)
        assert restored.method == "MM/GBSA"
        assert restored.decomposition is None  # not requested

    def test_decomposition_round_trip(self, tmp_path: Path) -> None:
        cache = Cache(directory=tmp_path)
        prov = _make_provenance(engine="AmberMMGBSA.run")
        result = FreeEnergyResult(
            delta_g=-21.0,
            uncertainty=0.7,
            method="MM/GBSA",
            decomposition=Decomposition(
                [
                    ResidueContribution("LEU 40", -6.5, 0.3, 1.0, -6.0, -3.0, 2.0, -0.5),
                    ResidueContribution("LIG 241", -7.3, 0.5, 2.0, -8.0, -2.0, 1.0, -0.3),
                ]
            ),
        )
        cache.put(prov, result, "free_energy_result")
        d = cache.get(prov, "free_energy_result").decomposition

        assert d is not None
        assert list(d) == ["LEU 40", "LIG 241"]  # order preserved
        leu = d["LEU 40"]
        assert leu.total == pytest.approx(-6.5)
        assert leu.uncertainty == pytest.approx(0.3)
        assert leu.vdw == pytest.approx(-6.0)
        assert leu.electrostatic == pytest.approx(-3.0)
        assert leu.polar_solvation == pytest.approx(2.0)
        assert leu.nonpolar_solvation == pytest.approx(-0.5)
        assert leu.internal == pytest.approx(1.0)
        assert [c.residue for c in d.hotspots()] == ["LIG 241", "LEU 40"]

    def test_components_entropy_none_preserved(self, tmp_path: Path) -> None:
        cache = Cache(directory=tmp_path)
        prov = _make_provenance(engine="AmberMMGBSA.run")
        result = FreeEnergyResult(
            delta_g=-21.0,
            uncertainty=0.7,
            method="MM/GBSA",
            components=FreeEnergyComponents(-45.0, -30.0, 60.0, -6.0, entropy=None),
        )
        cache.put(prov, result, "free_energy_result")
        c = cache.get(prov, "free_energy_result").components

        assert c is not None
        assert c.vdw == pytest.approx(-45.0)
        assert c.entropy is None  # unknown, not zero
        assert c.enthalpy == pytest.approx(-21.0)

    def test_entropy_value_preserved(self, tmp_path: Path) -> None:
        cache = Cache(directory=tmp_path)
        prov = _make_provenance(engine="AmberMMGBSA.run")
        result = FreeEnergyResult(
            delta_g=-9.0,
            uncertainty=0.5,
            method="MM/GBSA",
            components=FreeEnergyComponents(-1.0, -1.0, -1.0, -1.0, entropy=12.5),
        )
        cache.put(prov, result, "free_energy_result")
        assert cache.get(prov, "free_energy_result").components.entropy == pytest.approx(12.5)

    def test_convergence_array(self, tmp_path: Path) -> None:
        cache = Cache(directory=tmp_path)
        prov = _make_provenance(engine="AmberMMGBSA.run")
        trace = np.array([-5.0, -15.0, -21.0])
        result = FreeEnergyResult(
            delta_g=-21.0, uncertainty=0.7, method="MM/GBSA", convergence=trace
        )
        cache.put(prov, result, "free_energy_result")
        restored = cache.get(prov, "free_energy_result")

        assert restored.convergence is not None
        np.testing.assert_allclose(restored.convergence, trace)

    def test_provenance_with_parent(self, tmp_path: Path) -> None:
        cache = Cache(directory=tmp_path)
        parent = _make_provenance(engine="AMBER.run")
        prov = _make_provenance(engine="AmberMMGBSA.run", parent=parent)
        result = FreeEnergyResult(delta_g=-21.0, uncertainty=0.7, method="MM/GBSA", provenance=prov)
        cache.put(prov, result, "free_energy_result")
        restored = cache.get(prov, "free_energy_result")

        assert restored.provenance is not None
        assert restored.provenance.engine == "AmberMMGBSA.run"
        assert restored.provenance.parent is not None
        assert restored.provenance.parent.engine == "AMBER.run"

    def test_metadata_with_provenance_and_arrays(self, tmp_path: Path) -> None:
        cache = Cache(directory=tmp_path)
        prov = _make_provenance(engine="AmberMMGBSA.run")
        result = FreeEnergyResult(
            delta_g=-21.0,
            uncertainty=0.7,
            method="MM/GBSA",
            metadata={
                "receptor_mask": ":1-3",
                "ligand_mask": ":4",
                mk.PROVENANCE: prov,
                "per_frame": np.arange(3),
            },
        )
        cache.put(prov, result, "free_energy_result")
        meta = cache.get(prov, "free_energy_result").metadata

        assert meta["receptor_mask"] == ":1-3"
        assert meta["ligand_mask"] == ":4"
        assert isinstance(meta[mk.PROVENANCE], Provenance)  # rebuilt, not a dict
        np.testing.assert_array_equal(meta["per_frame"], np.arange(3))

    def test_minimal_result_no_arrays_file(self, tmp_path: Path) -> None:
        # No components, convergence, or array metadata -> only payload.json.
        cache = Cache(directory=tmp_path)
        prov = _make_provenance(engine="AmberMMGBSA.run")
        cache.put(
            prov,
            FreeEnergyResult(delta_g=-9.0, uncertainty=0.0, method="MM/PBSA"),
            "free_energy_result",
        )
        entry = cache.path_for(prov)
        assert (entry / "payload.json").is_file()
        assert not (entry / "arrays.npz").is_file()
        assert cache.get(prov, "free_energy_result").components is None

    def test_registered(self) -> None:
        from molforge.cache import _SERIALIZERS

        assert "free_energy_result" in _SERIALIZERS
