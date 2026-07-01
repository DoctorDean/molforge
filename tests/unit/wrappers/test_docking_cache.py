"""Cache-wiring tests for the docking engines (Vina, Gnina, DiffDock).

Each docking engine builds a :class:`Provenance` upfront from its
inputs + constructor parameters, consults the process-wide default
cache *before* requiring its binary or spawning any subprocess, and
stores the result on success under the ``"docking_result"`` type tag.

These tests verify the wiring without any docking binary installed:

- **Read path (hit):** a pre-populated cache entry is returned by
  ``dock()`` without the engine running at all — the compute seam is
  monkeypatched to raise, so reaching it fails the test.
- **Write + read (round-trip):** for the subprocess-driven engines
  (Gnina, DiffDock) a first ``dock()`` runs the (mocked) subprocess
  and a second identical ``dock()`` is a hit that does *not* re-run
  it; the subprocess is invoked exactly once across both calls.
- **Source-inspection net:** every ``dock()`` references the default
  cache and the ``get``/``put`` calls, so a refactor that drops the
  caching is caught cheaply.

The autouse ``_isolate_cache`` fixture (tests/conftest.py) gives each
test its own temp default-cache directory, so ``cache.put`` here and
the ``cache.get`` inside ``dock()`` share one slot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from molforge.cache import get_default_cache
from molforge.core import AtomArray, Protein
from molforge.docking import DockingResult, Pose
from molforge.wrappers.docking import DiffDock, Gnina, Vina

from tests.unit.wrappers.test_diffdock import _SAMPLE_SDF
from tests.unit.wrappers.test_gnina import _REAL_GNINA_SDF

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _receptor(name: str = "receptor") -> Protein:
    return Protein(AtomArray(0), name=name)


def _ligand(name: str = "lig", n_atoms: int = 3) -> Protein:
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


def _canned_result(engine: str) -> DockingResult:
    """A recognisable result we can pre-seed the cache with."""
    return DockingResult(
        poses=[Pose(ligand=_ligand("cached_pose"), score=-12.345, rank=0)],
        receptor=None,
        engine=engine,
        metadata={"sentinel": "from-cache"},
    )


def _boom(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("engine compute should not run on a cache hit")


# ---------------------------------------------------------------------
# Read path: a pre-populated entry short-circuits dock()
# ---------------------------------------------------------------------


class TestVinaCacheHit:
    def test_dock_returns_cached_without_engine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = Vina()
        receptor = _receptor()
        ligand = "/tmp/ligand.pdbqt"
        kwargs: dict[str, Any] = {
            "center": (10.0, 5.0, -2.0),
            "box_size": (20.0, 20.0, 20.0),
            "exhaustiveness": 8,
            "n_poses": 9,
            "energy_range": 3.0,
            "min_rmsd": 1.0,
        }
        prov = engine._build_provenance(receptor, ligand, **kwargs)
        get_default_cache().put(prov, _canned_result("Vina"), "docking_result")

        # If the cache short-circuit fails, dock() will try to build a
        # Vina handle and this raises.
        monkeypatch.setattr(engine, "_make_vina_handle", _boom)

        result = engine.dock(receptor, ligand, **kwargs)
        assert result.engine == "Vina"
        assert len(result) == 1
        assert result.best.score == pytest.approx(-12.345)
        assert result.metadata["sentinel"] == "from-cache"

    def test_different_params_miss_then_reach_engine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A different exhaustiveness is a different key — the cached
        entry must NOT satisfy it, so dock() proceeds to the engine."""
        engine = Vina()
        receptor = _receptor()
        ligand = "/tmp/ligand.pdbqt"
        base: dict[str, Any] = {
            "center": (10.0, 5.0, -2.0),
            "box_size": (20.0, 20.0, 20.0),
            "exhaustiveness": 8,
            "n_poses": 9,
            "energy_range": 3.0,
            "min_rmsd": 1.0,
        }
        prov = engine._build_provenance(receptor, ligand, **base)
        get_default_cache().put(prov, _canned_result("Vina"), "docking_result")

        monkeypatch.setattr(engine, "_make_vina_handle", _boom)
        # Same call but exhaustiveness=16 → different key → miss → engine.
        with pytest.raises(AssertionError, match="should not run"):
            engine.dock(receptor, ligand, **{**base, "exhaustiveness": 16})


class TestGninaCacheHit:
    def test_dock_returns_cached_without_binary(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = Gnina()
        receptor = _receptor()
        ligand = "/tmp/ligand.sdf"
        kwargs: dict[str, Any] = {
            "center": (10.0, 5.0, -2.0),
            "box_size": (20.0, 20.0, 20.0),
            "exhaustiveness": 8,
            "n_poses": 9,
            "min_rmsd": 1.0,
        }
        prov = engine._build_provenance(receptor, ligand, **kwargs)
        get_default_cache().put(prov, _canned_result("Gnina"), "docking_result")

        # _require_gnina would raise (no binary) — but a hit returns
        # before reaching it, so reaching it at all fails differently.
        monkeypatch.setattr(engine, "_require_gnina", _boom)

        result = engine.dock(receptor, ligand, **kwargs)
        assert result.engine == "Gnina"
        assert result.best.score == pytest.approx(-12.345)
        assert result.metadata["sentinel"] == "from-cache"


class TestDiffDockCacheHit:
    def test_dock_returns_cached_without_install(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = DiffDock()
        receptor = _receptor()
        ligand = "CCO"
        prov = engine._build_provenance(receptor, ligand)
        get_default_cache().put(prov, _canned_result("DiffDock"), "docking_result")

        # Both the install resolver and the CLI runner must be skipped
        # on a hit; patch both to raise.
        monkeypatch.setattr(engine, "_resolve_repo", _boom)
        monkeypatch.setattr(engine, "_run_cli", _boom)

        result = engine.dock(receptor=receptor, ligand=ligand)
        assert result.engine == "DiffDock"
        assert result.best.score == pytest.approx(-12.345)
        assert result.metadata["sentinel"] == "from-cache"


# ---------------------------------------------------------------------
# Write + read: a real (mocked) dock() stores its result, second hits
# ---------------------------------------------------------------------


class TestGninaRoundTrip:
    @staticmethod
    def _out_path_from_cmd(cmd: list[str]) -> Path:
        return Path(cmd[cmd.index("--out") + 1])

    def test_second_dock_is_a_hit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = {"run": 0}

        def fake_run(cmd: list[str], **_kwargs: Any) -> Any:
            calls["run"] += 1
            self._out_path_from_cmd(cmd).write_text(_REAL_GNINA_SDF, encoding="utf-8")
            proc = type("P", (), {})()
            proc.returncode = 0
            proc.stdout = ""
            proc.stderr = ""
            return proc

        monkeypatch.setattr("molforge.wrappers.docking.gnina.shutil.which", lambda _: "/usr/bin/gnina")
        monkeypatch.setattr("molforge.wrappers.docking.gnina.subprocess.run", fake_run)

        engine = Gnina()
        args = (_ligand("rec_as_prot"), _ligand("lig_as_prot"))
        kwargs: dict[str, Any] = {"center": (10.0, 5.0, -2.0), "box_size": (20.0, 20.0, 20.0)}

        first = engine.dock(*args, **kwargs)
        assert calls["run"] == 1
        second = engine.dock(*args, **kwargs)
        assert calls["run"] == 1  # served from cache, gnina not re-run

        assert len(second) == len(first)
        assert second.best.score == pytest.approx(first.best.score)
        assert [p.rank for p in second] == [p.rank for p in first]


class TestDiffDockRoundTrip:
    @staticmethod
    def _fake_install(tmp_path: Path) -> Path:
        repo = tmp_path / "DiffDock"
        repo.mkdir()
        (repo / "inference.py").write_text("# placeholder\n")
        return repo

    @staticmethod
    def _receptor_pdb(tmp_path: Path) -> Path:
        pdb = tmp_path / "receptor.pdb"
        pdb.write_text(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n"
        )
        return pdb

    def test_second_dock_is_a_hit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = self._fake_install(tmp_path)
        receptor = self._receptor_pdb(tmp_path)
        calls = {"run": 0}

        def fake_run(cmd: list[str], **_kwargs: Any) -> None:
            calls["run"] += 1
            out_dir = Path(cmd[cmd.index("--out_dir") + 1]) / "complex0"
            out_dir.mkdir(parents=True)
            (out_dir / "rank1_confidence0.80.sdf").write_text(_SAMPLE_SDF)
            return None

        monkeypatch.setattr("subprocess.run", fake_run)

        engine = DiffDock(repo_dir=repo)
        first = engine.dock(receptor=receptor, ligand="CCO")
        assert calls["run"] == 1
        second = engine.dock(receptor=receptor, ligand="CCO")
        assert calls["run"] == 1  # cache hit, DiffDock not re-run

        assert len(second) == len(first)
        assert second.best.metadata["confidence"] == pytest.approx(
            first.best.metadata["confidence"]
        )


# ---------------------------------------------------------------------
# Source-inspection net
# ---------------------------------------------------------------------


class TestSourceInspection:
    """Lock in that every docking dock() actually consults the cache."""

    @pytest.mark.parametrize(
        "module_path",
        [
            "molforge.wrappers.docking.vina",
            "molforge.wrappers.docking.gnina",
            "molforge.wrappers.docking.diffdock",
        ],
    )
    def test_dock_consults_cache(self, module_path: str) -> None:
        import importlib

        module = importlib.import_module(module_path)
        text = Path(module.__file__).read_text(encoding="utf-8")  # type: ignore[arg-type]
        assert "get_default_cache()" in text
        assert 'cache.get(provenance, "docking_result")' in text
        assert 'cache.put(provenance, result, "docking_result")' in text
        # The cache lookup must come before the put (read-before-write).
        assert text.index("cache.get(provenance") < text.index("cache.put(provenance")
