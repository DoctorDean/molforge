"""Tests for the AmberMMGBSA engine.

No AmberTools in the sandbox, so the pipeline is exercised the way the
AMBER MD wrapper tests itself: by replacing the single ``_run_subprocess``
choke point with a stub that simulates the tools writing their outputs
(dummy split topologies, and the results ``.dat`` copied from the
fixture). The tool-detection and input-resolution seams are tested
directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molforge.cache import CACHE_DIR_ENV, _reset_default_cache_for_testing
from molforge.core import AtomArray, Protein, Provenance
from molforge.core import metadata_keys as mk
from molforge.md import Trajectory
from molforge.wrappers.freeenergy import AmberMMGBSA

FIXTURE = (
    Path(__file__).resolve().parents[2] / "fixtures" / "freeenergy" / "FINAL_RESULTS_MMPBSA.dat"
)


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the default cache at a per-test dir so run() never touches
    the real one and tests don't cross-contaminate."""
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path / "cache"))
    _reset_default_cache_for_testing()


def _topology() -> Protein:
    spec = [
        (["N", "CA", "C"], "A", 1, "ALA", "protein"),
        (["N", "CA", "C"], "A", 2, "GLY", "protein"),
        (["N", "CA", "C"], "A", 3, "LEU", "protein"),
        (["C1", "C2"], "B", 1, "LIG", "ligand"),
    ]
    rows = [(nm, ch, rid, rn, ent) for atoms, ch, rid, rn, ent in spec for nm in atoms]
    n = len(rows)
    arr = AtomArray(n)
    arr.coords[:] = np.zeros((n, 3), dtype=np.float32)
    arr.atom_name[:] = [r[0] for r in rows]
    arr.chain_id[:] = [r[1] for r in rows]
    arr.residue_id[:] = [r[2] for r in rows]
    arr.residue_name[:] = [r[3] for r in rows]
    arr.entity_type[:] = [r[4] for r in rows]
    arr.element[:] = [r[0][0] for r in rows]
    return Protein(arr, name="cplx")


def _trajectory(metadata: dict | None = None, n_frames: int = 5) -> Trajectory:
    top = _topology()
    coords = np.zeros((n_frames, top.n_atoms, 3), dtype=np.float32)
    return Trajectory(topology=top, coordinates=coords, metadata=metadata or {})


RECEPTOR = {"entity_type": "protein"}
LIGAND = {"entity_type": "ligand"}


def _install_stub(engine: AmberMMGBSA, results_text: str) -> dict:
    """Replace the tool seams; return a record of what happened."""
    record: dict = {"calls": [], "mmpbsa_in": None}

    def fake_run(cmd, *, cwd, step):  # noqa: ANN001
        record["calls"].append((step, list(cmd)))
        cwd = Path(cwd)
        if step == "ante-MMPBSA":
            for name in ("complex.prmtop", "receptor.prmtop", "ligand.prmtop"):
                (cwd / name).write_text("dummy topology")
        elif step == "MMPBSA":
            record["mmpbsa_in"] = (cwd / "mmpbsa.in").read_text()
            (cwd / "FINAL_RESULTS_MMPBSA.dat").write_text(results_text)

    engine._require_tools = lambda: None  # type: ignore[method-assign]
    engine._run_subprocess = fake_run  # type: ignore[assignment]
    return record


def _inputs(tmp_path: Path) -> dict:
    prmtop = tmp_path / "system.prmtop"
    traj = tmp_path / "prod.nc"
    prmtop.write_text("prmtop")
    traj.write_text("traj")
    return {"prmtop": str(prmtop), "trajectory_file": str(traj)}


class TestResolveInputs:
    def test_explicit_paths(self, tmp_path: Path) -> None:
        prmtop = tmp_path / "c.prmtop"
        traj = tmp_path / "t.nc"
        prmtop.write_text("x")
        traj.write_text("y")
        p, t = AmberMMGBSA()._resolve_amber_inputs(_trajectory(), prmtop, traj)
        assert (p, t) == (prmtop, traj)

    def test_from_run_dir_metadata(self, tmp_path: Path) -> None:
        (tmp_path / "system.prmtop").write_text("x")
        (tmp_path / "prod.nc").write_text("y")
        traj = _trajectory({"run_dir": str(tmp_path)})
        p, t = AmberMMGBSA()._resolve_amber_inputs(traj, None, None)
        assert p.name == "system.prmtop" and t.name == "prod.nc"

    def test_from_explicit_metadata_keys(self, tmp_path: Path) -> None:
        meta = _inputs(tmp_path)
        p, t = AmberMMGBSA()._resolve_amber_inputs(_trajectory(meta), None, None)
        assert p == Path(meta["prmtop"]) and t == Path(meta["trajectory_file"])

    def test_missing_prmtop_raises(self) -> None:
        with pytest.raises(ValueError, match="no Amber topology"):
            AmberMMGBSA()._resolve_amber_inputs(_trajectory(), None, "somewhere.nc")

    def test_missing_trajectory_raises(self, tmp_path: Path) -> None:
        prmtop = tmp_path / "c.prmtop"
        prmtop.write_text("x")
        with pytest.raises(ValueError, match="no trajectory file"):
            AmberMMGBSA()._resolve_amber_inputs(_trajectory(), prmtop, None)

    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            AmberMMGBSA()._resolve_amber_inputs(
                _trajectory(), tmp_path / "missing.prmtop", tmp_path / "missing.nc"
            )


class TestNotInstalled:
    def test_run_without_tools_raises(self, tmp_path: Path) -> None:
        # Valid inputs, but MMPBSA.py isn't on PATH in the sandbox.
        from molforge.freeenergy import MMGBSAEngineNotInstalledError

        traj = _trajectory(_inputs(tmp_path))
        with pytest.raises(MMGBSAEngineNotInstalledError, match="AmberTools|PATH"):
            AmberMMGBSA().run(traj, receptor=RECEPTOR, ligand=LIGAND)


class TestPipeline:
    def test_gb_end_to_end(self, tmp_path: Path) -> None:
        engine = AmberMMGBSA()
        record = _install_stub(engine, FIXTURE.read_text())
        result = engine.run(
            _trajectory(_inputs(tmp_path)), receptor=RECEPTOR, ligand=LIGAND
        )
        assert result.method == "MM/GBSA"
        assert result.delta_g == pytest.approx(-21.0)
        assert result.uncertainty == pytest.approx(0.7)
        assert result.metadata["receptor_mask"] == ":1-3"
        assert result.metadata["ligand_mask"] == ":4"

    def test_pb_selected(self, tmp_path: Path) -> None:
        engine = AmberMMGBSA()
        _install_stub(engine, FIXTURE.read_text())
        result = engine.run(
            _trajectory(_inputs(tmp_path)),
            receptor=RECEPTOR,
            ligand=LIGAND,
            solvent_model="pb",
        )
        assert result.method == "MM/PBSA"
        assert result.delta_g == pytest.approx(-24.0)

    def test_commands_and_input_file(self, tmp_path: Path) -> None:
        engine = AmberMMGBSA()
        record = _install_stub(engine, FIXTURE.read_text())
        engine.run(_trajectory(_inputs(tmp_path)), receptor=RECEPTOR, ligand=LIGAND)

        steps = {step: cmd for step, cmd in record["calls"]}
        assert set(steps) == {"ante-MMPBSA", "MMPBSA"}
        # ante-MMPBSA carries the resolved masks and strip mask.
        ante = steps["ante-MMPBSA"]
        assert ante[ante.index("-m") + 1] == ":1-3"
        assert ante[ante.index("-n") + 1] == ":4"
        assert "-s" in ante
        # MMPBSA.py gets the three split topologies and the input file.
        mmpbsa = steps["MMPBSA"]
        for flag in ("-cp", "-rp", "-lp", "-y", "-i"):
            assert flag in mmpbsa
        # mmpbsa.in was written before MMPBSA.py ran.
        assert "&gb" in record["mmpbsa_in"]

    def test_provenance_attached_with_parent(self, tmp_path: Path) -> None:
        parent = Provenance.from_engine(engine="AMBER.run")
        engine = AmberMMGBSA()
        _install_stub(engine, FIXTURE.read_text())
        result = engine.run(
            _trajectory({**_inputs(tmp_path), mk.PROVENANCE: parent}),
            receptor=RECEPTOR,
            ligand=LIGAND,
        )
        assert result.provenance is not None
        assert result.provenance.engine == "AmberMMGBSA.run"
        assert result.provenance.parameters["receptor_mask"] == ":1-3"
        assert result.provenance.parent is not None
        assert result.provenance.parent.engine == "AMBER.run"


class TestCaching:
    def test_second_run_hits_cache_without_tools(self, tmp_path: Path) -> None:
        # First run populates the cache (tools stubbed). Second run uses a
        # fresh engine with NO stub — real _require_tools would raise
        # NotInstalled — so returning a result proves the cache short-
        # circuits before the tools.
        traj = _trajectory(_inputs(tmp_path))

        warm = AmberMMGBSA()
        _install_stub(warm, FIXTURE.read_text())
        first = warm.run(traj, receptor=RECEPTOR, ligand=LIGAND)

        cold = AmberMMGBSA()  # tools absent in sandbox, no stub
        second = cold.run(traj, receptor=RECEPTOR, ligand=LIGAND)

        assert second.delta_g == pytest.approx(first.delta_g)
        assert second.method == first.method
        assert second.metadata["receptor_mask"] == ":1-3"

    def test_cache_hit_skips_subprocess(self, tmp_path: Path) -> None:
        traj = _trajectory(_inputs(tmp_path))
        engine = AmberMMGBSA()
        record = _install_stub(engine, FIXTURE.read_text())

        engine.run(traj, receptor=RECEPTOR, ligand=LIGAND)
        calls_after_first = len(record["calls"])
        engine.run(traj, receptor=RECEPTOR, ligand=LIGAND)  # identical -> hit

        assert calls_after_first > 0
        assert len(record["calls"]) == calls_after_first  # no new tool calls

    def test_different_params_miss(self, tmp_path: Path) -> None:
        # GB then PB over the same trajectory are distinct runs -> both
        # invoke the tools (two sets of subprocess calls, not one).
        traj = _trajectory(_inputs(tmp_path))
        engine = AmberMMGBSA()
        record = _install_stub(engine, FIXTURE.read_text())

        engine.run(traj, receptor=RECEPTOR, ligand=LIGAND, solvent_model="gb")
        after_gb = len(record["calls"])
        engine.run(traj, receptor=RECEPTOR, ligand=LIGAND, solvent_model="pb")

        assert len(record["calls"]) > after_gb  # PB was not served from GB's entry


DECOMP_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "freeenergy"
    / "FINAL_DECOMP_MMPBSA.dat"
)


def _install_stub_with_decomp(
    engine: AmberMMGBSA, results_text: str, decomp_text: str
) -> dict:
    """Stub that also writes FINAL_DECOMP_MMPBSA.dat when -do is passed."""
    record: dict = {"calls": [], "mmpbsa_in": None, "wrote_decomp": False}

    def fake_run(cmd, *, cwd, step):  # noqa: ANN001
        record["calls"].append((step, list(cmd)))
        cwd = Path(cwd)
        if step == "ante-MMPBSA":
            for name in ("complex.prmtop", "receptor.prmtop", "ligand.prmtop"):
                (cwd / name).write_text("dummy topology")
        elif step == "MMPBSA":
            record["mmpbsa_in"] = (cwd / "mmpbsa.in").read_text()
            (cwd / "FINAL_RESULTS_MMPBSA.dat").write_text(results_text)
            if "-do" in cmd:
                record["wrote_decomp"] = True
                (cwd / "FINAL_DECOMP_MMPBSA.dat").write_text(decomp_text)

    engine._require_tools = lambda: None  # type: ignore[method-assign]
    engine._run_subprocess = fake_run  # type: ignore[assignment]
    return record


class TestDecomposition:
    def test_run_with_idecomp_attaches_decomposition(self, tmp_path: Path) -> None:
        engine = AmberMMGBSA()
        record = _install_stub_with_decomp(
            engine, FIXTURE.read_text(), DECOMP_FIXTURE.read_text()
        )
        result = engine.run(
            _trajectory(_inputs(tmp_path)),
            receptor=RECEPTOR,
            ligand=LIGAND,
            idecomp=1,
        )
        # &decomp written, -do passed, decomp file consumed
        assert "&decomp" in record["mmpbsa_in"]
        assert "idecomp=1" in record["mmpbsa_in"]
        assert record["wrote_decomp"]
        # the DELTAS decomposition is attached and parsed
        assert result.decomposition is not None
        assert list(result.decomposition) == ["LEU 40", "THR 41", "ALA 44", "LIG 241"]
        assert result.decomposition["LIG 241"].total == pytest.approx(-7.3)
        assert result.decomposition.hotspots(1)[0].residue == "LIG 241"

    def test_run_without_idecomp_has_no_decomposition(self, tmp_path: Path) -> None:
        engine = AmberMMGBSA()
        record = _install_stub_with_decomp(
            engine, FIXTURE.read_text(), DECOMP_FIXTURE.read_text()
        )
        result = engine.run(_trajectory(_inputs(tmp_path)), receptor=RECEPTOR, ligand=LIGAND)
        assert "&decomp" not in record["mmpbsa_in"]
        assert not record["wrote_decomp"]
        assert result.decomposition is None

    def test_decomposition_survives_cache(self, tmp_path: Path) -> None:
        engine = AmberMMGBSA()
        _install_stub_with_decomp(engine, FIXTURE.read_text(), DECOMP_FIXTURE.read_text())
        traj = _trajectory(_inputs(tmp_path))
        first = engine.run(traj, receptor=RECEPTOR, ligand=LIGAND, idecomp=1)

        # second call hits cache (tools would fail if called)
        engine._run_subprocess = _fail_if_called  # type: ignore[assignment]
        second = engine.run(traj, receptor=RECEPTOR, ligand=LIGAND, idecomp=1)
        assert second.decomposition is not None
        assert list(second.decomposition) == list(first.decomposition)
        assert second.decomposition["LEU 40"].total == pytest.approx(-6.5)


def _fail_if_called(cmd, *, cwd, step):  # noqa: ANN001
    raise AssertionError("tools should not run on a cache hit")
