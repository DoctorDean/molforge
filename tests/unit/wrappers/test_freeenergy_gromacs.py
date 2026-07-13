"""Tests for the gmx_MMPBSA results parser.

gmx_MMPBSA writes the same file structure as MMPBSA.py but with
Δ-prefixed delta rows and five numeric columns; these check that the
shared helpers read column 0 (ΔG) and column -1 (SEM) correctly and that
the Δ-labels don't collide (ΔVDWAALS vs Δ1-4 VDW).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molforge.cache import CACHE_DIR_ENV, _reset_default_cache_for_testing
from molforge.core import AtomArray, Protein, Provenance
from molforge.core import metadata_keys as mk
from molforge.freeenergy import MMGBSAEngineNotInstalledError
from molforge.md import Trajectory
from molforge.wrappers.freeenergy import (
    GromacsMMGBSA,
    parse_gmx_mmpbsa_dat,
    selection_to_ndx_group,
)

FIXTURE = (
    Path(__file__).resolve().parents[2] / "fixtures" / "freeenergy" / "gmx_FINAL_RESULTS_MMPBSA.dat"
)


@pytest.fixture
def dat() -> str:
    return FIXTURE.read_text()


class TestGeneralizedBorn:
    def test_delta_g_and_uncertainty(self, dat: str) -> None:
        r = parse_gmx_mmpbsa_dat(dat)  # gb default
        assert r.method == "MM/GBSA"
        assert r.delta_g == pytest.approx(-21.0)
        assert r.uncertainty == pytest.approx(0.7)  # last column (SEM), not SD

    def test_components(self, dat: str) -> None:
        c = parse_gmx_mmpbsa_dat(dat).components
        assert c is not None
        assert c.vdw == pytest.approx(-45.0)
        assert c.electrostatic == pytest.approx(-30.0)
        assert c.polar_solvation == pytest.approx(60.0)  # ΔEGB
        assert c.nonpolar_solvation == pytest.approx(-6.0)  # ΔESURF
        assert c.entropy is None

    def test_enthalpy_reconstructs_total(self, dat: str) -> None:
        r = parse_gmx_mmpbsa_dat(dat)
        assert r.components.enthalpy == pytest.approx(r.delta_g)

    def test_metadata(self, dat: str) -> None:
        r = parse_gmx_mmpbsa_dat(dat)
        assert r.metadata["solvent_model"] == "gb"
        assert r.metadata["n_frames"] == 16
        assert r.metadata["delta_total_std_dev"] == pytest.approx(7.0)  # sample SD column


class TestPoissonBoltzmann:
    def test_selects_pb_section(self, dat: str) -> None:
        r = parse_gmx_mmpbsa_dat(dat, solvent_model="pb")
        assert r.method == "MM/PBSA"
        assert r.delta_g == pytest.approx(-24.0)
        assert r.uncertainty == pytest.approx(0.72)

    def test_nonpolar_sums_enpolar_and_edisper(self, dat: str) -> None:
        c = parse_gmx_mmpbsa_dat(dat, solvent_model="pb").components
        assert c is not None
        assert c.polar_solvation == pytest.approx(55.0)  # ΔEPB
        assert c.nonpolar_solvation == pytest.approx(-4.0)  # -8 + 4
        assert c.enthalpy == pytest.approx(-24.0)


class TestRobustness:
    def test_delta_label_anchoring(self, dat: str) -> None:
        # ΔVDWAALS must not pick up "Δ1-4 VDW", nor ΔEEL "Δ1-4 EEL".
        c = parse_gmx_mmpbsa_dat(dat).components
        assert c.vdw == pytest.approx(-45.0)
        assert c.electrostatic == pytest.approx(-30.0)

    def test_reads_delta_not_complex_block(self, dat: str) -> None:
        # The Complex block has VDWAALS -900 (no Δ); must be ignored.
        assert parse_gmx_mmpbsa_dat(dat).components.vdw == pytest.approx(-45.0)

    def test_unknown_model_raises(self, dat: str) -> None:
        with pytest.raises(ValueError, match="'gb' or 'pb'"):
            parse_gmx_mmpbsa_dat(dat, solvent_model="rism")

    def test_missing_section_raises(self, dat: str) -> None:
        gb_only = dat.split("POISSON BOLTZMANN:")[0]
        with pytest.raises(ValueError, match="POISSON BOLTZMANN"):
            parse_gmx_mmpbsa_dat(gb_only, solvent_model="pb")

    def test_missing_row_raises(self) -> None:
        text = "GENERALIZED BORN:\n\nDelta (Complex - Receptor - Ligand):\nΔVDWAALS 1 2 3 4 5\n"
        with pytest.raises(ValueError, match="ΔEEL"):
            parse_gmx_mmpbsa_dat(text)


def _complex(n_lig_atoms: int = 2) -> Protein:
    spec = [
        (["N", "CA", "C"], "A", 1, "ALA", "protein"),
        (["N", "CA", "C"], "A", 2, "GLY", "protein"),
        (["N", "CA", "C"], "A", 3, "LEU", "protein"),
        ([f"L{i}" for i in range(n_lig_atoms)], "B", 1, "LIG", "ligand"),
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
    arr.element[:] = ["C"] * n
    return Protein(arr, name="cplx")


class TestNdxGroup:
    def test_receptor_group(self) -> None:
        # 3 protein residues x 3 atoms = atoms 1..9 (1-based).
        block = selection_to_ndx_group(_complex(), {"entity_type": "protein"}, "receptor")
        assert block == "[ receptor ]\n1 2 3 4 5 6 7 8 9\n"

    def test_ligand_group(self) -> None:
        block = selection_to_ndx_group(_complex(), {"entity_type": "ligand"}, "ligand")
        assert block == "[ ligand ]\n10 11\n"

    def test_boolean_mask_input(self) -> None:
        cplx = _complex()
        mask = cplx.atom_array.entity_type == "ligand"
        assert selection_to_ndx_group(cplx, mask, "ligand") == "[ ligand ]\n10 11\n"

    def test_wraps_long_groups(self) -> None:
        # 9 protein + 20 ligand atoms; ligand indices 10..29 wrap at 15.
        block = selection_to_ndx_group(
            _complex(n_lig_atoms=20), {"entity_type": "ligand"}, "lig", per_line=15
        )
        lines = block.splitlines()
        assert lines[0] == "[ lig ]"
        assert lines[1] == "10 11 12 13 14 15 16 17 18 19 20 21 22 23 24"
        assert lines[2] == "25 26 27 28 29"

    def test_empty_selection_raises(self) -> None:
        with pytest.raises(ValueError, match="matches no atoms"):
            selection_to_ndx_group(_complex(), {"residue_name": "ZZZ"}, "x")


# ---------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the default cache at a per-test dir so run() never touches
    the real one and tests don't cross-contaminate."""
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path / "cache"))
    _reset_default_cache_for_testing()


def _trajectory(metadata: dict | None = None, n_frames: int = 5) -> Trajectory:
    top = _complex()
    coords = np.zeros((n_frames, top.n_atoms, 3), dtype=np.float32)
    return Trajectory(topology=top, coordinates=coords, metadata=metadata or {})


def _inputs(tmp_path: Path, *, with_top: bool = True) -> dict:
    (tmp_path / "md.tpr").write_text("tpr")
    (tmp_path / "md.xtc").write_text("xtc")
    meta = {"structure": str(tmp_path / "md.tpr"), "trajectory_file": str(tmp_path / "md.xtc")}
    if with_top:
        (tmp_path / "topol.top").write_text("top")
        meta["topology"] = str(tmp_path / "topol.top")
    return meta


RECEPTOR = {"entity_type": "protein"}
LIGAND = {"entity_type": "ligand"}


def _install_stub(engine: GromacsMMGBSA, results_text: str) -> dict:
    record: dict = {"calls": [], "ndx": None, "mmpbsa_in": None}

    def fake_run(cmd, *, cwd, step):
        record["calls"].append((step, list(cmd)))
        cwd = Path(cwd)
        record["ndx"] = (cwd / "index.ndx").read_text()
        record["mmpbsa_in"] = (cwd / "mmpbsa.in").read_text()
        (cwd / "FINAL_RESULTS_MMPBSA.dat").write_text(results_text)

    engine._require_tool = lambda: None  # type: ignore[method-assign]
    engine._run_subprocess = fake_run  # type: ignore[assignment]
    return record


class TestGromacsResolveInputs:
    def test_from_run_dir_metadata(self, tmp_path: Path) -> None:
        (tmp_path / "md.tpr").write_text("s")
        (tmp_path / "md.xtc").write_text("t")
        (tmp_path / "topol.top").write_text("p")
        traj = _trajectory({"run_dir": str(tmp_path)})
        s, t, top = GromacsMMGBSA()._resolve_inputs(traj, None, None, None)
        assert s.name == "md.tpr" and t.name == "md.xtc" and top.name == "topol.top"

    def test_explicit_paths(self, tmp_path: Path) -> None:
        meta = _inputs(tmp_path)
        s, t, top = GromacsMMGBSA()._resolve_inputs(
            _trajectory(), meta["structure"], meta["trajectory_file"], meta["topology"]
        )
        assert (s.name, t.name, top.name) == ("md.tpr", "md.xtc", "topol.top")

    def test_topology_optional(self, tmp_path: Path) -> None:
        meta = _inputs(tmp_path, with_top=False)
        _, _, top = GromacsMMGBSA()._resolve_inputs(_trajectory(meta), None, None, None)
        assert top is None  # no topol.top -> -cp omitted

    def test_missing_structure_raises(self) -> None:
        with pytest.raises(ValueError, match="no GROMACS structure"):
            GromacsMMGBSA()._resolve_inputs(_trajectory(), None, "some.xtc", None)

    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            GromacsMMGBSA()._resolve_inputs(
                _trajectory(), tmp_path / "missing.tpr", tmp_path / "missing.xtc", None
            )


class TestGromacsNotInstalled:
    def test_run_without_tool_raises(self, tmp_path: Path) -> None:
        traj = _trajectory(_inputs(tmp_path))
        with pytest.raises(MMGBSAEngineNotInstalledError, match=r"gmx_MMPBSA|PATH"):
            GromacsMMGBSA().run(traj, receptor=RECEPTOR, ligand=LIGAND)

    def test_empty_selection_raises(self, tmp_path: Path) -> None:
        traj = _trajectory(_inputs(tmp_path))
        with pytest.raises(ValueError, match="ligand selection matches no atoms"):
            GromacsMMGBSA().run(traj, receptor=RECEPTOR, ligand={"residue_name": "ZZZ"})


class TestGromacsPipeline:
    def test_gb_end_to_end(self, tmp_path: Path) -> None:
        engine = GromacsMMGBSA()
        _install_stub(engine, FIXTURE.read_text())
        result = engine.run(_trajectory(_inputs(tmp_path)), receptor=RECEPTOR, ligand=LIGAND)
        assert result.method == "MM/GBSA"
        assert result.delta_g == pytest.approx(-21.0)
        assert result.uncertainty == pytest.approx(0.7)
        assert result.metadata["receptor_natoms"] == 9
        assert result.metadata["ligand_natoms"] == 2

    def test_pb_selected(self, tmp_path: Path) -> None:
        engine = GromacsMMGBSA()
        _install_stub(engine, FIXTURE.read_text())
        result = engine.run(
            _trajectory(_inputs(tmp_path)), receptor=RECEPTOR, ligand=LIGAND, solvent_model="pb"
        )
        assert result.method == "MM/PBSA"
        assert result.delta_g == pytest.approx(-24.0)

    def test_command_and_written_inputs(self, tmp_path: Path) -> None:
        engine = GromacsMMGBSA()
        record = _install_stub(engine, FIXTURE.read_text())
        engine.run(_trajectory(_inputs(tmp_path)), receptor=RECEPTOR, ligand=LIGAND)

        (step, cmd) = record["calls"][0]
        assert step == "gmx_MMPBSA"
        for flag in ("-cs", "-ci", "-cg", "-ct", "-i", "-cp", "-nogui"):
            assert flag in cmd
        cg = cmd.index("-cg")
        assert cmd[cg + 1 : cg + 3] == ["0", "1"]  # receptor group 0, ligand group 1
        # index.ndx has both groups; receptor first (9 atoms), ligand (2).
        assert record["ndx"] == "[ receptor ]\n1 2 3 4 5 6 7 8 9\n[ ligand ]\n10 11\n"
        assert "&gb" in record["mmpbsa_in"]

    def test_topology_omitted_drops_cp(self, tmp_path: Path) -> None:
        engine = GromacsMMGBSA()
        record = _install_stub(engine, FIXTURE.read_text())
        engine.run(_trajectory(_inputs(tmp_path, with_top=False)), receptor=RECEPTOR, ligand=LIGAND)
        assert "-cp" not in record["calls"][0][1]

    def test_provenance_attached_with_parent(self, tmp_path: Path) -> None:
        parent = Provenance.from_engine(engine="GROMACS.run")
        engine = GromacsMMGBSA()
        _install_stub(engine, FIXTURE.read_text())
        result = engine.run(
            _trajectory({**_inputs(tmp_path), mk.PROVENANCE: parent}),
            receptor=RECEPTOR,
            ligand=LIGAND,
        )
        assert result.provenance is not None
        assert result.provenance.engine == "GromacsMMGBSA.run"
        assert result.provenance.parent is not None
        assert result.provenance.parent.engine == "GROMACS.run"


class TestGromacsCaching:
    def test_second_run_hits_cache_without_tool(self, tmp_path: Path) -> None:
        traj = _trajectory(_inputs(tmp_path))
        warm = GromacsMMGBSA()
        _install_stub(warm, FIXTURE.read_text())
        first = warm.run(traj, receptor=RECEPTOR, ligand=LIGAND)

        cold = GromacsMMGBSA()  # no stub; real _require_tool would raise
        second = cold.run(traj, receptor=RECEPTOR, ligand=LIGAND)
        assert second.delta_g == pytest.approx(first.delta_g)

    def test_cache_hit_skips_subprocess(self, tmp_path: Path) -> None:
        traj = _trajectory(_inputs(tmp_path))
        engine = GromacsMMGBSA()
        record = _install_stub(engine, FIXTURE.read_text())
        engine.run(traj, receptor=RECEPTOR, ligand=LIGAND)
        after_first = len(record["calls"])
        engine.run(traj, receptor=RECEPTOR, ligand=LIGAND)
        assert after_first == 1
        assert len(record["calls"]) == after_first  # no new tool call

    def test_different_selection_misses(self, tmp_path: Path) -> None:
        traj = _trajectory(_inputs(tmp_path))
        engine = GromacsMMGBSA()
        record = _install_stub(engine, FIXTURE.read_text())
        engine.run(traj, receptor=RECEPTOR, ligand=LIGAND)
        # Swap receptor/ligand -> different index groups -> different key.
        engine.run(traj, receptor=LIGAND, ligand=RECEPTOR)
        assert len(record["calls"]) == 2


DECOMP_FIXTURE = (
    Path(__file__).resolve().parents[2] / "fixtures" / "freeenergy" / "gmx_FINAL_DECOMP_MMPBSA.dat"
)


def _install_stub_with_decomp(engine: GromacsMMGBSA, results_text: str, decomp_text: str) -> dict:
    """Stub that also writes FINAL_DECOMP_MMPBSA.dat when -do is passed."""
    record: dict = {"mmpbsa_in": None, "wrote_decomp": False, "cmd": None}

    def fake_run(cmd, *, cwd, step):
        cwd = Path(cwd)
        record["cmd"] = list(cmd)
        record["mmpbsa_in"] = (cwd / "mmpbsa.in").read_text()
        (cwd / "FINAL_RESULTS_MMPBSA.dat").write_text(results_text)
        if "-do" in cmd:
            record["wrote_decomp"] = True
            (cwd / "FINAL_DECOMP_MMPBSA.dat").write_text(decomp_text)

    engine._require_tool = lambda: None  # type: ignore[method-assign]
    engine._run_subprocess = fake_run  # type: ignore[assignment]
    return record


class TestGromacsDecomposition:
    def test_run_with_idecomp_attaches_decomposition(self, tmp_path: Path) -> None:
        engine = GromacsMMGBSA()
        record = _install_stub_with_decomp(engine, FIXTURE.read_text(), DECOMP_FIXTURE.read_text())
        result = engine.run(
            _trajectory(_inputs(tmp_path)), receptor=RECEPTOR, ligand=LIGAND, idecomp=2
        )
        assert "&decomp" in record["mmpbsa_in"]
        assert "idecomp=2" in record["mmpbsa_in"]
        assert "-do" in record["cmd"]
        assert record["wrote_decomp"]
        assert result.decomposition is not None
        # Location column stripped, ligand keeps complex numbering
        assert list(result.decomposition) == ["LEU 40", "THR 41", "ALA 44", "RAL 241"]
        assert result.decomposition.hotspots(1)[0].residue == "RAL 241"

    def test_run_without_idecomp_has_no_decomposition(self, tmp_path: Path) -> None:
        engine = GromacsMMGBSA()
        record = _install_stub_with_decomp(engine, FIXTURE.read_text(), DECOMP_FIXTURE.read_text())
        result = engine.run(_trajectory(_inputs(tmp_path)), receptor=RECEPTOR, ligand=LIGAND)
        assert "&decomp" not in record["mmpbsa_in"]
        assert "-do" not in record["cmd"]
        assert not record["wrote_decomp"]
        assert result.decomposition is None
