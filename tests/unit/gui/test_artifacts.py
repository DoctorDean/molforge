"""Writing a step's output down, and summarising it for the panel.

The result types are built as light stand-ins rather than by running
engines: what is under test is the dispatch and the shape of the view,
and a test that needed Vina installed would simply not run anywhere.
Real `Protein` objects are used where the writing path is the point,
because `save()` is the part that can actually fail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from molforge.gui.artifacts import MAX_ENSEMBLE, ResultView, RunArtifacts, summarize

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------
# Stand-ins, duck-typed exactly as the predicates in artifacts.py expect
# ---------------------------------------------------------------------


@dataclass
class FakePose:
    ligand: Any
    score: float
    rank: int = 0
    rmsd_lb: float | None = None
    rmsd_ub: float | None = None


@dataclass
class FakeDocking:
    poses: list[FakePose]
    receptor: Any = None
    engine: str = "FakeDock"


@dataclass
class FakePocket:
    center: Any
    druggability: float | None = None
    score: float | None = None
    volume: float | None = None
    rank: int = 0
    residues: list[tuple[str, int, str]] = field(default_factory=list)


@dataclass
class FakeDesign:
    sequence: str
    score: float | None = None
    recovery: float | None = None


@dataclass
class FakeHit:
    identifier: str = ""
    source_database: str = ""
    pairing_key: str = ""


@dataclass
class FakeAlignment:
    depth: int
    hits: list[FakeHit]
    n_columns: int = 100
    query_sequence: str = "MKTV"


#: Two residues is enough to exercise save(): the writing path is what
#: these tests are about, and a fixture that downloaded 1CRN would make
#: a unit test depend on RCSB being up.
_PDB = (
    "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N  \n"
    "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C  \n"
    "ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 20.00           C  \n"
    "ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00 20.00           O  \n"
    "ATOM      5  N   GLY A   2       3.332   1.552   0.000  1.00 20.00           N  \n"
    "ATOM      6  CA  GLY A   2       3.999   2.845   0.000  1.00 20.00           C  \n"
    "ATOM      7  C   GLY A   2       5.509   2.690   0.000  1.00 20.00           C  \n"
    "ATOM      8  O   GLY A   2       6.016   1.570   0.000  1.00 20.00           O  \n"
    "END\n"
)


@pytest.fixture
def protein(tmp_path: Path):
    """A real Protein, built offline."""
    from molforge.io import load

    path = tmp_path / "tiny.pdb"
    path.write_text(_PDB)
    return load(path)


@pytest.fixture
def store(tmp_path: Path) -> RunArtifacts:
    return RunArtifacts(tmp_path / "run")


# ---------------------------------------------------------------------


class TestSummarize:
    def test_a_structure_offers_itself_to_the_viewer(self, protein) -> None:
        view = summarize(protein)
        assert view.kind == "structure"
        assert [s.key for s in view.structures] == ["structure"]
        assert view.structures[0].role == "receptor"
        assert view.detail["residues"] == protein.n_residues

    def test_docking_lists_the_receptor_and_every_pose(self, protein) -> None:
        """Both are needed: a pose shown without its receptor is a ligand
        floating in space, which says nothing about whether it bound."""
        result = FakeDocking(
            receptor=protein,
            poses=[FakePose(ligand=protein, score=-9.1 - i, rank=i) for i in range(3)],
        )
        view = summarize(result)
        assert view.kind == "poses"
        assert [s.key for s in view.structures] == [
            "receptor",
            "pose-1",
            "pose-2",
            "pose-3",
        ]
        assert view.structures[0].role == "receptor"
        assert {s.role for s in view.structures[1:]} == {"ligand"}

    def test_poses_table_leads_with_the_score(self) -> None:
        result = FakeDocking(poses=[FakePose(ligand=None, score=-9.123456, rank=0)])
        view = summarize(result)
        assert view.columns[:2] == ["rank", "score"]
        assert view.rows[0][1] == -9.123  # rounded for a table, not for maths

    def test_a_docking_run_with_no_receptor_still_shows_its_poses(self) -> None:
        view = summarize(FakeDocking(poses=[FakePose(ligand=None, score=-7.0)]))
        assert [s.key for s in view.structures] == ["pose-1"]

    def test_an_ensemble_is_capped_and_says_so(self, protein) -> None:
        """A diffusion run can return hundreds; a viewer shows a handful,
        and the manifest records how to reproduce the rest."""
        view = summarize([protein] * (MAX_ENSEMBLE + 5))
        assert view.kind == "ensemble"
        assert len(view.structures) == MAX_ENSEMBLE
        assert view.truncated
        assert view.detail["models"] == MAX_ENSEMBLE + 5

    def test_pockets_are_drawn_on_the_protein_they_were_found_in(self, protein) -> None:
        """Without it a pocket is a sphere in empty space."""
        pockets = [FakePocket(center=np.array([1.0, 2.0, 3.0], dtype=np.float32), score=0.8)]
        view = summarize(pockets, upstream=protein)
        assert view.kind == "pockets"
        assert [s.key for s in view.structures] == ["protein"]
        assert view.detail["centers"] == [[1.0, 2.0, 3.0]]

    def test_pockets_without_an_upstream_structure_still_tabulate(self) -> None:
        view = summarize([FakePocket(center=np.zeros(3, dtype=np.float32))])
        assert view.kind == "pockets"
        assert view.structures == []
        assert view.rows

    def test_designs_tabulate_their_sequences(self) -> None:
        view = summarize([FakeDesign("MKTV", score=-1.2, recovery=0.44)])
        assert view.kind == "designs"
        assert view.rows[0][-1] == "MKTV"
        assert view.rows[0][3] == 4  # length

    def test_an_alignment_reports_its_depth(self) -> None:
        """Depth is the number that decides the fold that consumes it."""
        view = summarize(FakeAlignment(depth=1024, hits=[FakeHit("h1", "uniref90", "ECOLI")]))
        assert view.kind == "alignment"
        assert view.detail["depth"] == 1024
        assert view.rows[0][1:] == ["h1", "uniref90", "ECOLI"]

    def test_nothing_is_nothing(self) -> None:
        assert summarize(None).kind == "none"

    def test_an_unrecognised_output_still_describes_itself(self) -> None:
        view = summarize(42)
        assert view.kind == "value"
        assert view.summary == "42"

    def test_it_serializes(self, protein) -> None:
        import json

        json.dumps(summarize(protein).to_dict())


class TestWriting:
    def test_a_structure_is_written_as_pdb(self, store: RunArtifacts, protein) -> None:
        view = store.store(0, protein)
        assert not view.error
        text = store.read(0, "structure")
        assert text.startswith(("HEADER", "ATOM", "CRYST", "REMARK", "TITLE"))
        assert "ATOM" in text

    def test_each_step_writes_to_its_own_directory(self, store: RunArtifacts, protein) -> None:
        store.store(0, protein)
        store.store(1, protein)
        assert store.step_dir(0) != store.step_dir(1)
        assert store.read(1, "structure")

    def test_poses_and_receptor_are_written_separately(
        self, store: RunArtifacts, protein
    ) -> None:
        """Separate files because the viewer styles them differently —
        cartoon for the receptor, sticks for the ligand."""
        result = FakeDocking(receptor=protein, poses=[FakePose(ligand=protein, score=-8.0)])
        view = store.store(0, result)
        assert {s.key for s in view.structures} == {"receptor", "pose-1"}
        assert store.read(0, "receptor")
        assert store.read(0, "pose-1")

    def test_an_output_with_no_structure_writes_nothing(self, store: RunArtifacts) -> None:
        view = store.store(0, [FakeDesign("MKTV")])
        assert view.structures == []
        assert not store.step_dir(0).exists()

    def test_a_failure_to_write_does_not_fail_the_run(self, store: RunArtifacts) -> None:
        """A pipeline that worked must not be reported as broken because
        its output couldn't be previewed."""

        class Unwritable:
            n_residues = 10
            atom_array = object()  # looks like a Protein, won't survive save()

        view = store.store(0, Unwritable())
        assert view.kind == "structure"
        assert view.structures == []  # dropped, not crashed
        assert view.detail["unwritten"]

    def test_a_summary_failure_is_reported_not_raised(
        self, store: RunArtifacts, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import molforge.gui.artifacts as module

        def explode(*_a: object, **_k: object) -> ResultView:
            raise RuntimeError("boom")

        monkeypatch.setattr(module, "summarize", explode)
        view = store.store(0, object())
        assert view.kind == "none"
        assert "boom" in view.error


class TestKeySafety:
    @pytest.mark.parametrize("key", ["../etc/passwd", "a/b", "a\\b", "", ".."])
    def test_a_key_that_escapes_the_run_directory_is_refused(
        self, store: RunArtifacts, key: str
    ) -> None:
        """Keys arrive from a URL. Joined unchecked, one with a separator
        addresses any file the server process can read."""
        with pytest.raises(ValueError, match="unsafe artifact key"):
            store.path(0, key)

    def test_an_ordinary_key_resolves_inside_the_run(self, store: RunArtifacts) -> None:
        path = store.path(0, "pose-1")
        assert store.root in path.parents


class TestPruning:
    def test_it_keeps_the_most_recent_runs(self, tmp_path: Path) -> None:
        """A dashboard left open for a month must not quietly fill a disk."""
        import os
        import time

        base = tmp_path / "runs"
        for i in range(6):
            run = base / f"run-{i}"
            run.mkdir(parents=True)
            (run / "f.txt").write_text("x")
            stamp = time.time() + i
            os.utime(run, (stamp, stamp))

        removed = RunArtifacts.prune(base=base, keep=2)
        assert removed == 4
        assert sorted(d.name for d in base.iterdir()) == ["run-4", "run-5"]

    def test_pruning_nothing_is_not_an_error(self, tmp_path: Path) -> None:
        assert RunArtifacts.prune(base=tmp_path / "absent", keep=5) == 0

    def test_it_shares_molforge_s_cache_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One setting for 'where molforge puts big things', not two."""
        monkeypatch.setenv("MOLFORGE_CACHE_DIR", str(tmp_path))
        assert tmp_path in RunArtifacts.base_dir().parents
