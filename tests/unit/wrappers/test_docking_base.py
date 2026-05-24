"""Tests for the docking ABC, Pose, and DockingResult."""

from __future__ import annotations

import pytest

from molforge.core import AtomArray, Protein
from molforge.docking import DockingEngine, DockingResult, Pose


def _empty_protein() -> Protein:
    return Protein(AtomArray(0))


class TestPose:
    def test_construction(self) -> None:
        p = Pose(ligand=_empty_protein(), score=-7.5)
        assert p.score == -7.5
        assert p.rank == 0
        assert p.rmsd_lb is None
        assert p.metadata == {}

    def test_with_extras(self) -> None:
        p = Pose(
            ligand=_empty_protein(),
            score=-7.5,
            rank=3,
            rmsd_lb=1.2,
            rmsd_ub=2.5,
            metadata={"foo": "bar"},
        )
        assert p.rank == 3
        assert p.rmsd_lb == 1.2
        assert p.metadata["foo"] == "bar"


class TestDockingResult:
    def test_empty_result(self) -> None:
        r = DockingResult()
        assert len(r) == 0
        assert r.engine == ""
        with pytest.raises(IndexError):
            _ = r.best

    def test_iter_and_indexing(self) -> None:
        poses = [
            Pose(ligand=_empty_protein(), score=-8.0, rank=0),
            Pose(ligand=_empty_protein(), score=-7.0, rank=1),
        ]
        r = DockingResult(poses=poses, engine="Test")
        assert len(r) == 2
        assert list(r) == poses
        assert r.best.score == -8.0

    def test_top_n(self) -> None:
        poses = [Pose(ligand=_empty_protein(), score=-i, rank=i) for i in range(5)]
        r = DockingResult(poses=poses)
        assert len(r.top_n(3)) == 3
        assert r.top_n(10) == poses


class _DummyEngine(DockingEngine):
    """Minimal concrete engine for testing the ABC contract."""

    name = "Dummy"

    def dock(self, receptor: Protein, ligand: object, **kwargs: object) -> DockingResult:
        return DockingResult(receptor=receptor, engine=self.name)


class TestEngineContract:
    def test_abstract_class_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            DockingEngine()  # type: ignore[abstract]

    def test_subclass_can_be_instantiated(self) -> None:
        engine = _DummyEngine()
        assert isinstance(engine, DockingEngine)
        assert engine.dock(_empty_protein(), "ligand.sdf").engine == "Dummy"

    def test_repr(self) -> None:
        assert repr(_DummyEngine()) == "_DummyEngine()"


class TestDiffDockStub:
    """DiffDock is a committed-but-unimplemented stub. It must be a
    coherent stub: instantiable, satisfying the DockingEngine ABC, and
    failing loud with a clear message that points at Vina."""

    def test_instantiates(self) -> None:
        from molforge.wrappers.docking import DiffDock

        engine = DiffDock()
        assert isinstance(engine, DockingEngine)

    def test_name(self) -> None:
        from molforge.wrappers.docking import DiffDock

        assert DiffDock().name == "DiffDock"

    def test_dock_raises_with_hint(self) -> None:
        from molforge.wrappers.docking import DiffDock

        with pytest.raises(NotImplementedError, match="not yet implemented"):
            DiffDock().dock(_empty_protein(), None)

    def test_error_points_at_vina(self) -> None:
        """The error message should steer users to the working engine."""
        from molforge.wrappers.docking import DiffDock

        with pytest.raises(NotImplementedError, match="Vina"):
            DiffDock().dock(_empty_protein(), None)
