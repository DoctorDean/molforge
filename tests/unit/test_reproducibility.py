"""Tests for molforge.reproducibility (pipeline.yaml emit/load)."""

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING

import pytest

from molforge.core import Protein
from molforge.core import metadata_keys as mk
from molforge.core.provenance import Provenance
from molforge.reproducibility import (
    PipelineManifest,
    PipelineStep,
    ReplayError,
    emit_pipeline,
    load_pipeline,
    pipeline_manifest,
    register_replay_handler,
    replay,
)

if TYPE_CHECKING:
    from pathlib import Path


def _chain() -> Provenance:
    """A 3-step ESMFold → Vina → OpenMM provenance chain."""
    p1 = Provenance.from_engine(
        "ESMFold",
        engine_version="1.0.3",
        inputs={"sequence": "MKTAYIAK"},
        parameters={"recycles": 4},
    )
    p2 = Provenance.from_engine(
        "Vina",
        engine_version="1.2.5",
        inputs={"ligand": "lig.sdf"},
        parameters={"exhaustiveness": 8},
        parent=p1,
    )
    return Provenance.from_engine(
        "OpenMM",
        engine_version="8.1",
        inputs={"system": "sys.xml"},
        parameters={"steps": 100000},
        parent=p2,
    )


def _output() -> Protein:
    obj = Protein(name="7XYZ_model")
    obj.metadata[mk.PROVENANCE] = _chain()
    return obj


class TestManifestConstruction:
    def test_steps_linearized_oldest_first(self) -> None:
        m = pipeline_manifest(_output())
        assert len(m) == 3
        assert [s.engine for s in m] == ["ESMFold", "Vina", "OpenMM"]
        assert [s.step for s in m] == [1, 2, 3]

    def test_step_carries_inputs_and_parameters(self) -> None:
        m = pipeline_manifest(_output())
        assert m.steps[0].inputs == {"sequence": "MKTAYIAK"}
        assert m.steps[0].parameters == {"recycles": 4}
        assert m.steps[0].engine_version == "1.0.3"

    def test_environment_block(self) -> None:
        env = pipeline_manifest(_output()).environment
        assert env["engines"] == {"ESMFold": "1.0.3", "Vina": "1.2.5", "OpenMM": "8.1"}
        assert "python_version" in env
        assert "platform" in env
        assert env["molforge_version"]  # non-empty

    def test_output_descriptor(self) -> None:
        m = pipeline_manifest(_output())
        assert m.output == {"type": "Protein", "name": "7XYZ_model"}

    def test_accepts_raw_provenance(self) -> None:
        m = pipeline_manifest(_chain())
        assert len(m) == 3
        assert m.output == {}  # no output object to describe

    def test_missing_provenance_raises(self) -> None:
        with pytest.raises(ValueError, match="no provenance found"):
            pipeline_manifest(Protein(name="bare"))


class TestRoundTrip:
    def test_dict_round_trip(self) -> None:
        m = pipeline_manifest(_output())
        assert PipelineManifest.from_dict(m.to_dict()) == m

    def test_json_round_trip(self) -> None:
        m = pipeline_manifest(_output())
        assert PipelineManifest.from_json(m.to_json()) == m

    def test_yaml_round_trip(self) -> None:
        pytest.importorskip("yaml")
        m = pipeline_manifest(_output())
        assert PipelineManifest.from_yaml(m.to_yaml()) == m

    def test_step_dict_round_trip(self) -> None:
        step = PipelineStep(
            step=1, engine="E", engine_version="1", inputs={"a": 1}, parameters={"b": 2}
        )
        assert PipelineStep.from_dict(step.to_dict()) == step

    def test_to_dict_shape(self) -> None:
        d = pipeline_manifest(_output()).to_dict()
        assert d["molforge_pipeline"] == 1
        assert set(d) == {"molforge_pipeline", "generated", "environment", "steps", "output"}


class TestEmitLoad:
    def test_emit_load_yaml(self, tmp_path: Path) -> None:
        pytest.importorskip("yaml")
        m = pipeline_manifest(_output())
        path = tmp_path / "pipeline.yaml"
        returned = emit_pipeline(_output(), path)
        assert path.exists()
        assert returned == m == load_pipeline(path)

    def test_emit_load_json(self, tmp_path: Path) -> None:
        m = pipeline_manifest(_output())
        path = tmp_path / "pipeline.json"
        emit_pipeline(_output(), path, fmt="json")
        assert load_pipeline(path) == m

    def test_emit_unknown_format_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown fmt"):
            emit_pipeline(_output(), tmp_path / "p.xml", fmt="xml")

    def test_json_load_needs_no_yaml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # A .json file must load even when PyYAML is unavailable.
        path = tmp_path / "pipeline.json"
        emit_pipeline(_output(), path, fmt="json")
        _hide_yaml(monkeypatch)
        assert len(load_pipeline(path)) == 3


class TestDescribe:
    def test_describe_lists_steps(self) -> None:
        text = pipeline_manifest(_output()).describe()
        assert "3 steps" in text
        assert "1. ESMFold v1.0.3" in text
        assert "2. Vina v1.2.5" in text


class TestYamlDependencyGate:
    def test_to_yaml_without_pyyaml_raises_install_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        m = pipeline_manifest(_chain())
        _hide_yaml(monkeypatch)
        with pytest.raises(ImportError, match=r"molforge\[repro\]"):
            m.to_yaml()

    def test_json_still_works_without_pyyaml(self, monkeypatch: pytest.MonkeyPatch) -> None:
        m = pipeline_manifest(_chain())
        _hide_yaml(monkeypatch)
        assert m.to_json()  # no exception


def _hide_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``import yaml`` raise ImportError, simulating the extra not installed."""
    real_import = builtins.__import__

    def _blocked(name: str, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if name == "yaml" or name.startswith("yaml."):
            raise ImportError("No module named 'yaml'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)


# ----------------------------------------------------------------------
# Replay
# ----------------------------------------------------------------------


class _FakeFolder:
    name = "FakeFold"
    parallelism = "serial"

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def predict(self, sequence: str, **kwargs: object) -> str:
        return f"structure({sequence},T={self.temperature})"


class _FakeDocker:
    name = "FakeDock"

    def __init__(self, exhaustiveness: int = 8) -> None:
        self.exhaustiveness = exhaustiveness

    def dock(self, receptor: object, ligand: object) -> str:
        return f"poses(rec={receptor},lig={ligand},ex={self.exhaustiveness})"


@pytest.fixture
def registered_fakes():
    """Register fake fold/dock engines in the plugin registry for a test."""
    from molforge import plugins

    plugins.register_engine("FakeFold", _FakeFolder)
    plugins.register_engine("FakeDock", _FakeDocker)
    yield
    plugins.clear()


def _fold_prov() -> Provenance:
    return Provenance.from_engine(
        "FakeFold",
        operation="predict",
        parameters={"temperature": 0.5},
        inputs={"sequence": "MKTV"},
    )


def _fold_dock_prov() -> Provenance:
    return Provenance.from_engine(
        "FakeDock",
        operation="dock",
        parameters={"exhaustiveness": 16},
        inputs={"receptor": "model.pdb", "ligand": "aspirin.sdf"},
        parent=_fold_prov(),
    )


class TestReplay:
    def test_fold_only(self, registered_fakes) -> None:
        assert replay(_fold_prov()) == "structure(MKTV,T=0.5)"

    def test_fold_then_dock_threads_output(self, registered_fakes) -> None:
        # The fold output becomes the dock receptor; ligand from context;
        # constructor params (temperature, exhaustiveness) reconstructed.
        out = replay(_fold_dock_prov(), context={"ligand": "CCO"})
        assert out == "poses(rec=structure(MKTV,T=0.5),lig=CCO,ex=16)"

    def test_ligand_falls_back_to_recorded_literal(self, registered_fakes) -> None:
        # No context → the recorded ligand identifier is used verbatim.
        out = replay(_fold_dock_prov())
        assert "lig=aspirin.sdf" in out

    def test_replay_from_manifest(self, registered_fakes) -> None:
        manifest = pipeline_manifest(_fold_dock_prov())
        assert replay(manifest, context={"ligand": "CCO"}).startswith("poses(")

    def test_replay_round_tripped_manifest(self, registered_fakes) -> None:
        # A manifest that survived to_dict/from_dict still replays.
        manifest = PipelineManifest.from_dict(pipeline_manifest(_fold_dock_prov()).to_dict())
        assert replay(manifest, context={"ligand": "CCO"}).startswith("poses(")

    def test_missing_engine_raises(self, registered_fakes) -> None:
        from molforge import plugins

        plugins.clear()
        plugins.register_engine("FakeFold", _FakeFolder)  # dock engine absent
        with pytest.raises(ReplayError, match="no engine registered as 'FakeDock'"):
            replay(_fold_dock_prov(), context={"ligand": "CCO"})

    def test_missing_operation_raises(self, registered_fakes) -> None:
        prov = Provenance.from_engine("FakeFold", inputs={"sequence": "MKTV"})
        with pytest.raises(ReplayError, match="no recorded operation"):
            replay(prov)

    def test_unregistered_operation_raises(self, registered_fakes) -> None:
        prov = Provenance.from_engine("FakeFold", operation="teleport", inputs={"sequence": "M"})
        with pytest.raises(ReplayError, match="no replay handler"):
            replay(prov)

    def test_missing_input_raises(self, registered_fakes) -> None:
        prov = Provenance.from_engine("FakeFold", operation="predict", inputs={})
        with pytest.raises(ReplayError, match="needs a sequence"):
            replay(prov)

    def test_context_overrides_recorded_input(self, registered_fakes) -> None:
        out = replay(_fold_prov(), context={"sequence": "AAAA"})
        assert out == "structure(AAAA,T=0.5)"

    def test_builtin_engines_resolve(self) -> None:
        from molforge.reproducibility import _resolve_engine

        assert _resolve_engine("ESMFold").__name__ == "ESMFold"
        assert _resolve_engine("Vina").__name__ == "Vina"

    def test_custom_operation_handler(self, registered_fakes) -> None:
        from molforge import plugins

        seen = {}

        @register_replay_handler("scan")
        def _scan(factory, step, upstream, context):  # type: ignore[no-untyped-def]
            seen["ran"] = True
            return "scanned"

        plugins.register_engine("Scanner", _FakeFolder)
        prov = Provenance.from_engine("Scanner", operation="scan", inputs={})
        assert replay(prov) == "scanned"
        assert seen["ran"]


class TestOperationInManifest:
    def test_operation_flows_into_step(self) -> None:
        m = pipeline_manifest(_fold_prov())
        assert m.steps[0].operation == "predict"

    def test_operation_round_trips(self) -> None:
        step = PipelineStep(step=1, engine="E", operation="dock")
        assert PipelineStep.from_dict(step.to_dict()).operation == "dock"
