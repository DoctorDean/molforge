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
    emit_pipeline,
    load_pipeline,
    pipeline_manifest,
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
