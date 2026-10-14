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


# ----------------------------------------------------------------------
# Replay: operations that 0.9.0 added but didn't teach replay about
# ----------------------------------------------------------------------


class _FakeSampler:
    """An engine that returns a whole ensemble, as Chai-1 does."""

    name = "FakeSampler"

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed

    def predict(self, sequence: str) -> str:
        return f"best({sequence})"

    def predict_complex(self, spec: object) -> str:
        return f"best_complex({len(spec.entities)})"  # type: ignore[attr-defined]

    def predict_samples(self, sequence: str) -> list[str]:
        return [f"sample{i}({sequence})" for i in range(5)]

    def predict_complex_samples(self, spec: object) -> list[str]:
        return [f"csample{i}" for i in range(5)]


@pytest.fixture
def registered_sampler():
    from molforge import plugins

    plugins.register_engine("FakeSampler", _FakeSampler)
    yield
    plugins.clear()


def _spec_payload() -> dict:
    from molforge.folding import ComplexSpec, Entity

    return ComplexSpec(
        entities=(
            Entity(kind="protein", sequence="MKQ", copies=2),
            Entity(kind="ligand", smiles="CCO"),
        )
    ).to_provenance()


class TestReplayComplexPredictions:
    """A recorded `predict_complex` was describable but not replayable:
    the handler only ever looked for a `sequence` input."""

    def test_complex_predict_replays(self, registered_sampler) -> None:
        prov = Provenance.from_engine(
            "FakeSampler", operation="predict", inputs={"complex_spec": _spec_payload()}
        )
        assert replay(prov) == "best_complex(2)"

    def test_single_sequence_still_replays(self, registered_sampler) -> None:
        prov = Provenance.from_engine(
            "FakeSampler", operation="predict", inputs={"sequence": "MKQ"}
        )
        assert replay(prov) == "best(MKQ)"

    def test_live_spec_from_context_wins(self, registered_sampler) -> None:
        from molforge.folding import ComplexSpec, Entity

        prov = Provenance.from_engine(
            "FakeSampler", operation="predict", inputs={"complex_spec": _spec_payload()}
        )
        live = ComplexSpec(entities=(Entity(kind="protein", sequence="AAA"),))
        assert replay(prov, context={"complex_spec": live}) == "best_complex(1)"

    def test_engine_without_complex_support_explains_itself(self, registered_fakes) -> None:
        prov = Provenance.from_engine(
            "FakeFold", operation="predict", inputs={"complex_spec": _spec_payload()}
        )
        with pytest.raises(ReplayError, match="has no predict_complex"):
            replay(prov)


class TestReplaySampleEnsembles:
    def test_sequence_ensemble_replays(self, registered_sampler) -> None:
        prov = Provenance.from_engine(
            "FakeSampler", operation="predict_samples", inputs={"sequence": "MKQ"}
        )
        out = replay(prov)
        assert len(out) == 5
        assert out[0] == "sample0(MKQ)"

    def test_complex_ensemble_replays(self, registered_sampler) -> None:
        prov = Provenance.from_engine(
            "FakeSampler", operation="predict_samples", inputs={"complex_spec": _spec_payload()}
        )
        assert replay(prov) == [f"csample{i}" for i in range(5)]

    def test_constructor_parameters_are_reconstructed(self, registered_sampler) -> None:
        prov = Provenance.from_engine(
            "FakeSampler",
            operation="predict_samples",
            parameters={"seed": 7},
            inputs={"sequence": "MKQ"},
        )
        # The stub encodes nothing about the seed, so assert via the engine
        # the handler built rather than the output.
        from molforge.reproducibility import _construct, _resolve_engine

        engine = _construct(_resolve_engine("FakeSampler"), prov.parameters)
        assert engine.seed == 7
        assert len(replay(prov)) == 5

    def test_engine_without_ensemble_support_explains_itself(self, registered_fakes) -> None:
        prov = Provenance.from_engine(
            "FakeFold", operation="predict_samples", inputs={"sequence": "MKQ"}
        )
        with pytest.raises(ReplayError, match="has no predict_samples"):
            replay(prov)


class TestReplayFetch:
    """A download is a recorded step; a chain that can't replay its first
    step can't replay at all."""

    _PDB = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\nEND\n"

    @classmethod
    def _resp(cls) -> object:
        from unittest.mock import MagicMock

        resp = MagicMock()
        resp.read.return_value = cls._PDB.encode("utf-8")
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False
        return resp

    def test_fetch_is_resolvable_as_a_step_producer(self) -> None:
        from molforge.reproducibility import _resolve_engine

        assert _resolve_engine("molforge.io.fetch").__name__ == "fetch"

    def test_fetched_structure_replays(self) -> None:
        from unittest.mock import patch

        from molforge.io import fetch

        with patch("urllib.request.urlopen", return_value=self._resp()):
            protein = fetch("1ubq")

        # The download is cached, so replaying costs no network at all.
        with patch("urllib.request.urlopen", side_effect=AssertionError("should hit the cache")):
            out = replay(pipeline_manifest(protein))

        assert out.n_atoms == protein.n_atoms
        assert out.coords.tolist() == protein.coords.tolist()

    def test_replay_honours_the_recorded_source_and_format(self) -> None:
        from unittest.mock import patch

        from molforge.io import fetch

        with patch("urllib.request.urlopen", return_value=self._resp()):
            protein = fetch("P00520", source="alphafold")

        with patch("urllib.request.urlopen", return_value=self._resp()) as m:
            replay(pipeline_manifest(protein), context={})
        # Cached, so no call; the point is it didn't raise on an
        # unrecognised source/format.
        assert m.call_count == 0

    def test_context_can_redirect_the_id(self) -> None:
        from unittest.mock import patch

        from molforge.io import fetch

        with patch("urllib.request.urlopen", return_value=self._resp()):
            protein = fetch("1ubq")

        with patch("urllib.request.urlopen", return_value=self._resp()) as m:
            replay(pipeline_manifest(protein), context={"pdb_id": "4HHB"})
        assert m.call_args[0][0].endswith("4HHB.pdb")


class TestEveryRecordedOperationHasAHandler:
    def test_no_operation_is_left_unreplayable(self) -> None:
        """The regression this whole change closes: an operation a wrapper
        records but replay has never heard of."""
        import re
        from pathlib import Path

        import molforge.reproducibility as repro

        src = Path(repro.__file__).resolve().parents[1]
        recorded = set()
        for path in src.rglob("*.py"):
            recorded.update(re.findall(r'operation="([a-z_]+)"', path.read_text()))

        missing = recorded - set(repro._REPLAY_HANDLERS)
        assert not missing, f"operations recorded but not replayable: {sorted(missing)}"


# ----------------------------------------------------------------------
# Replay: generate / detect_pockets / predict_affinity
# ----------------------------------------------------------------------


class _FakeInverseFolder:
    name = "FakeInverse"

    def __init__(self, model_name: str = "default") -> None:
        self.model_name = model_name

    def generate(
        self, backbone: object, *, chains_to_design: str | None = None, timeout: float | None = None
    ) -> str:
        return f"designs(bb={backbone},chains={chains_to_design},model={self.model_name})"


class _FakeDeNovo:
    name = "FakeDeNovo"

    def __init__(self, device: str | None = None) -> None:
        self.device = device

    def generate(self, *, length: int | None = None, contigs: object = None) -> str:
        return f"backbones(length={length},contigs={contigs},device={self.device})"


class _FakeAffinity:
    name = "FakeAffinity"

    def __init__(self, model_version: str = "boltz2") -> None:
        self.model_version = model_version

    def predict_complex(self, spec: object) -> str:
        return "folded_only"

    def predict_affinity(self, spec: object) -> str:
        return f"affinity({len(spec.entities)})"  # type: ignore[attr-defined]


def _fake_detector(
    protein: object, *, fpocket_executable: str = "fpocket", min_volume: float | None = None
) -> str:
    return f"pockets({protein},exe={fpocket_executable},vol={min_volume})"


@pytest.fixture
def registered_extras():
    from molforge import plugins

    plugins.register_engine("FakeInverse", _FakeInverseFolder)
    plugins.register_engine("FakeDeNovo", _FakeDeNovo)
    plugins.register_engine("FakeAffinity", _FakeAffinity)
    plugins.register_engine("FakeDetector", _fake_detector)
    yield
    plugins.clear()


class TestReplayGenerate:
    """Inverse folding and de-novo generation share one operation; which
    shape a step is depends on whether a backbone was recorded."""

    def test_inverse_folding_uses_the_recorded_backbone(self, registered_extras) -> None:
        prov = Provenance.from_engine(
            "FakeInverse",
            operation="generate",
            parameters={"model_name": "v_48_020", "chains_to_design": "A"},
            inputs={"backbone": "bb.pdb"},
        )
        out = replay(prov)
        assert "bb=bb.pdb" in out
        assert "chains=A" in out

    def test_call_arguments_are_separated_from_constructor_ones(self, registered_extras) -> None:
        """Provenance stores both in one flat dict; each must land in the
        right place or the call is replayed with its arguments dropped."""
        prov = Provenance.from_engine(
            "FakeInverse",
            operation="generate",
            parameters={"model_name": "v_48_020", "chains_to_design": "B"},
            inputs={"backbone": "bb.pdb"},
        )
        out = replay(prov)
        assert "model=v_48_020" in out  # constructor
        assert "chains=B" in out  # call

    def test_backbone_threads_from_the_previous_step(self, registered_fakes) -> None:
        from molforge import plugins

        plugins.register_engine("FakeInverse", _FakeInverseFolder)
        prov = Provenance.from_engine(
            "FakeInverse", operation="generate", inputs={}, parent=_fold_prov()
        )
        assert "bb=structure(MKTV,T=0.5)" in replay(prov)

    def test_context_overrides_the_recorded_backbone(self, registered_extras) -> None:
        prov = Provenance.from_engine(
            "FakeInverse", operation="generate", inputs={"backbone": "bb.pdb"}
        )
        assert "bb=other.pdb" in replay(prov, context={"backbone": "other.pdb"})

    def test_de_novo_needs_no_backbone(self, registered_extras) -> None:
        """RFdiffusion-shaped: the parameters are the whole input."""
        prov = Provenance.from_engine(
            "FakeDeNovo",
            operation="generate",
            parameters={"length": 120, "device": "cuda"},
            inputs={},
        )
        out = replay(prov)
        assert "length=120" in out
        assert "device=cuda" in out


class TestReplayDetectPockets:
    def test_structure_comes_from_context(self, registered_extras) -> None:
        prov = Provenance.from_engine(
            "FakeDetector",
            operation="detect_pockets",
            parameters={"fpocket_executable": "fpocket", "min_volume": 100},
            inputs={"protein": "1UBQ"},
        )
        out = replay(prov, context={"protein": "<structure>"})
        assert "pockets(<structure>" in out
        assert "vol=100" in out

    def test_structure_threads_from_the_previous_step(self, registered_fakes) -> None:
        from molforge import plugins

        plugins.register_engine("FakeDetector", _fake_detector)
        prov = Provenance.from_engine(
            "FakeDetector", operation="detect_pockets", inputs={}, parent=_fold_prov()
        )
        assert "pockets(structure(MKTV,T=0.5)" in replay(prov)

    def test_unchained_step_without_context_explains_itself(self, registered_extras) -> None:
        """The recorded `protein` input is a display name, not something
        that can be reloaded — so say that rather than passing it on."""
        prov = Provenance.from_engine(
            "FakeDetector", operation="detect_pockets", inputs={"protein": "1UBQ"}
        )
        with pytest.raises(ReplayError, match="only the structure's name is recorded"):
            replay(prov)

    def test_builtin_detectors_resolve(self) -> None:
        from molforge.reproducibility import _resolve_engine

        assert _resolve_engine("fpocket").__name__ == "detect_pockets"
        assert _resolve_engine("p2rank").__name__ == "detect_pockets_p2rank"


class TestReplayPredictAffinity:
    def test_calls_predict_affinity_not_predict(self, registered_extras) -> None:
        """Recorded as `predict` until 0.9.2, which replayed as a plain fold
        and returned a structure with no affinity on it."""
        from molforge.folding import ComplexSpec, Entity

        spec = ComplexSpec(
            entities=(
                Entity(kind="protein", sequence="MKQ"),
                Entity(kind="ligand", smiles="CCO"),
            )
        )
        prov = Provenance.from_engine(
            "FakeAffinity",
            operation="predict_affinity",
            parameters={"model_version": "boltz2"},
            inputs={"complex_spec": spec.to_provenance()},
        )
        assert replay(prov) == "affinity(2)"

    def test_sequence_only_step_is_rejected(self, registered_extras) -> None:
        prov = Provenance.from_engine(
            "FakeAffinity", operation="predict_affinity", inputs={"sequence": "MKQ"}
        )
        with pytest.raises(ReplayError, match="needs a complex_spec"):
            replay(prov)

    def test_engine_without_affinity_support_explains_itself(self, registered_fakes) -> None:
        from molforge.folding import ComplexSpec, Entity

        spec = ComplexSpec(entities=(Entity(kind="protein", sequence="MKQ"),))
        prov = Provenance.from_engine(
            "FakeFold", operation="predict_affinity", inputs={"complex_spec": spec.to_provenance()}
        )
        with pytest.raises(ReplayError, match="has no predict_affinity"):
            replay(prov)


class TestWrappersRecordTheirOperation:
    """The gap this closes was never a missing handler — it was wrappers
    attaching provenance with no operation at all, which replay reports as
    'no recorded operation'."""

    def test_boltz_affinity_is_its_own_operation(self) -> None:
        from molforge.folding import ComplexSpec, Entity
        from molforge.wrappers.folding import Boltz

        spec = ComplexSpec(
            entities=(
                Entity(kind="protein", sequence="MKQ"),
                Entity(kind="ligand", smiles="CCO"),
            )
        )
        fold = Boltz()._build_provenance(spec, single_sequence=None)
        affinity = Boltz()._build_provenance(spec, single_sequence=None, affinity_binder="B")
        assert fold.operation == "predict"
        assert affinity.operation == "predict_affinity"

    def test_recording_an_operation_does_not_move_the_cache_key(self) -> None:
        """`operation` is excluded from the cache key by design, so naming
        these operations invalidated nothing on anyone's disk."""
        from molforge.cache import cache_key

        base = Provenance.from_engine("X", parameters={"a": 1}, inputs={"b": 2})
        keys = {
            cache_key(base.replace(operation=op))
            for op in ("", "generate", "detect_pockets", "predict_affinity")
        }
        assert len(keys) == 1


class TestCallKwargs:
    def test_filters_to_the_callables_own_parameters(self) -> None:
        from molforge.reproducibility import _call_kwargs

        def fn(protein: object, *, min_volume: float | None = None) -> None: ...

        got = _call_kwargs(fn, {"min_volume": 100, "device": "cuda", "protein": "x"})
        assert got == {"min_volume": 100, "protein": "x"}

    def test_unintrospectable_callable_yields_nothing(self) -> None:
        from molforge.reproducibility import _call_kwargs

        assert _call_kwargs(print, {"a": 1}) in ({}, {})
