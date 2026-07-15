"""Tests for :func:`molforge.parallel.map_parallel`."""

from __future__ import annotations

import pytest

from molforge.parallel import dock_many, fold_many, map_parallel, run_many


def _square(x: int) -> int:
    return x * x


def _fail_on_three(x: int) -> int:
    if x == 3:
        raise ValueError("boom on 3")
    return x * x


class TestMapParallel:
    @pytest.mark.parametrize("backend", ["serial", "thread"])
    def test_preserves_input_order(self, backend: str) -> None:
        assert map_parallel(_square, [1, 2, 3, 4], backend=backend, workers=3) == [1, 4, 9, 16]  # type: ignore[arg-type]

    def test_process_backend_runs(self) -> None:
        # `abs` is a picklable builtin, so this exercises the real process pool
        # without the "function defined in a test module isn't importable in the
        # worker" pickling gotcha.
        assert map_parallel(abs, [-1, -2, -3, -4], backend="process", workers=2) == [1, 2, 3, 4]

    def test_empty_returns_empty(self) -> None:
        assert map_parallel(_square, []) == []

    def test_workers_one_forces_serial(self) -> None:
        assert map_parallel(_square, [2, 3, 4], workers=1, backend="process") == [4, 9, 16]

    def test_single_item(self) -> None:
        assert map_parallel(_square, [5], backend="process") == [25]

    @pytest.mark.parametrize("backend", ["serial", "thread"])
    def test_on_error_raise_propagates(self, backend: str) -> None:
        with pytest.raises(ValueError, match="boom on 3"):
            map_parallel(_fail_on_three, [1, 2, 3, 4], backend=backend, workers=3, on_error="raise")  # type: ignore[arg-type]

    @pytest.mark.parametrize("backend", ["serial", "thread"])
    def test_on_error_skip_drops_failures_keeps_order(self, backend: str) -> None:
        # 3 raises and is dropped; the rest come back in order.
        result = map_parallel(
            _fail_on_three,
            [1, 2, 3, 4],
            backend=backend,
            workers=3,
            on_error="skip",  # type: ignore[arg-type]
        )
        assert result == [1, 4, 16]


class _FakeFolder:
    def predict(self, sequence: str, **kwargs: object) -> str:
        return f"structure:{sequence}:{kwargs.get('temperature', 1)}"


class _FailingFolder:
    def predict(self, sequence: str, **kwargs: object) -> str:
        if sequence == "bad":
            raise ValueError("bad sequence")
        return f"ok:{sequence}"


class _FakeDocker:
    def dock(self, receptor: str, ligand: str, **kwargs: object) -> str:
        return f"pose:{receptor}:{ligand}"


class _FakeMD:
    def run(self, system: str, **kwargs: object) -> str:
        return f"run:{system}"


class _Hinted:
    parallelism = "process"


class TestEngineWrappers:
    def test_fold_many_maps_predict(self) -> None:
        assert fold_many(_FakeFolder(), ["MKTV", "AAAA"]) == [
            "structure:MKTV:1",
            "structure:AAAA:1",
        ]

    def test_fold_many_forwards_kwargs(self) -> None:
        assert fold_many(_FakeFolder(), ["MK"], temperature=2) == ["structure:MK:2"]

    def test_fold_many_on_error_skip(self) -> None:
        assert fold_many(_FailingFolder(), ["a", "bad", "b"], on_error="skip") == ["ok:a", "ok:b"]

    def test_dock_many_one_receptor_many_ligands(self) -> None:
        assert dock_many(_FakeDocker(), "REC", ["ligA", "ligB"]) == [
            "pose:REC:ligA",
            "pose:REC:ligB",
        ]

    def test_run_many_generic_method(self) -> None:
        assert run_many(_FakeMD(), ["s1", "s2"], method="run") == ["run:s1", "run:s2"]

    def test_engine_backend_resolution(self) -> None:
        from molforge.parallel import _engine_backend

        assert _engine_backend(_Hinted(), None) == "process"  # engine hint
        assert _engine_backend(object(), None) == "serial"  # no hint -> safe default
        assert _engine_backend(_Hinted(), "thread") == "thread"  # explicit override wins


class TestEngineParallelismHints:
    """The batch wrappers pick their backend from the engine's `parallelism`
    hint. GPU engines default to serial; the CPU Vina wrapper opts into
    process parallelism.
    """

    def test_bases_default_to_serial(self) -> None:
        from molforge.docking import DockingEngine
        from molforge.generative import GenerativeEngine
        from molforge.md import MDEngine
        from molforge.wrappers.folding import FoldingEngine

        assert FoldingEngine.parallelism == "serial"
        assert DockingEngine.parallelism == "serial"
        assert MDEngine.parallelism == "serial"
        assert GenerativeEngine.parallelism == "serial"

    def test_vina_opts_into_process(self) -> None:
        from molforge.wrappers.docking import Vina

        assert Vina.parallelism == "process"
