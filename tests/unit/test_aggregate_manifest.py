"""Tests for aggregate provenance manifests.

A single manifest answers "how was this one thing made". Plenty of
results aren't one thing, and emitting one manifest per contributing
prediction answers the question in a form nobody can read — while
restating the shared upstream work once per prediction. These cover the
aggregate shape and, above all, that shared ancestry really collapses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molforge.core.provenance import Provenance
from molforge.reproducibility import (
    AggregateManifest,
    AggregateStep,
    PipelineManifest,
    aggregate_manifest,
    emit_aggregate,
    load_aggregate,
    pipeline_manifest,
)


def _fan_out(n: int = 5, *, shared_depth: int = 2) -> list[Provenance]:
    """`n` sibling outputs over a shared upstream chain of `shared_depth`."""
    node: Provenance | None = None
    for i in range(shared_depth):
        node = Provenance.from_engine(f"shared{i}", operation="run", parent=node)
    return [
        Provenance.from_engine("Boltz", operation="predict", parameters={"seed": s}, parent=node)
        for s in range(n)
    ]


class TestDeduplication:
    """The point of the whole exercise."""

    def test_shared_ancestry_collapses(self) -> None:
        manifest = aggregate_manifest(_fan_out(5, shared_depth=2))
        # 2 shared + 5 distinct leaves = 7 unique, from 5 * 3 = 15 positions.
        assert manifest.summary() == {
            "outputs": 5,
            "unique_steps": 7,
            "total_steps": 15,
            "shared_steps": 2,
        }

    def test_shared_steps_record_their_fan_out(self) -> None:
        manifest = aggregate_manifest(_fan_out(5, shared_depth=2))
        for step in manifest.shared_steps:
            assert step.contributes_to == 5
        assert [s.engine for s in manifest.shared_steps] == ["shared0", "shared1"]

    def test_leaves_are_not_shared(self) -> None:
        manifest = aggregate_manifest(_fan_out(4))
        leaves = [s for s in manifest if s.engine == "Boltz"]
        assert len(leaves) == 4
        assert all(s.contributes_to == 1 for s in leaves)

    def test_identical_outputs_collapse_entirely(self) -> None:
        """Two outputs of the same computation are one chain, not two."""
        prov = Provenance.from_engine("ESMFold", operation="predict", inputs={"sequence": "MK"})
        twin = Provenance.from_engine("ESMFold", operation="predict", inputs={"sequence": "MK"})
        manifest = aggregate_manifest([prov, twin])
        assert len(manifest) == 1
        assert manifest.steps[0].contributes_to == 2
        assert manifest.n_outputs == 2

    def test_divergence_point_is_respected(self) -> None:
        """Chains sharing a prefix must split exactly where they diverge."""
        root = Provenance.from_engine("prep", operation="prepare")
        mid = Provenance.from_engine("MMseqs2", operation="search", parent=root)
        a = Provenance.from_engine("Boltz", operation="predict", parent=mid)
        b = Provenance.from_engine("Chai-1", operation="predict", parent=mid)
        manifest = aggregate_manifest([a, b])

        assert len(manifest) == 4
        by_engine = {s.engine: s for s in manifest}
        assert by_engine["prep"].contributes_to == 2
        assert by_engine["MMseqs2"].contributes_to == 2
        assert by_engine["Boltz"].contributes_to == 1
        assert by_engine["Chai-1"].contributes_to == 1
        assert by_engine["Boltz"].parent == by_engine["MMseqs2"].id
        assert by_engine["Chai-1"].parent == by_engine["MMseqs2"].id

    def test_same_engine_different_parameters_stays_distinct(self) -> None:
        """Dedup must not merge two genuinely different runs."""
        root = Provenance.from_engine("prep", operation="prepare")
        a = Provenance.from_engine(
            "Boltz", operation="predict", parameters={"seed": 1}, parent=root
        )
        b = Provenance.from_engine(
            "Boltz", operation="predict", parameters={"seed": 2}, parent=root
        )
        manifest = aggregate_manifest([a, b])
        assert len([s for s in manifest if s.engine == "Boltz"]) == 2

    def test_same_step_under_different_ancestry_stays_distinct(self) -> None:
        """Identity covers the whole chain: same work, different history,
        different step — otherwise a manifest would claim one run served
        two ancestries it never saw."""
        a = Provenance.from_engine(
            "Vina", operation="dock", parent=Provenance.from_engine("ESMFold", operation="predict")
        )
        b = Provenance.from_engine(
            "Vina", operation="dock", parent=Provenance.from_engine("Boltz", operation="predict")
        )
        manifest = aggregate_manifest([a, b])
        assert len([s for s in manifest if s.engine == "Vina"]) == 2


class TestOrdering:
    def test_a_parent_always_precedes_its_child(self) -> None:
        """Walking oldest-first is what makes a topological sort unnecessary
        — this asserts the invariant that relies on."""
        manifest = aggregate_manifest(_fan_out(6, shared_depth=4))
        seen: set[str] = set()
        for step in manifest:
            if step.parent is not None:
                assert step.parent in seen, f"{step.id} precedes its parent {step.parent}"
            seen.add(step.id)

    def test_roots_have_no_parent(self) -> None:
        manifest = aggregate_manifest(_fan_out(3, shared_depth=2))
        roots = manifest.roots()
        assert len(roots) == 1
        assert roots[0].engine == "shared0"
        assert roots[0].parent is None

    def test_outputs_point_at_their_terminal_step(self) -> None:
        outputs = _fan_out(3)
        manifest = aggregate_manifest(outputs)
        terminal_ids = {o["step"] for o in manifest.outputs}
        assert len(terminal_ids) == 3
        for step_id in terminal_ids:
            assert manifest.step_by_id(step_id).engine == "Boltz"


class TestStepIds:
    def test_ids_are_short_but_unique(self) -> None:
        manifest = aggregate_manifest(_fan_out(50, shared_depth=3))
        ids = [s.id for s in manifest]
        assert len(set(ids)) == len(ids)
        assert all(len(i) == 12 for i in ids)

    def test_ids_derive_from_content_id(self) -> None:
        prov = Provenance.from_engine("ESMFold", operation="predict")
        manifest = aggregate_manifest([prov])
        assert prov.content_id().startswith(manifest.steps[0].id)

    def test_ids_are_stable_across_runs(self) -> None:
        """Two aggregations of the same work must agree, or a manifest
        can't be diffed against an earlier one."""
        first = aggregate_manifest(_fan_out(4, shared_depth=2))
        second = aggregate_manifest(_fan_out(4, shared_depth=2))
        assert [s.id for s in first] == [s.id for s in second]

    def test_ids_lengthen_rather_than_collide(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A prefix collision would silently merge unrelated steps, so the
        shortener must back off to a longer prefix instead."""
        import molforge.reproducibility as repro

        # Force the 12-char tier to collide by making it 1 char wide.
        monkeypatch.setattr(repro, "_ID_LENGTHS", (1, 64))
        manifest = aggregate_manifest(_fan_out(40, shared_depth=2))
        ids = [s.id for s in manifest]
        assert len(set(ids)) == len(ids)
        assert all(len(i) == 64 for i in ids)

    def test_step_by_id_raises_for_unknown(self) -> None:
        manifest = aggregate_manifest(_fan_out(2))
        with pytest.raises(KeyError, match="no step with id"):
            manifest.step_by_id("deadbeef")


class TestFromManifests:
    """A directory of pipeline.yaml files is the usual starting point."""

    def test_matches_aggregating_the_live_objects(self) -> None:
        outputs = _fan_out(4, shared_depth=3)
        live = aggregate_manifest(outputs)
        via = PipelineManifest.aggregate([pipeline_manifest(o) for o in outputs])

        assert [s.id for s in live] == [s.id for s in via]
        assert [s.parent for s in live] == [s.parent for s in via]
        assert [s.contributes_to for s in live] == [s.contributes_to for s in via]
        assert live.summary() == via.summary()

    def test_round_trips_through_disk(self, tmp_path: Path) -> None:
        from molforge.reproducibility import emit_pipeline, load_pipeline

        outputs = _fan_out(3, shared_depth=2)
        paths = []
        for i, obj in enumerate(outputs):
            path = tmp_path / f"run{i}.json"
            emit_pipeline(obj, path, fmt="json")
            paths.append(path)

        combined = PipelineManifest.aggregate([load_pipeline(p) for p in paths])
        assert combined.summary() == aggregate_manifest(outputs).summary()

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one manifest"):
            PipelineManifest.aggregate([])

    def test_manifest_without_steps_rejected(self) -> None:
        empty = PipelineManifest(environment={}, steps=[])
        with pytest.raises(ValueError, match="no steps"):
            PipelineManifest.aggregate([empty])


class TestEnvironment:
    def test_engine_versions_are_unioned_across_outputs(self) -> None:
        a = Provenance.from_engine("ESMFold", operation="predict", engine_version="1.0.3")
        b = Provenance.from_engine("Boltz", operation="predict", engine_version="2.1.0")
        env = aggregate_manifest([a, b]).environment
        assert env["engines"] == {"ESMFold": "1.0.3", "Boltz": "2.1.0"}

    def test_disagreeing_versions_are_both_recorded(self) -> None:
        """A version two outputs disagree about is a finding about the run,
        not something to quietly pick a winner for."""
        a = Provenance.from_engine("Boltz", operation="predict", engine_version="2.1.0")
        b = Provenance.from_engine("Boltz", operation="predict", engine_version="2.2.0")
        env = aggregate_manifest([a, b]).environment
        assert env["engines"]["Boltz"] == ["2.1.0", "2.2.0"]

    def test_platform_keys_present(self) -> None:
        env = aggregate_manifest(_fan_out(2)).environment
        for key in ("molforge_version", "python_version", "platform"):
            assert key in env


class TestFirstRun:
    def test_earliest_timestamp_wins(self) -> None:
        root_early = Provenance(engine="prep", timestamp="2026-01-01T00:00:00+00:00")
        root_late = Provenance(engine="prep", timestamp="2026-06-01T00:00:00+00:00")
        a = Provenance(engine="Boltz", parameters={"seed": 1}, parent=root_late)
        b = Provenance(engine="Boltz", parameters={"seed": 2}, parent=root_early)
        # Same content id for both roots (timestamps are excluded), so they
        # are one step whose first_run is the earlier of the two.
        manifest = aggregate_manifest([a, b])
        prep = next(s for s in manifest if s.engine == "prep")
        assert prep.contributes_to == 2
        assert prep.first_run == "2026-01-01T00:00:00+00:00"

    def test_blank_timestamp_does_not_win(self) -> None:
        root_blank = Provenance(engine="prep", timestamp="")
        root_dated = Provenance(engine="prep", timestamp="2026-06-01T00:00:00+00:00")
        a = Provenance(engine="Boltz", parameters={"seed": 1}, parent=root_dated)
        b = Provenance(engine="Boltz", parameters={"seed": 2}, parent=root_blank)
        manifest = aggregate_manifest([a, b])
        prep = next(s for s in manifest if s.engine == "prep")
        assert prep.first_run == "2026-06-01T00:00:00+00:00"


class TestSerialisation:
    def test_json_round_trip(self) -> None:
        manifest = aggregate_manifest(_fan_out(4, shared_depth=2))
        assert AggregateManifest.from_json(manifest.to_json()).to_dict() == manifest.to_dict()

    def test_yaml_round_trip(self) -> None:
        pytest.importorskip("yaml")
        manifest = aggregate_manifest(_fan_out(4, shared_depth=2))
        assert AggregateManifest.from_yaml(manifest.to_yaml()).to_dict() == manifest.to_dict()

    def test_schema_key_is_emitted(self) -> None:
        payload = aggregate_manifest(_fan_out(2)).to_dict()
        assert payload["molforge_aggregate"] == 1

    def test_summary_is_derived_not_trusted(self) -> None:
        """A hand-edited summary must not be able to contradict the steps."""
        manifest = aggregate_manifest(_fan_out(3, shared_depth=2))
        payload = manifest.to_dict()
        payload["summary"] = {"outputs": 999, "unique_steps": 999}
        assert AggregateManifest.from_dict(payload).summary() == manifest.summary()

    def test_from_dict_tolerates_missing_keys(self) -> None:
        manifest = AggregateManifest.from_dict({})
        assert len(manifest) == 0
        assert manifest.n_outputs == 0

    def test_step_from_dict_requires_id_and_engine(self) -> None:
        with pytest.raises(ValueError, match="missing required"):
            AggregateStep.from_dict({"engine": "Boltz"})
        with pytest.raises(ValueError, match="missing required"):
            AggregateStep.from_dict({"id": "abc"})

    def test_emit_and_load(self, tmp_path: Path) -> None:
        pytest.importorskip("yaml")
        outputs = _fan_out(3, shared_depth=2)
        path = tmp_path / "aggregate.yaml"
        written = emit_aggregate(outputs, path)
        assert load_aggregate(path).to_dict() == written.to_dict()

    def test_emit_json_by_format(self, tmp_path: Path) -> None:
        path = tmp_path / "aggregate.json"
        written = emit_aggregate(_fan_out(2), path, fmt="json")
        assert load_aggregate(path).to_dict() == written.to_dict()

    def test_emit_rejects_unknown_format(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown fmt"):
            emit_aggregate(_fan_out(2), tmp_path / "x.toml", fmt="toml")

    def test_json_path_needs_no_extra(self, tmp_path: Path) -> None:
        """The whole aggregate shape has to work without PyYAML — it is
        part of molforge's numpy-only core, like PipelineManifest."""
        manifest = aggregate_manifest(_fan_out(3, shared_depth=2))
        assert AggregateManifest.from_json(manifest.to_json()).to_dict() == manifest.to_dict()
        path = tmp_path / "aggregate.json"
        emit_aggregate(_fan_out(2), path, fmt="json")
        assert load_aggregate(path).n_outputs == 2

    def test_yaml_without_pyyaml_raises_the_install_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def no_yaml(name: str, *args: object, **kwargs: object) -> object:
            if name == "yaml":
                raise ImportError("No module named 'yaml'")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", no_yaml)
        with pytest.raises(ImportError, match=r"molforge\[repro\]"):
            aggregate_manifest(_fan_out(2)).to_yaml()


class TestInputValidation:
    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one output"):
            aggregate_manifest([])

    def test_object_without_provenance_rejected(self) -> None:
        with pytest.raises(ValueError, match="no provenance found"):
            aggregate_manifest([object()])

    def test_accepts_a_generator(self) -> None:
        manifest = aggregate_manifest(p for p in _fan_out(3))
        assert manifest.n_outputs == 3


class TestDescribe:
    def test_reports_the_dedup_saving(self) -> None:
        text = aggregate_manifest(_fan_out(5, shared_depth=2)).describe()
        assert "5 outputs" in text
        assert "7 unique steps from 15" in text
        assert "8 deduplicated" in text

    def test_lists_shared_steps_most_shared_first(self) -> None:
        root = Provenance.from_engine("prep", operation="prepare")
        mid = Provenance.from_engine("MMseqs2", operation="search", parent=root)
        outputs = [
            Provenance.from_engine("Boltz", parameters={"seed": s}, parent=mid) for s in range(3)
        ]
        outputs.append(Provenance.from_engine("Chai-1", parent=root))
        text = aggregate_manifest(outputs).describe()
        # prep feeds 4 outputs, MMseqs2 only 3.
        assert text.index("prep") < text.index("MMseqs2")

    def test_single_output_reads_naturally(self) -> None:
        text = aggregate_manifest([Provenance.from_engine("ESMFold")]).describe()
        assert "1 output," in text
        assert "1 unique step " in text


class TestDunders:
    def test_len_is_unique_steps(self) -> None:
        manifest = aggregate_manifest(_fan_out(5, shared_depth=2))
        assert len(manifest) == 7

    def test_iterates_steps(self) -> None:
        manifest = aggregate_manifest(_fan_out(3))
        assert all(isinstance(s, AggregateStep) for s in manifest)
        assert len(list(manifest)) == len(manifest)
