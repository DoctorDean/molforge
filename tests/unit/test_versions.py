"""Tests for the engine/backend version registry (GAP-0004).

The registry's contract is that it is *safe to call anywhere*: on a bare
machine with none of the engines installed, on one where a binary hangs,
on one where a tool prints a banner in a shape nobody anticipated. So most
of what's tested here is the failure behaviour.
"""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from molforge.versions import _REGISTRY as REGISTRY
from molforge.versions import BackendVersion, engine_versions


@pytest.fixture(autouse=True)
def _clear_probe_cache() -> None:
    """The registry memoizes its sweep; don't leak it between tests."""
    import molforge.versions as versions_module

    versions_module._probe_cache.clear()


class TestRegistryShape:
    def test_every_known_backend_is_reported(self) -> None:
        """Absent engines still appear — that's a fact about the machine."""
        backends = engine_versions(probe_executables=False)
        assert set(backends) == {b.name for b in REGISTRY}
        assert len(backends) == len(REGISTRY)

    def test_names_are_unique(self) -> None:
        names = [b.name for b in REGISTRY]
        assert len(names) == len(set(names)), "registry keys on name; duplicates would shadow"

    def test_kinds_and_categories_are_from_the_documented_sets(self) -> None:
        for backend in REGISTRY:
            assert backend.kind in {"python", "executable", "repo"}
            assert backend.category in {
                "folding",
                "docking",
                "pockets",
                "md",
                "freeenergy",
                "generative",
                "runtime",
            }

    def test_detectable_backends_declare_what_to_look_for(self) -> None:
        for backend in REGISTRY:
            if backend.kind in {"python", "executable"}:
                assert backend.requirements, f"{backend.name} has nothing to look up"
            else:
                assert backend.no_version_reason, f"{backend.name} must explain why it can't"

    def test_engine_names_match_what_wrappers_record(self) -> None:
        """The registry is only useful for provenance if the keys line up.

        These are the ``engine=`` strings molforge's wrappers pass to
        ``Provenance.from_engine`` (dotted ones like ``GROMACS.minimize``
        match on the part before the dot).
        """
        names = {b.name for b in REGISTRY}
        for recorded in (
            "Boltz",
            "Chai1",
            "ESMFold",
            "AlphaFold",
            "RoseTTAFold",
            "Vina",
            "Gnina",
            "DiffDock",
            "fpocket",
            "p2rank",
            "ProteinMPNN",
            "RFdiffusion",
            "ESM-IF1",
        ):
            assert recorded in names, f"wrappers record engine={recorded!r}; registry misses it"
        for dotted in ("OpenMM.run", "GROMACS.minimize", "AMBER.prepare"):
            assert dotted.split(".", 1)[0] in names


class TestFiltering:
    def test_category_filter(self) -> None:
        md = engine_versions(category="md", probe_executables=False)
        assert sorted(md) == ["AMBER", "GROMACS", "OpenMM", "tleap"]
        assert all(b.category == "md" for b in md.values())

    def test_unknown_category_raises(self) -> None:
        """A typo must not read as 'nothing is installed'."""
        with pytest.raises(ValueError, match="unknown category"):
            engine_versions(category="foldin")

    def test_filtering_does_not_mutate_the_memoized_sweep(self) -> None:
        engine_versions(category="md", probe_executables=False)
        assert len(engine_versions(probe_executables=False)) == len(REGISTRY)


class TestPythonBackends:
    def test_installed_distribution_is_reported(self) -> None:
        """molforge is always installed when its own tests run."""
        backend = engine_versions(probe_executables=False)["molforge"]
        assert backend.available
        assert backend.kind == "python"
        assert backend.version
        assert backend.detail == ""

    def test_absent_distribution_says_so_without_raising(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import molforge.versions as versions_module

        monkeypatch.setattr(versions_module, "engine_version", lambda _dist: "")
        backend = engine_versions(probe_executables=False, refresh=True)["Boltz"]
        assert not backend.available
        assert backend.version == ""
        assert "boltz" in backend.detail

    def test_alias_distributions_are_tried_in_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RDKit was ``rdkit-pypi`` before it was ``rdkit``."""
        import molforge.versions as versions_module

        monkeypatch.setattr(
            versions_module,
            "engine_version",
            lambda dist: "2022.03.1" if dist == "rdkit-pypi" else "",
        )
        backend = engine_versions(probe_executables=False, refresh=True)["RDKit"]
        assert backend.available
        assert backend.requirement == "rdkit-pypi"
        assert backend.version == "2022.03.1"


class TestRepoBackends:
    def test_repo_engines_are_undetectable_and_say_why(self) -> None:
        backends = engine_versions(probe_executables=False)
        for name in ("RoseTTAFold", "DiffDock", "ProteinMPNN", "RFdiffusion"):
            backend = backends[name]
            assert backend.kind == "repo"
            assert not backend.available
            assert "repo_dir" in backend.detail


class TestExecutableBackends:
    @staticmethod
    def _patch_which(monkeypatch: pytest.MonkeyPatch, mapping: dict[str, str]) -> None:
        import molforge.versions as versions_module

        monkeypatch.setattr(versions_module.shutil, "which", lambda name: mapping.get(name))

    @staticmethod
    def _patch_run(monkeypatch: pytest.MonkeyPatch, stdout: str = "", stderr: str = "") -> None:
        import molforge.versions as versions_module

        def fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr=stderr)

        monkeypatch.setattr(versions_module.subprocess, "run", fake_run)

    def test_missing_binary_reports_not_on_path(self) -> None:
        backend = engine_versions(probe_executables=False)["Gnina"]
        if backend.available:  # pragma: no cover - only on a machine with gnina
            pytest.skip("gnina is genuinely installed here")
        assert not backend.available
        assert "$PATH" in backend.detail

    def test_present_binary_records_its_location(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_which(monkeypatch, {"gnina": "/opt/bin/gnina"})
        self._patch_run(monkeypatch, stdout="gnina 1.3\n")
        backend = engine_versions(refresh=True)["Gnina"]
        assert backend.available
        assert backend.location == "/opt/bin/gnina"
        assert backend.version == "1.3"

    def test_gromacs_pattern_beats_the_generic_scan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`gmx --version` leads with a banner, not the version."""
        self._patch_which(monkeypatch, {"gmx": "/usr/bin/gmx"})
        self._patch_run(
            monkeypatch,
            stdout=(
                "                  :-) GROMACS - gmx, 2023.3 (-:\n"
                "GROMACS version:    2023.3\n"
                "Precision:          mixed\n"
                "Compiler:           GNU 11.4.0\n"
            ),
        )
        assert engine_versions(refresh=True)["GROMACS"].version == "2023.3"

    def test_version_read_from_stderr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Plenty of tools print their banner to stderr."""
        self._patch_which(monkeypatch, {"gnina": "/opt/bin/gnina"})
        self._patch_run(monkeypatch, stdout="", stderr="gnina 1.0.3\n")
        assert engine_versions(refresh=True)["Gnina"].version == "1.0.3"

    def test_unrecognisable_output_yields_no_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A wrong number in a manifest is worse than no number."""
        self._patch_which(monkeypatch, {"gnina": "/opt/bin/gnina"})
        self._patch_run(monkeypatch, stdout="unrecognised option '--version'\n")
        backend = engine_versions(refresh=True)["Gnina"]
        assert backend.available
        assert backend.version == ""
        assert "nothing recognisable" in backend.detail

    def test_timeout_is_survivable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import molforge.versions as versions_module

        self._patch_which(monkeypatch, {"gnina": "/opt/bin/gnina"})

        def hang(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            raise subprocess.TimeoutExpired(cmd="gnina", timeout=0.1)

        monkeypatch.setattr(versions_module.subprocess, "run", hang)
        backend = engine_versions(refresh=True)["Gnina"]
        assert backend.available
        assert backend.version == ""

    def test_unrunnable_binary_is_survivable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import molforge.versions as versions_module

        self._patch_which(monkeypatch, {"gnina": "/opt/bin/gnina"})

        def boom(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            raise PermissionError("not executable")

        monkeypatch.setattr(versions_module.subprocess, "run", boom)
        assert engine_versions(refresh=True)["Gnina"].version == ""

    def test_no_version_flag_is_distinct_from_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """fpocket is installed but will not say which fpocket it is."""
        import molforge.versions as versions_module

        self._patch_which(monkeypatch, {"fpocket": "/usr/local/bin/fpocket"})

        def never(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            raise AssertionError("must not probe a tool with no version flag")

        monkeypatch.setattr(versions_module.subprocess, "run", never)
        backend = engine_versions(refresh=True)["fpocket"]
        assert backend.available
        assert backend.version == ""
        assert "no version flag" in backend.detail

    def test_probe_executables_false_spawns_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import molforge.versions as versions_module

        self._patch_which(monkeypatch, {"gnina": "/opt/bin/gnina"})

        def never(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            raise AssertionError("probe_executables=False must not spawn a subprocess")

        monkeypatch.setattr(versions_module.subprocess, "run", never)
        backend = engine_versions(probe_executables=False, refresh=True)["Gnina"]
        assert backend.available
        assert backend.version == ""
        assert backend.detail == "probe_executables=False"


class TestMemoization:
    def test_sweep_is_reused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Emitting one manifest per output must not re-probe every binary."""
        import molforge.versions as versions_module

        calls: list[str] = []

        def counting_which(name: str) -> str | None:
            calls.append(name)
            return None

        monkeypatch.setattr(versions_module.shutil, "which", counting_which)
        engine_versions(probe_executables=False)
        first = len(calls)
        engine_versions(probe_executables=False)
        assert len(calls) == first, "second sweep should have come from the memo"

    def test_refresh_re_probes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import molforge.versions as versions_module

        calls: list[str] = []
        monkeypatch.setattr(
            versions_module.shutil, "which", lambda name: calls.append(name) or None
        )
        engine_versions(probe_executables=False)
        first = len(calls)
        engine_versions(probe_executables=False, refresh=True)
        assert len(calls) > first


class TestBackendVersionDict:
    def test_to_dict_is_json_native(self) -> None:
        import json

        payload = engine_versions(probe_executables=False)["molforge"].to_dict()
        assert json.loads(json.dumps(payload)) == payload

    def test_quiet_fields_are_omitted(self) -> None:
        bv = BackendVersion(
            name="X", category="md", kind="python", requirement="x", available=True, version="1.0"
        )
        assert "location" not in bv.to_dict()
        assert "detail" not in bv.to_dict()

    def test_reported_fields_are_present(self) -> None:
        bv = BackendVersion(
            name="X",
            category="md",
            kind="executable",
            requirement="x",
            available=True,
            location="/usr/bin/x",
            detail="no version flag",
        )
        payload = bv.to_dict()
        assert payload["location"] == "/usr/bin/x"
        assert payload["detail"] == "no version flag"


class TestManifestIntegration:
    """The registry fills the blanks a native-binary wrapper leaves."""

    def test_recorded_versions_are_untouched(self) -> None:
        from molforge.core.provenance import Provenance
        from molforge.reproducibility import pipeline_manifest

        prov = Provenance.from_engine("ESMFold", operation="predict", engine_version="1.0.3")
        env = pipeline_manifest(prov).environment
        assert env["engines"] == {"ESMFold": "1.0.3"}
        assert "engines_detected" not in env

    def test_blank_version_is_backfilled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import molforge.reproducibility as repro_module
        from molforge.core.provenance import Provenance
        from molforge.reproducibility import pipeline_manifest

        monkeypatch.setattr(
            repro_module,
            "engine_versions",
            lambda **_kwargs: {
                "GROMACS": BackendVersion(
                    name="GROMACS",
                    category="md",
                    kind="executable",
                    requirement="gmx",
                    available=True,
                    version="2023.3",
                )
            },
        )
        prov = Provenance.from_engine("GROMACS.minimize", operation="minimize")
        env = pipeline_manifest(prov).environment
        assert env["engines"] == {}
        assert env["engines_detected"] == {"GROMACS.minimize": "2023.3"}

    def test_runtime_libraries_are_not_passed_off_as_engines(self) -> None:
        """`molforge.io.fetch` must not be credited with molforge's version.

        It would match the ``molforge`` runtime row on the dotted prefix,
        and restate what ``molforge_version`` already says.
        """
        from molforge.core.provenance import Provenance
        from molforge.reproducibility import pipeline_manifest

        prov = Provenance.from_engine("molforge.io.fetch", operation="fetch")
        env = pipeline_manifest(prov).environment
        assert "engines_detected" not in env

    def test_undetectable_engine_adds_no_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import molforge.reproducibility as repro_module
        from molforge.core.provenance import Provenance
        from molforge.reproducibility import pipeline_manifest

        monkeypatch.setattr(repro_module, "engine_versions", lambda **_kwargs: {})
        prov = Provenance.from_engine("fpocket", operation="detect")
        assert "engines_detected" not in pipeline_manifest(prov).environment
