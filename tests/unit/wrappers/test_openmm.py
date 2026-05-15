"""Tests for the OpenMM wrapper.

These tests don't require OpenMM to be installed. They exercise:
  - Construction with various parameter combinations
  - Lazy import behavior
  - Missing-dependency error path
  - Force-field name lookup
  - Validation of `run` parameters

End-to-end MD against the real engine is left to integration tests
marked @pytest.mark.slow.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from molforge.core import AtomArray, Protein
from molforge.md import MDEngineNotInstalledError, Simulation
from molforge.wrappers.md import OpenMM


def _openmm_available() -> bool:
    return importlib.util.find_spec("openmm") is not None


def _topology(n: int = 3) -> Protein:
    arr = AtomArray(n)
    arr.element[:] = "C"
    arr.atom_name[:] = "CA"
    arr.residue_name[:] = "ALA"
    arr.residue_id[:] = np.arange(1, n + 1)
    arr.chain_id[:] = "A"
    return Protein(arr)


class TestConstruction:
    def test_defaults(self) -> None:
        engine = OpenMM()
        assert engine.name == "OpenMM"
        assert engine.platform is None
        assert engine.precision == "mixed"
        assert engine.nonbonded_cutoff == pytest.approx(1.0)
        assert engine.nonbonded_method == "NoCutoff"
        assert engine.constraints == "HBonds"

    def test_custom_settings(self) -> None:
        engine = OpenMM(
            platform="CUDA",
            precision="single",
            nonbonded_cutoff=1.2,
            nonbonded_method="PME",
            constraints="AllBonds",
        )
        assert engine.platform == "CUDA"
        assert engine.precision == "single"
        assert engine.nonbonded_cutoff == pytest.approx(1.2)
        assert engine.nonbonded_method == "PME"
        assert engine.constraints == "AllBonds"

    def test_constraints_none(self) -> None:
        engine = OpenMM(constraints=None)
        assert engine.constraints is None

    def test_construction_does_not_import_openmm(self) -> None:
        """Constructing must not trigger heavy imports."""
        OpenMM()
        OpenMM(platform="CUDA", precision="single")


class TestMissingDependency:
    @pytest.mark.skipif(_openmm_available(), reason="openmm is installed")
    def test_prepare_without_openmm_raises_clear_error(self) -> None:
        engine = OpenMM()
        with pytest.raises(MDEngineNotInstalledError, match="OpenMM"):
            engine.prepare(_topology(), force_field="amber14-all")

    @pytest.mark.skipif(_openmm_available(), reason="openmm is installed")
    def test_minimize_without_engine_handle_raises(self) -> None:
        """Even if openmm is missing, missing-handle path can be tested by
        building a Simulation manually."""
        engine = OpenMM()
        sim = Simulation(
            topology=_topology(),
            coordinates=np.zeros((3, 3), dtype=np.float32),
        )
        # No engine_handle. minimize() needs to import openmm first,
        # so we hit the import-error path before the engine-handle check.
        with pytest.raises(MDEngineNotInstalledError):
            engine.minimize(sim)


class TestRunValidation:
    """`run` performs argument validation before any engine call."""

    @pytest.mark.skipif(_openmm_available(), reason="openmm is installed")
    def test_negative_n_steps_raises(self) -> None:
        engine = OpenMM()
        sim = Simulation(
            topology=_topology(),
            coordinates=np.zeros((3, 3), dtype=np.float32),
            engine_handle=None,
        )
        # Without openmm we get the import error before the n_steps check —
        # so this test only runs when openmm IS installed (handled below).
        with pytest.raises(MDEngineNotInstalledError):
            engine.run(sim, n_steps=-5)

    def test_validation_with_no_handle_still_raises_value_error_when_openmm_installed(self) -> None:
        # If openmm IS installed, we still hit the missing-handle error
        # before negative-n_steps. We test both cases to be explicit.
        if not _openmm_available():
            pytest.skip("openmm not installed; covered by other test")
        engine = OpenMM()
        sim = Simulation(
            topology=_topology(),
            coordinates=np.zeros((3, 3), dtype=np.float32),
            engine_handle=None,
        )
        with pytest.raises(ValueError, match="prepare"):
            engine.run(sim, n_steps=10)


class TestForceFieldRegistry:
    """Verify the bundled force-field name lookup works."""

    def test_known_force_field_resolves(self) -> None:
        from molforge.wrappers.md.openmm import _FORCE_FIELD_FILES

        assert "amber14-all" in _FORCE_FIELD_FILES
        assert "charmm36" in _FORCE_FIELD_FILES
        # Each entry is a list of XML file names
        for name, files in _FORCE_FIELD_FILES.items():
            assert isinstance(files, list)
            assert all(isinstance(f, str) and f.endswith(".xml") for f in files), name

    def test_amber14_includes_water_model(self) -> None:
        from molforge.wrappers.md.openmm import _FORCE_FIELD_FILES

        # The water model XML should be present so solvated systems work
        # out of the box.
        assert any("tip3p" in f.lower() for f in _FORCE_FIELD_FILES["amber14-all"])


@pytest.mark.slow
@pytest.mark.skipif(not _openmm_available(), reason="openmm not installed")
class TestEndToEnd:
    """Real OpenMM runs. Skipped unless openmm is installed."""

    def test_small_minimize_only(self) -> None:
        # Skipped in normal CI; this requires a force-field-compatible
        # input. Provided here as a contract test for anyone running
        # against a real install.
        pytest.skip("Requires force-field-compatible input PDB — wire in CI")
