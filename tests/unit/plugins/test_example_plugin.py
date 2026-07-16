"""Verify the reference example plugin under ``plugins/example_plugin/``.

This guards the copy-paste template against rot: if its entry-point wiring
or its engine's contract breaks (as it silently did while it still imported
the pre-rename ``biocore`` namespace), this test fails. It exercises the
plugin without pip-installing it — the source dir is put on ``sys.path`` and
``register()`` is called directly into an isolated registry.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from molforge.core import Protein
from molforge.core import metadata_keys as mk
from molforge.core.provenance import Provenance
from molforge.plugins import available, clear, get

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType

_PLUGIN_ROOT = Path(__file__).resolve().parents[3] / "plugins" / "example_plugin"
_PLUGIN_SRC = _PLUGIN_ROOT / "src"


@pytest.fixture
def example_plugin(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """Import the example plugin from its source tree and isolate the registry."""
    monkeypatch.syspath_prepend(str(_PLUGIN_SRC))
    sys.modules.pop("example_plugin", None)
    clear()
    import example_plugin

    yield example_plugin
    clear()
    sys.modules.pop("example_plugin", None)


class TestRegistration:
    def test_register_adds_the_engine(self, example_plugin: ModuleType) -> None:
        example_plugin.register()
        assert "example" in available("engine")

    def test_registered_factory_is_the_engine_class(self, example_plugin: ModuleType) -> None:
        example_plugin.register()
        assert get("engine", "example") is example_plugin.ExtendedChainFolder


class TestEngineContract:
    def test_predict_returns_a_valid_protein(self, example_plugin: ModuleType) -> None:
        example_plugin.register()
        engine = get("engine", "example")()
        protein = engine.predict("MKTAYIAKQR")
        assert isinstance(protein, Protein)
        assert protein.n_residues == 10

    def test_predict_sets_uniform_confidence_metadata(self, example_plugin: ModuleType) -> None:
        engine = example_plugin.ExtendedChainFolder()
        protein = engine.predict("ACDEFGHIK")
        assert protein.metadata[mk.ENGINE] == "ExtendedChainFolder"
        assert protein.metadata[mk.MEAN_CONFIDENCE] == pytest.approx(50.0)
        assert protein.metadata[mk.CONFIDENCE_PER_RESIDUE].shape == (9,)

    def test_predict_attaches_provenance(self, example_plugin: ModuleType) -> None:
        engine = example_plugin.ExtendedChainFolder()
        protein = engine.predict("ACDEF")
        prov = protein.metadata[mk.PROVENANCE]
        assert isinstance(prov, Provenance)
        assert prov.engine == "ExtendedChainFolder"
        assert prov.inputs == {"sequence": "ACDEF"}

    def test_predict_rejects_non_letters(self, example_plugin: ModuleType) -> None:
        engine = example_plugin.ExtendedChainFolder()
        with pytest.raises(ValueError, match="letters"):
            engine.predict("ACD3F")


class TestDeclaredWiring:
    def test_pyproject_declares_the_entry_point(self) -> None:
        # String-check (not tomllib) so this runs on the 3.10 floor too.
        text = (_PLUGIN_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert '[project.entry-points."molforge.plugins"]' in text
        assert 'example = "example_plugin:register"' in text

    def test_entry_point_target_resolves(self, example_plugin: ModuleType) -> None:
        # The "example_plugin:register" the pyproject points at must exist,
        # be callable, and actually register the engine when called.
        assert callable(example_plugin.register)
        example_plugin.register()
        assert "example" in available("engine")
