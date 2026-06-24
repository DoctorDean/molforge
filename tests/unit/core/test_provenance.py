"""Tests for ``molforge.core.provenance``.

The :class:`Provenance` dataclass is a frozen, JSON-round-trippable
record of "what produced this output." These tests cover the contract
surface: construction, immutability, traversal, JSON round-trip,
strict input validation, and the integration with the metadata key
vocabulary.
"""

from __future__ import annotations

import json

import pytest

from molforge.core import Provenance
from molforge.core import metadata_keys as mk

# ---------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------


class TestConstruction:
    def test_minimal_via_factory(self) -> None:
        """``from_engine`` produces a usable Provenance with just an
        engine name; everything else is defaulted."""
        prov = Provenance.from_engine(engine="ESMFold")
        assert prov.engine == "ESMFold"
        assert prov.engine_version == ""
        assert prov.parameters == {}
        assert prov.inputs == {}
        assert prov.parent is None

    def test_factory_autofills_timestamp(self) -> None:
        prov = Provenance.from_engine(engine="ESMFold")
        # ISO-8601 UTC: matches "YYYY-MM-DDTHH:MM:SS+00:00".
        assert prov.timestamp
        assert "T" in prov.timestamp
        assert prov.timestamp.endswith("+00:00")

    def test_factory_autofills_molforge_version(self) -> None:
        """The molforge version comes from the installed package."""
        import molforge

        prov = Provenance.from_engine(engine="ESMFold")
        assert prov.molforge_version == molforge.__version__

    def test_factory_takes_defensive_copies(self) -> None:
        """Mutating the caller's parameters dict after construction
        must not change the stored Provenance."""
        params = {"x": 1}
        prov = Provenance.from_engine(engine="E", parameters=params)
        params["x"] = 99
        params["new"] = 42
        assert prov.parameters == {"x": 1}

    def test_factory_attaches_parent(self) -> None:
        upstream = Provenance.from_engine(engine="ESMFold")
        downstream = Provenance.from_engine(engine="Vina", parent=upstream)
        assert downstream.parent is upstream

    def test_bare_constructor_does_not_autofill(self) -> None:
        """The bare dataclass constructor leaves fields at their
        defaults — only ``from_engine`` auto-fills timestamp /
        molforge_version. This is intentional so tests and
        deserialisation can reconstruct exact values without
        clobbering."""
        prov = Provenance(engine="X")
        assert prov.timestamp == ""
        assert prov.molforge_version == ""


# ---------------------------------------------------------------------
# Strict JSON validation
# ---------------------------------------------------------------------


class TestValidation:
    def test_non_serialisable_parameter_raises(self) -> None:
        """A Path / NumPy array / arbitrary object in parameters
        fails *at construction time*, not at later serialisation
        time."""
        with pytest.raises(ValueError, match="JSON-serialisable"):
            Provenance.from_engine(engine="X", parameters={"obj": object()})

    def test_non_serialisable_input_raises(self) -> None:
        with pytest.raises(ValueError, match="JSON-serialisable"):
            Provenance.from_engine(engine="X", inputs={"key": object()})

    def test_nested_non_serialisable_raises(self) -> None:
        """Validation must descend into nested structures, not just
        check the top-level keys."""
        with pytest.raises(ValueError, match="JSON-serialisable"):
            Provenance.from_engine(
                engine="X",
                parameters={"nested": {"deep": object()}},
            )

    def test_native_types_all_accepted(self) -> None:
        """str / int / float / bool / None / list / dict all work."""
        prov = Provenance.from_engine(
            engine="X",
            parameters={
                "s": "hello",
                "i": 1,
                "f": 1.5,
                "b": True,
                "n": None,
                "lst": [1, "two", None],
                "d": {"k": "v"},
            },
        )
        assert prov.parameters["s"] == "hello"
        assert prov.parameters["lst"] == [1, "two", None]


# ---------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------


class TestImmutability:
    def test_cannot_mutate_attribute(self) -> None:
        """Frozen dataclass: attribute assignment raises."""
        from dataclasses import FrozenInstanceError

        prov = Provenance.from_engine(engine="X")
        with pytest.raises(FrozenInstanceError):
            prov.engine = "Y"  # type: ignore[misc]

    def test_replace_produces_new_instance(self) -> None:
        upstream = Provenance.from_engine(engine="ESMFold")
        downstream = Provenance.from_engine(engine="Vina")
        amended = downstream.replace(parent=upstream)
        # Original unchanged.
        assert downstream.parent is None
        # Amended has the parent.
        assert amended.parent is upstream
        # Same other fields.
        assert amended.engine == downstream.engine
        assert amended.timestamp == downstream.timestamp


# ---------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------


class TestTraversal:
    def _make_three_step_chain(self) -> Provenance:
        a = Provenance.from_engine(engine="A")
        b = Provenance.from_engine(engine="B", parent=a)
        c = Provenance.from_engine(engine="C", parent=b)
        return c

    def test_walk_newest_first(self) -> None:
        c = self._make_three_step_chain()
        engines = [step.engine for step in c.walk()]
        assert engines == ["C", "B", "A"]

    def test_chain_oldest_first(self) -> None:
        c = self._make_three_step_chain()
        engines = [step.engine for step in c.chain()]
        assert engines == ["A", "B", "C"]

    def test_depth_terminal_is_one(self) -> None:
        prov = Provenance.from_engine(engine="X")
        assert prov.depth == 1

    def test_depth_counts_ancestors(self) -> None:
        c = self._make_three_step_chain()
        assert c.depth == 3

    def test_walk_on_single_step(self) -> None:
        prov = Provenance.from_engine(engine="X")
        steps = list(prov.walk())
        assert len(steps) == 1
        assert steps[0] is prov


# ---------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------


class TestSerialisation:
    def test_to_dict_shape(self) -> None:
        prov = Provenance.from_engine(
            engine="Vina",
            engine_version="1.2.3",
            parameters={"exhaustiveness": 8},
            inputs={"ligand": "/tmp/lig.sdf"},
        )
        d = prov.to_dict()
        assert d["engine"] == "Vina"
        assert d["engine_version"] == "1.2.3"
        assert d["parameters"] == {"exhaustiveness": 8}
        assert d["inputs"] == {"ligand": "/tmp/lig.sdf"}
        assert d["parent"] is None
        # The dict must contain every documented field, including the
        # auto-filled ones.
        assert "molforge_version" in d
        assert "timestamp" in d

    def test_to_dict_emits_nested_parent(self) -> None:
        upstream = Provenance.from_engine(engine="ESMFold")
        downstream = Provenance.from_engine(engine="Vina", parent=upstream)
        d = downstream.to_dict()
        assert isinstance(d["parent"], dict)
        assert d["parent"]["engine"] == "ESMFold"

    def test_to_dict_is_json_serialisable(self) -> None:
        prov = Provenance.from_engine(
            engine="A",
            parameters={"x": 1, "y": [1, 2, 3]},
            parent=Provenance.from_engine(engine="B"),
        )
        # json.dumps should NOT need a default= — values are guaranteed
        # native by construction.
        s = json.dumps(prov.to_dict())
        assert "A" in s
        assert "B" in s

    def test_from_dict_roundtrip(self) -> None:
        original = Provenance.from_engine(
            engine="Vina",
            engine_version="1.2.3",
            parameters={"exhaustiveness": 8},
            inputs={"ligand": "/tmp/lig.sdf"},
            parent=Provenance.from_engine(engine="ESMFold"),
        )
        d = original.to_dict()
        restored = Provenance.from_dict(d)
        assert restored == original

    def test_from_dict_reconstructs_parent_chain(self) -> None:
        a = Provenance.from_engine(engine="A")
        b = Provenance.from_engine(engine="B", parent=a)
        c = Provenance.from_engine(engine="C", parent=b)
        restored = Provenance.from_dict(c.to_dict())
        assert [s.engine for s in restored.chain()] == ["A", "B", "C"]

    def test_from_dict_missing_engine_raises(self) -> None:
        with pytest.raises(ValueError, match="engine"):
            Provenance.from_dict({"engine_version": "x"})

    def test_from_dict_tolerates_missing_optional_fields(self) -> None:
        """A minimal dict (engine only) must reconstruct cleanly, so an
        older on-disk shape continues to load after fields are added."""
        prov = Provenance.from_dict({"engine": "MinimalEngine"})
        assert prov.engine == "MinimalEngine"
        assert prov.parameters == {}
        assert prov.inputs == {}
        assert prov.parent is None
        assert prov.engine_version == ""

    def test_json_roundtrip(self) -> None:
        original = Provenance.from_engine(
            engine="A",
            parameters={"x": 1},
            parent=Provenance.from_engine(engine="B"),
        )
        text = original.to_json()
        # Two-step decode: text -> Provenance.
        restored = Provenance.from_json(text)
        assert restored == original
        # And the JSON itself is valid.
        parsed = json.loads(text)
        assert parsed["engine"] == "A"

    def test_json_compact(self) -> None:
        prov = Provenance.from_engine(engine="X")
        # indent=None gives compact JSON.
        text = prov.to_json(indent=None)
        assert "\n" not in text


# ---------------------------------------------------------------------
# Equality
# ---------------------------------------------------------------------


class TestEquality:
    def test_equal_when_all_fields_equal(self) -> None:
        a = Provenance(engine="X", timestamp="2024-01-01T00:00:00+00:00")
        b = Provenance(engine="X", timestamp="2024-01-01T00:00:00+00:00")
        assert a == b

    def test_unequal_when_timestamps_differ(self) -> None:
        a = Provenance(engine="X", timestamp="2024-01-01T00:00:00+00:00")
        b = Provenance(engine="X", timestamp="2024-01-02T00:00:00+00:00")
        assert a != b

    def test_unequal_when_parameters_differ(self) -> None:
        a = Provenance(engine="X", parameters={"a": 1})
        b = Provenance(engine="X", parameters={"a": 2})
        assert a != b

    def test_unequal_across_engines(self) -> None:
        a = Provenance(engine="X")
        b = Provenance(engine="Y")
        assert a != b


# ---------------------------------------------------------------------
# Integration with metadata vocabulary
# ---------------------------------------------------------------------


class TestMetadataIntegration:
    """The intended use is ``protein.metadata[mk.PROVENANCE] = prov``.
    These tests confirm the key is documented and the attach pattern
    works through Protein.metadata as a plain dict."""

    def test_provenance_key_is_documented(self) -> None:
        assert mk.PROVENANCE == "provenance"
        assert mk.PROVENANCE in mk.DOCUMENTED_KEYS

    def test_attaches_to_protein_metadata(self) -> None:
        from molforge.core import AtomArray, Protein

        prot = Protein(AtomArray(0))
        prov = Provenance.from_engine(engine="ESMFold")
        prot.metadata[mk.PROVENANCE] = prov

        attached = prot.metadata[mk.PROVENANCE]
        assert isinstance(attached, Provenance)
        assert attached.engine == "ESMFold"

    def test_protein_metadata_typed_view_accepts_provenance(self) -> None:
        """The ProteinMetadata TypedDict declares provenance: Any so
        annotating a dict literal with the type doesn't fight us."""
        from molforge.core import ProteinMetadata

        prov = Provenance.from_engine(engine="ESMFold")
        meta: ProteinMetadata = {"provenance": prov}
        # Just exercise the type — mypy would catch a real mismatch.
        assert meta["provenance"] is prov
