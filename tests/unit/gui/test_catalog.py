"""The block catalog is derived, so its tests assert on shape, not content.

A test that spelled out which engines to expect would be the same
hand-written list that let ``Chai1`` sit in the version registry unchallenged
— it would pass while describing a catalog nobody built. These check the
invariants a node editor actually depends on: ids are addressable, ports
are wireable, parameters are renderable, and building the palette stays
cheap.
"""

from __future__ import annotations

import json
import sys

import pytest

from molforge.gui.catalog import (
    _OPERATIONS,
    _RETURN_PORTS,
    BlockSpec,
    build_catalog,
    catalog_to_dict,
)

#: Widget kinds BlockParameter documents. A kind outside this set is a
#: parameter the frontend has no renderer for.
_KINDS = {"choice", "bool", "int", "float", "vector3", "path", "text"}


@pytest.fixture(scope="module")
def catalog() -> tuple[BlockSpec, ...]:
    return build_catalog()


class TestCatalogShape:
    def test_it_is_not_empty(self, catalog: tuple[BlockSpec, ...]) -> None:
        """A catalog that silently builds to nothing would render as an empty
        palette, which looks like a frontend bug rather than a backend one."""
        assert len(catalog) > 5

    def test_ids_are_unique(self, catalog: tuple[BlockSpec, ...]) -> None:
        ids = [b.id for b in catalog]
        assert len(ids) == len(set(ids)), "the editor addresses blocks by id"

    def test_ids_name_their_engine_and_operation(self, catalog: tuple[BlockSpec, ...]) -> None:
        """An id has to round-trip to the thing that will run it."""
        for block in catalog:
            assert block.id == f"{block.engine}.{block.operation}"

    def test_every_operation_can_be_replayed(self, catalog: tuple[BlockSpec, ...]) -> None:
        """The whole point of the GUI is that a graph runs through replay().
        A block whose operation has no handler is one nobody can execute."""
        from molforge.reproducibility import _REPLAY_HANDLERS

        for block in catalog:
            assert block.operation in _REPLAY_HANDLERS, f"{block.id} can't be replayed"

    def test_every_engine_resolves(self, catalog: tuple[BlockSpec, ...]) -> None:
        from molforge.reproducibility import _resolve_engine

        for block in catalog:
            assert _resolve_engine(block.engine) is not None


class TestPorts:
    def test_outputs_are_known_types(self, catalog: tuple[BlockSpec, ...]) -> None:
        """An ``unknown`` output is a block the editor will not wire — it
        means an operation returns something the port map has never been
        told about, which is a gap to close, not a block to ship."""
        unknown = [b.id for b in catalog if b.output.type == "unknown"]
        assert not unknown, f"unmapped return types: {unknown}"

    def test_inputs_are_known_types(self, catalog: tuple[BlockSpec, ...]) -> None:
        known = set(_RETURN_PORTS.values()) | {"sequence", "ligand", "complex_spec", "pdb_id"}
        for block in catalog:
            for port in block.inputs:
                assert port.type in known, f"{block.id}.{port.name}: {port.type}"

    def test_a_graph_can_be_wired_end_to_end(self, catalog: tuple[BlockSpec, ...]) -> None:
        """Every non-source block's required inputs must be satisfiable by
        some other block's output, or the palette contains a dead end."""
        produced = {b.output.type for b in catalog}
        # Types a user supplies directly rather than wiring from a block.
        supplied = {"sequence", "ligand", "pdb_id", "complex_spec"}
        for block in catalog:
            for port in block.inputs:
                if not port.required:
                    continue
                assert port.type in produced | supplied, (
                    f"{block.id} needs {port.type}, which nothing produces or supplies"
                )

    def test_docking_takes_a_receptor_and_a_ligand(self, catalog: tuple[BlockSpec, ...]) -> None:
        """One concrete shape worth pinning: docking is the operation whose
        two-input wiring the canvas exists to make obvious."""
        dockers = [b for b in catalog if b.operation == "dock"]
        assert dockers
        for block in dockers:
            assert [p.type for p in block.inputs] == ["structure", "ligand"]
            assert block.output.type == "docking_result"


class TestParameters:
    def test_kinds_are_renderable(self, catalog: tuple[BlockSpec, ...]) -> None:
        for block in catalog:
            for param in block.parameters:
                assert param.kind in _KINDS, f"{block.id}.{param.name}: {param.kind}"

    def test_no_internal_parameters_leak(self, catalog: tuple[BlockSpec, ...]) -> None:
        """``self`` and the ``**_kwargs`` sinks are not knobs."""
        for block in catalog:
            names = [p.name for p in block.parameters]
            assert "self" not in names
            assert not [n for n in names if n.startswith("_")]

    def test_choices_are_populated(self, catalog: tuple[BlockSpec, ...]) -> None:
        """A dropdown with no options is worse than a text field."""
        for block in catalog:
            for param in block.parameters:
                if param.kind == "choice":
                    assert param.choices, f"{block.id}.{param.name} is an empty dropdown"

    def test_a_literal_becomes_its_own_options(self, catalog: tuple[BlockSpec, ...]) -> None:
        """Parsed from the annotation, so the options can't drift from what
        the wrapper accepts."""
        choices = {
            (b.id, p.name): p.choices for b in catalog for p in b.parameters if p.kind == "choice"
        }
        assert choices, "no Literal-typed parameter was found to check"
        for (block_id, name), options in choices.items():
            assert len(options) >= 2, f"{block_id}.{name} has one option"
            assert all(options), f"{block_id}.{name} has a blank option"

    def test_defaults_are_json_safe(self, catalog: tuple[BlockSpec, ...]) -> None:
        """Defaults cross the wire as-is; a tuple default (the docking box)
        has to arrive as a list or the editor sees a malformed payload."""
        for block in catalog:
            for param in block.parameters:
                assert not isinstance(param.default, tuple)
                json.dumps(param.default)

    def test_optional_is_distinct_from_defaulted(self, catalog: tuple[BlockSpec, ...]) -> None:
        """``X | None`` means "let the engine decide", which is a different
        state from a knob sitting on its default."""
        optional = [p for b in catalog for p in b.parameters if p.optional]
        assert optional
        assert all(p.default is None for p in optional)


class TestAvailability:
    def test_it_matches_the_environment_sweep(self, catalog: tuple[BlockSpec, ...]) -> None:
        from molforge import engine_versions

        env = engine_versions(probe_executables=False)
        for block in catalog:
            if block.engine in env:
                assert block.available == env[block.engine].available, block.id

    def test_unavailable_blocks_explain_themselves(self, catalog: tuple[BlockSpec, ...]) -> None:
        """Greying a block out without saying why just looks broken."""
        for block in catalog:
            if not block.available:
                assert block.detail, f"{block.id} is unavailable with no reason"

    def test_filtering_drops_only_the_unavailable(self) -> None:
        full = build_catalog()
        runnable = build_catalog(include_unavailable=False)
        assert all(b.available for b in runnable)
        assert {b.id for b in runnable} <= {b.id for b in full}


class TestPayload:
    def test_it_serializes(self, catalog: tuple[BlockSpec, ...]) -> None:
        payload = json.loads(json.dumps(catalog_to_dict(catalog)))
        assert len(payload["blocks"]) == len(catalog)

    def test_port_types_cover_every_port_in_use(self, catalog: tuple[BlockSpec, ...]) -> None:
        """Shipped with the blocks so the frontend validates wires against
        the vocabulary the backend derived, not a copy of its own."""
        payload = catalog_to_dict(catalog)
        declared = set(payload["port_types"])
        for block in catalog:
            assert block.output.type in declared
            assert all(p.type in declared for p in block.inputs)


class TestCost:
    def test_building_imports_no_heavy_dependency(self) -> None:
        """A dashboard rebuilds the catalog on every page load. Reading
        annotations as text rather than resolving them is what keeps that
        from importing torch, so the test guards the mechanism, not just
        the speed."""
        heavy = {"torch", "chai_lab", "boltz", "openmm", "vina"}
        already = heavy & set(sys.modules)
        build_catalog()
        newly_imported = (heavy & set(sys.modules)) - already
        assert not newly_imported, f"building the catalog imported {newly_imported}"

    def test_building_spawns_no_subprocess(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``probe_executables=False``: running ``gnina --version`` to fill in
        a palette would be a poor trade for a string no block needs."""
        import subprocess

        def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("the catalog must not shell out")

        monkeypatch.setattr(subprocess, "run", forbidden)
        monkeypatch.setattr(subprocess, "check_output", forbidden)
        from molforge import engine_versions

        engine_versions(refresh=True, probe_executables=False)
        build_catalog()


class TestOperationTable:
    def test_each_catalogued_operation_has_a_handler(self) -> None:
        """The table is a subset of the replay handlers by design; an entry
        with no handler would produce blocks that cannot run."""
        from molforge.reproducibility import _REPLAY_HANDLERS

        for op in _OPERATIONS:
            assert op.operation in _REPLAY_HANDLERS

    def test_operations_are_unique(self) -> None:
        assert len({op.operation for op in _OPERATIONS}) == len(_OPERATIONS)
