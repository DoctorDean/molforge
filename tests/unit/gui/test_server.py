"""The editor's HTTP API.

Driven through FastAPI's test client, so these exercise the real routing,
serialization and status codes rather than calling the handlers directly.
Skipped wholesale when the ``gui`` extra isn't installed — the API is
opt-in, and the catalog and graph layers it sits on are tested without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

    from fastapi.testclient import TestClient

pytest.importorskip("fastapi", reason="the editor API needs the `gui` extra")
pytest.importorskip("httpx", reason="the FastAPI test client needs httpx")


@pytest.fixture
def client() -> Iterator[TestClient]:
    from fastapi.testclient import TestClient

    from molforge.gui.server import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def fake_engines() -> Iterator[None]:
    """Register engines the catalog will report as available.

    The API builds against the *live* catalog, so a test that wants a run
    to start needs a block that can actually run here — otherwise the
    assertion under test is just "nothing is installed on this machine".
    """
    from molforge import plugins

    class Folder:
        name = "APIFolder"

        def __init__(self, *, temperature: float = 1.0) -> None:
            self.temperature = temperature

        def predict(self, sequence: str, **_kwargs: object) -> str:
            return f"structure({sequence})"

    plugins.register_engine("APIFolder", Folder)
    yield
    plugins.clear()


def _fold_graph(sequence: str = "MKTV") -> dict[str, Any]:
    return {
        "graph": {
            "nodes": [
                {
                    "id": "fold",
                    "block_id": "APIFolder.predict",
                    "inputs": {"sequence": sequence},
                    "parameters": {},
                }
            ],
            "edges": [],
        }
    }


class TestBlocks:
    def test_it_lists_blocks_with_their_wiring(self, client: TestClient) -> None:
        payload = client.get("/api/blocks").json()
        assert payload["blocks"]
        assert payload["port_types"]
        block = payload["blocks"][0]
        assert {"id", "engine", "operation", "inputs", "output", "available"} <= set(block)

    def test_unavailable_blocks_are_listed_with_a_reason(self, client: TestClient) -> None:
        """Hiding them would be worse: knowing molforge drives Gnina is
        worth more than a palette that pretends it doesn't."""
        blocks = client.get("/api/blocks").json()["blocks"]
        for block in blocks:
            if not block["available"]:
                assert block["detail"]


class TestValidate:
    def test_a_coherent_graph_validates(self, client: TestClient, fake_engines: None) -> None:
        body = client.post("/api/graph/validate", json=_fold_graph()).json()
        assert body["valid"]
        assert body["runnable"]

    def test_a_broken_graph_names_the_node(self, client: TestClient, fake_engines: None) -> None:
        graph = _fold_graph()
        graph["graph"]["nodes"][0]["inputs"] = {}
        body = client.post("/api/graph/validate", json=graph).json()
        assert not body["valid"]
        assert body["node_id"] == "fold"
        assert "sequence" in body["error"]

    def test_valid_but_not_runnable_here_is_a_distinct_answer(self, client: TestClient) -> None:
        """A graph drawn for a cluster is savable and not runnable, and the
        editor has to be able to tell the user which."""
        graph = {
            "graph": {
                "nodes": [
                    {
                        "id": "x",
                        "block_id": "RFdiffusion.generate",
                        "inputs": {},
                        "parameters": {},
                    }
                ],
                "edges": [],
            }
        }
        body = client.post("/api/graph/validate", json=graph).json()
        if body["valid"]:  # RFdiffusion is a repo checkout, absent in CI
            assert not body["runnable"]
            assert body["node_id"] == "x"

    def test_a_malformed_payload_is_a_400_not_a_500(self, client: TestClient) -> None:
        response = client.post("/api/graph/validate", json={"graph": {"nodes": [{}]}})
        assert response.status_code == 400
        assert "malformed" in response.json()["detail"]["error"]


class TestPipelineRoundTrip:
    def test_saving_returns_a_manifest(self, client: TestClient, fake_engines: None) -> None:
        body = client.post("/api/pipeline/save", json=_fold_graph()).json()
        assert '"molforge_pipeline"' in body["json"]
        assert body["context"] == {"sequence": "MKTV"}

    def test_a_saved_pipeline_loads_back_as_the_same_graph(
        self, client: TestClient, fake_engines: None
    ) -> None:
        """The round-trip the editor's save and open buttons depend on."""
        saved = client.post("/api/pipeline/save", json=_fold_graph()).json()
        loaded = client.post("/api/pipeline/load", json={"text": saved["json"]}).json()
        nodes = loaded["graph"]["nodes"]
        assert [n["block_id"] for n in nodes] == ["APIFolder.predict"]
        assert nodes[0]["inputs"]["sequence"] == "MKTV"

    def test_loading_something_that_isnt_a_manifest(self, client: TestClient) -> None:
        response = client.post("/api/pipeline/load", json={"text": "not a manifest"})
        assert response.status_code == 400

    def test_loading_a_manifest_with_an_unknown_engine(self, client: TestClient) -> None:
        manifest = (
            '{"molforge_pipeline": 1, "environment": {}, '
            '"steps": [{"step": 1, "engine": "Nope", "operation": "predict"}]}'
        )
        response = client.post("/api/pipeline/load", json={"text": manifest})
        assert response.status_code == 400
        assert "no block for" in response.json()["detail"]["error"]


class TestRuns:
    def test_a_run_completes_and_reports_each_step(
        self, client: TestClient, fake_engines: None
    ) -> None:
        started = client.post("/api/runs", json=_fold_graph())
        assert started.status_code == 200
        run_id = started.json()["id"]

        body = _await_run(client, run_id)
        assert body["state"] == "succeeded"
        assert [s["node_id"] for s in body["steps"]] == ["fold"]
        assert body["steps"][0]["output"] == "structure(MKTV)"

    def test_an_unrunnable_graph_is_refused_before_it_starts(
        self, client: TestClient, fake_engines: None
    ) -> None:
        graph = _fold_graph()
        graph["graph"]["nodes"][0]["inputs"] = {}
        response = client.post("/api/runs", json=graph)
        assert response.status_code == 400
        assert response.json()["detail"]["node_id"] == "fold"

    def test_polling_an_unknown_run(self, client: TestClient) -> None:
        assert client.get("/api/runs/nope").status_code == 404

    def test_cancelling_an_unknown_run(self, client: TestClient) -> None:
        assert client.post("/api/runs/nope/cancel").status_code == 404

    def test_runs_are_listed(self, client: TestClient, fake_engines: None) -> None:
        run_id = client.post("/api/runs", json=_fold_graph()).json()["id"]
        _await_run(client, run_id)
        assert run_id in {r["id"] for r in client.get("/api/runs").json()["runs"]}


class TestPage:
    def test_a_missing_frontend_explains_itself(self, client: TestClient) -> None:
        """A source checkout has no built assets until someone builds them,
        and a 404 is a poor way to learn that."""
        response = client.get("/")
        assert response.status_code == 200
        assert "frontend isn't built" in response.text

    def test_a_built_frontend_is_served(self, tmp_path: Any) -> None:
        from fastapi.testclient import TestClient

        from molforge.gui.server import create_app

        (tmp_path / "index.html").write_text("<h1>editor</h1>")
        with TestClient(create_app(static_dir=tmp_path)) as built:
            assert "editor" in built.get("/").text


def _await_run(client: TestClient, run_id: str, *, tries: int = 100) -> dict[str, Any]:
    """Poll until the run settles — the same loop the editor runs."""
    import time

    for _ in range(tries):
        body = client.get(f"/api/runs/{run_id}").json()
        if body["state"] != "running":
            return body
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} never finished")
