"""The editor's HTTP API, and the command that serves it.

A thin layer over the three modules that do the work: the catalog says
what blocks exist, the graph validates and translates them, the runner
executes. Everything here is plumbing — JSON in, JSON out, and the
mapping from a :class:`~molforge.gui.graph.GraphError` to a 400 with the
node to highlight.

    pip install "molforge[gui]"
    molforge-gui

Three deliberate limits, all of them because this is a *local* tool:

**It binds to localhost.** The API runs engines, reads and writes files at
paths the client supplies, and has no authentication. That combination is
a remote code execution service, so the default host is ``127.0.0.1`` and
``--host`` carries a warning rather than being a neutral option.

**It holds runs in memory.** A dashboard someone leaves open for a week
would otherwise accumulate every structure it ever produced; records keep
descriptions, not objects, and a restart clears them. Pipelines are saved
as ``pipeline.yaml`` — that file is the durable artifact, not the server's
memory.

**One run at a time.** Enforced here rather than left to the client,
because the client is a browser tab someone can double-click a button in,
and the thing on the other end is a GPU.
"""

from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Any

from molforge.gui.catalog import build_catalog, catalog_to_dict
from molforge.gui.graph import GraphError, PipelineGraph
from molforge.gui.runner import RunRecord

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from fastapi import FastAPI

__all__ = ["create_app", "main", "serve"]

#: Where the built frontend lands. Absent in a source checkout until the
#: frontend is built, which is why its absence is a served page rather
#: than a crash.
_STATIC = Path(__file__).parent / "static"

_MISSING_DEPS = (
    "The molforge editor needs FastAPI and uvicorn. Install them with:\n"
    '    pip install "molforge[gui]"'
)

_NO_FRONTEND = """<!doctype html>
<meta charset="utf-8">
<title>molforge editor</title>
<style>
 body { font: 15px/1.6 ui-sans-serif, system-ui, sans-serif; margin: 6rem auto; max-width: 34rem;
        color: #1a1a1a; padding: 0 1.5rem; }
 code { background: #f4f4f5; padding: .15em .4em; border-radius: 4px; }
 pre  { background: #f4f4f5; padding: 1rem; border-radius: 8px; overflow-x: auto; }
</style>
<h1>The frontend isn't built</h1>
<p>The API is running — <a href="/api/blocks">/api/blocks</a> answers — but the
editor's JavaScript hasn't been compiled into this install.</p>
<p>From a source checkout:</p>
<pre>cd gui &amp;&amp; npm install &amp;&amp; npm run build</pre>
<p>A wheel from PyPI ships the built assets, so this page means you're running
from source.</p>
"""


def create_app(*, static_dir: Path | None = None) -> FastAPI:
    """Build the FastAPI application.

    Args:
        static_dir: Where the built frontend lives. Defaults to the
            package's own ``static/``; overridden by the dev server, which
            serves assets from the Vite build directory instead.

    Returns:
        The application, ready to serve.

    Raises:
        ImportError: If the ``gui`` extra isn't installed. Raised here
            rather than at import time so that
            :mod:`molforge.gui.catalog` — which needs nothing beyond
            molforge itself — stays importable.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as e:  # pragma: no cover - exercised by the extra being absent
        raise ImportError(_MISSING_DEPS) from e

    app = FastAPI(title="molforge editor", version=_version())
    runs: dict[str, RunRecord] = {}

    def _graph(payload: dict[str, Any]) -> PipelineGraph:
        """Parse a graph, turning a malformed one into a 400 rather than a 500."""
        try:
            return PipelineGraph.from_dict(payload.get("graph") or payload)
        except GraphError as e:
            raise HTTPException(status_code=400, detail=_problem(e)) from e

    # ---------- what can be built ----------

    @app.get("/api/blocks")
    def blocks() -> dict[str, Any]:
        """Every block, with its ports, parameters, and whether it can run here."""
        return catalog_to_dict()

    # ---------- checking and saving ----------

    @app.post("/api/graph/validate")
    def validate(payload: dict[str, Any]) -> dict[str, Any]:
        """Check a graph without running it.

        Availability is *not* required: this answers "is this a coherent
        pipeline", which a graph drawn here for a cluster elsewhere can be.
        ``runnable`` answers the narrower question separately, so the
        editor can show a valid-but-not-here graph as savable.
        """
        graph = _graph(payload)
        catalog = build_catalog()
        try:
            graph.validate(catalog=catalog)
        except GraphError as e:
            return {"valid": False, "runnable": False, **_problem(e)}
        try:
            graph.validate(catalog=catalog, require_available=True)
        except GraphError as e:
            return {"valid": True, "runnable": False, **_problem(e)}
        return {"valid": True, "runnable": True, "error": "", "node_id": ""}

    @app.post("/api/pipeline/save")
    def save(payload: dict[str, Any]) -> dict[str, Any]:
        """Serialize a graph to the pipeline manifest, as YAML and JSON.

        Returned as text rather than written to disk: the server has no
        business choosing a path in the user's filesystem, and the browser
        already knows how to save a file it was handed.
        """
        manifest, context = _graph(payload).to_manifest()
        try:
            yaml = manifest.to_yaml()
        except ImportError:
            # PyYAML is the `repro` extra. JSON replays identically, so a
            # missing optional dependency costs a format, not the feature.
            yaml = ""
        return {"yaml": yaml, "json": manifest.to_json(), "context": context}

    @app.post("/api/pipeline/load")
    def load(payload: dict[str, Any]) -> dict[str, Any]:
        """Open a ``pipeline.yaml`` — or its JSON form — as a graph."""
        from molforge.reproducibility import PipelineManifest

        text = str(payload.get("text") or "")
        try:
            manifest = (
                PipelineManifest.from_json(text)
                if text.lstrip().startswith("{")
                else PipelineManifest.from_yaml(text)
            )
        except ImportError as e:
            raise HTTPException(status_code=400, detail={"error": str(e), "node_id": ""}) from e
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={"error": f"that isn't a pipeline manifest: {e}", "node_id": ""},
            ) from e
        try:
            return {"graph": PipelineGraph.from_manifest(manifest).to_dict()}
        except GraphError as e:
            raise HTTPException(status_code=400, detail=_problem(e)) from e

    # ---------- running ----------

    @app.post("/api/runs")
    def start_run(payload: dict[str, Any]) -> dict[str, Any]:
        """Start a run. 409 if one is already going.

        The limit is enforced here because the client is a browser tab
        someone can double-click a button in, and the thing on the other
        end is a GPU.
        """
        if any(r.running for r in runs.values()):
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "a run is already going; wait for it or cancel it.",
                    "node_id": "",
                },
            )
        graph = _graph(payload)
        try:
            record = RunRecord.start(graph, context=payload.get("context") or None)
        except GraphError as e:
            raise HTTPException(status_code=400, detail=_problem(e)) from e
        runs[record.id] = record
        return record.to_dict()

    @app.get("/api/runs")
    def list_runs() -> dict[str, Any]:
        return {"runs": [r.to_dict() for r in runs.values()]}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        """Poll one run. The editor's per-block status comes from here."""
        record = runs.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail={"error": "no such run", "node_id": ""})
        return record.to_dict()

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str) -> dict[str, Any]:
        """Ask a run to stop after the step in flight."""
        record = runs.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail={"error": "no such run", "node_id": ""})
        record.cancel()
        return record.to_dict()

    # ---------- the page ----------

    static = _STATIC if static_dir is None else static_dir
    if (static / "index.html").exists():
        app.mount("/", StaticFiles(directory=static, html=True), name="static")
    else:

        @app.get("/", response_class=HTMLResponse)
        def placeholder() -> str:
            """Explain the missing frontend instead of 404-ing.

            A source checkout has no built assets until someone runs the
            frontend build, and "404" is a poor way to learn that.
            """
            return _NO_FRONTEND

    return app


def _problem(error: GraphError) -> dict[str, str]:
    """A graph failure as the editor consumes it: a message, and a node to blame."""
    return {"error": str(error), "node_id": error.node_id}


def _version() -> str:
    from molforge import __version__

    return str(__version__)


def serve(
    *, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True, reload: bool = False
) -> None:
    """Run the editor.

    Args:
        host: Interface to bind. Defaults to loopback, and should stay
            there — see the module docstring.
        port: Port to bind.
        open_browser: Open the page once the server is up.
        reload: Reload on source changes, for working on the server itself.
    """
    try:
        import uvicorn
    except ImportError as e:  # pragma: no cover - exercised by the extra being absent
        raise ImportError(_MISSING_DEPS) from e

    url = f"http://{'localhost' if host == '127.0.0.1' else host}:{port}"
    if host != "127.0.0.1":
        print(
            f"warning: binding to {host}, not loopback. This API runs engines and "
            "reads files the client names, with no authentication. Only do this on "
            "a network you trust."
        )
    print(f"molforge editor → {url}")

    if open_browser:
        webbrowser.open(url)
    uvicorn.run(create_app(), host=host, port=port, reload=reload, log_level="warning")


def main(argv: list[str] | None = None) -> int:
    """``molforge-gui`` entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="molforge-gui", description="Draw, save and run molforge pipelines."
    )
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind (default: loopback)")
    parser.add_argument("--port", type=int, default=8765, help="port to bind (default: 8765)")
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser")
    parser.add_argument("--reload", action="store_true", help="reload on source changes")
    args = parser.parse_args(argv)

    try:
        serve(
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
            reload=args.reload,
        )
    except ImportError as e:
        print(e)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
