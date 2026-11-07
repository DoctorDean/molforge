/**
 * The editor.
 *
 * Holds the graph, mirrors it to the API for validation and running, and
 * keeps one rule above all others: the canvas is a view of a pipeline
 * manifest, not a thing of its own. Every button here maps to one server
 * call that works on the same `pipeline.yaml` a script would.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  addEdge,
  Background,
  Controls,
  MiniMap,
  ReactFlow,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
} from "@xyflow/react";
import type { Connection, Edge, Node } from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { ApiError, api } from "./api";
import type { BlockSpec, GraphPayload, RunRecord } from "./api";
import { BlockNode } from "./components/BlockNode";
import type { BlockNodeData } from "./components/BlockNode";
import { Inspector } from "./components/Inspector";
import { Palette } from "./components/Palette";
import { portColor } from "./ports";

const NODE_TYPES = { block: BlockNode };

/** How often a live run is polled. Fast enough to feel immediate, slow
 *  enough that a pipeline of hour-long steps isn't answering a request
 *  every frame for no new information. */
const POLL_MS = 600;

export default function App() {
  return (
    <ReactFlowProvider>
      <Editor />
    </ReactFlowProvider>
  );
}

function Editor() {
  const [blocks, setBlocks] = useState<BlockSpec[]>([]);
  const [loadError, setLoadError] = useState("");
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>({ kind: "idle", message: "" });
  const [run, setRun] = useState<RunRecord | null>(null);
  const counter = useRef(0);

  useEffect(() => {
    api
      .blocks()
      .then((catalog) => setBlocks(catalog.blocks))
      .catch((e) => setLoadError(String(e.message ?? e)));
  }, []);

  const byId = useMemo(
    () => new Map(blocks.map((block) => [block.id, block])),
    [blocks],
  );

  // ---------- the graph, as the API wants it ----------

  const payload = useCallback((): GraphPayload => {
    return {
      nodes: nodes.map((node) => {
        const data = node.data as BlockNodeData;
        return {
          id: node.id,
          block_id: data.block.id,
          parameters: data.parameters,
          inputs: data.inputs,
          position: [node.position.x, node.position.y] as [number, number],
        };
      }),
      edges: edges.map((edge) => ({
        source: edge.source,
        target: edge.target,
        target_port: edge.targetHandle ?? "",
      })),
    };
  }, [nodes, edges]);

  // ---------- editing ----------

  const addBlock = useCallback(
    (block: BlockSpec) => {
      const id = `${block.engine.replace(/\W+/g, "")}-${++counter.current}`;
      const data: BlockNodeData = {
        block,
        inputs: {},
        parameters: Object.fromEntries(
          block.parameters
            .filter((p) => p.default !== null && p.default !== undefined)
            .map((p) => [p.name, p.default]),
        ),
        wired: [],
      };
      setNodes((current) => [
        ...current,
        {
          id,
          type: "block",
          position: {
            x: 80 + (current.length % 4) * 300,
            y: 90 + Math.floor(current.length / 4) * 220,
          },
          data: data as unknown as Record<string, unknown>,
        },
      ]);
      setSelected(id);
    },
    [setNodes],
  );

  /** Refuse a wire whose types don't match, before it is ever drawn. */
  const isValidConnection = useCallback(
    (connection: Connection | Edge) => {
      const source = nodes.find((n) => n.id === connection.source);
      const target = nodes.find((n) => n.id === connection.target);
      if (!source || !target || source.id === target.id) return false;
      const produced = (source.data as BlockNodeData).block.output.type;
      const port = (target.data as BlockNodeData).block.inputs.find(
        (p) => p.name === connection.targetHandle,
      );
      return Boolean(port) && port!.type === produced;
    },
    [nodes],
  );

  const onConnect = useCallback(
    (connection: Connection) => {
      const source = nodes.find((n) => n.id === connection.source);
      const type = source
        ? (source.data as BlockNodeData).block.output.type
        : "unknown";
      setEdges((current) =>
        addEdge(
          {
            ...connection,
            style: { stroke: portColor(type), strokeWidth: 2 },
          },
          // One wire per input port: the second would silently win.
          current.filter(
            (e) =>
              !(
                e.target === connection.target &&
                e.targetHandle === connection.targetHandle
              ),
          ),
        ),
      );
    },
    [nodes, setEdges],
  );

  // Each node has to know which of its ports are wired, so it can say
  // "wired" instead of "needs a value" — kept in the node data so the
  // node itself stays a pure render of what it was handed.
  useEffect(() => {
    setNodes((current) =>
      current.map((node) => {
        const wired = edges
          .filter((e) => e.target === node.id)
          .map((e) => e.targetHandle ?? "");
        const data = node.data as BlockNodeData;
        if (same(data.wired, wired)) return node;
        return { ...node, data: { ...data, wired } };
      }),
    );
  }, [edges, setNodes]);

  const patch = useCallback(
    (id: string, change: Partial<BlockNodeData>) => {
      setNodes((current) =>
        current.map((node) =>
          node.id === id ? { ...node, data: { ...node.data, ...change } } : node,
        ),
      );
    },
    [setNodes],
  );

  // ---------- server actions ----------

  const validate = useCallback(async () => {
    clearProblems(setNodes);
    try {
      const result = await api.validate(payload());
      if (!result.valid) {
        blame(setNodes, result.node_id, result.error);
        setStatus({ kind: "error", message: result.error });
      } else if (!result.runnable) {
        blame(setNodes, result.node_id, result.error);
        setStatus({
          kind: "warn",
          message: `Valid, but not runnable here — ${result.error}`,
        });
      } else {
        setStatus({ kind: "ok", message: "Valid, and runnable here." });
      }
    } catch (e) {
      report(e, setStatus, setNodes);
    }
  }, [payload, setNodes]);

  const save = useCallback(async () => {
    try {
      const result = await api.save(payload());
      const text = result.yaml || result.json;
      download(result.yaml ? "pipeline.yaml" : "pipeline.json", text);
      setStatus({
        kind: "ok",
        message: result.yaml
          ? "Saved pipeline.yaml."
          : "Saved pipeline.json (install molforge[repro] for YAML).",
      });
    } catch (e) {
      report(e, setStatus, setNodes);
    }
  }, [payload, setNodes]);

  const open = useCallback(
    async (file: File) => {
      try {
        const { graph } = await api.load(await file.text());
        const restored = graph.nodes.flatMap((node) => {
          const block = byId.get(node.block_id);
          if (!block) return [];
          const data: BlockNodeData = {
            block,
            inputs: node.inputs,
            parameters: node.parameters,
            wired: [],
          };
          return [
            {
              id: node.id,
              type: "block",
              position: { x: node.position[0], y: node.position[1] },
              data: data as unknown as Record<string, unknown>,
            } as Node,
          ];
        });
        setNodes(restored);
        setEdges(
          graph.edges.map((edge, i) => {
            const source = graph.nodes.find((n) => n.id === edge.source);
            const type = source
              ? (byId.get(source.block_id)?.output.type ?? "unknown")
              : "unknown";
            return {
              id: `e${i}`,
              source: edge.source,
              target: edge.target,
              targetHandle: edge.target_port,
              style: { stroke: portColor(type), strokeWidth: 2 },
            };
          }),
        );
        setRun(null);
        setStatus({ kind: "ok", message: `Opened ${file.name}.` });
      } catch (e) {
        report(e, setStatus, setNodes);
      }
    },
    [byId, setEdges, setNodes],
  );

  const start = useCallback(async () => {
    clearProblems(setNodes);
    try {
      setRun(await api.startRun(payload()));
      setStatus({ kind: "ok", message: "Running…" });
    } catch (e) {
      report(e, setStatus, setNodes);
    }
  }, [payload, setNodes]);

  // Poll while a run is live. Each poll repaints the nodes, which is how
  // the canvas shows which block is working rather than only that
  // something is.
  useEffect(() => {
    if (!run || run.state !== "running") return;
    const timer = setInterval(async () => {
      try {
        setRun(await api.run(run.id));
      } catch {
        /* a dropped poll is not worth interrupting the run over */
      }
    }, POLL_MS);
    return () => clearInterval(timer);
  }, [run]);

  useEffect(() => {
    if (!run) return;
    const byNode = new Map(run.steps.map((step) => [step.node_id, step]));
    setNodes((current) =>
      current.map((node) => ({
        ...node,
        data: { ...node.data, run: byNode.get(node.id) },
      })),
    );
    if (run.state === "succeeded")
      setStatus({ kind: "ok", message: `Finished — ${run.result}` });
    else if (run.state === "failed")
      setStatus({ kind: "error", message: run.error });
    else if (run.state === "cancelled")
      setStatus({ kind: "warn", message: "Cancelled." });
  }, [run, setNodes]);

  const selectedNode = nodes.find((n) => n.id === selected);
  const running = run?.state === "running";

  if (loadError) {
    return (
      <div className="fatal">
        <h1>Can't reach the molforge server</h1>
        <p>{loadError}</p>
        <p className="sub">
          Start it with <code>molforge-gui</code>, then reload.
        </p>
      </div>
    );
  }

  return (
    <div className="app">
      <Palette blocks={blocks} onAdd={addBlock} />

      <main>
        <Toolbar
          running={Boolean(running)}
          empty={nodes.length === 0}
          onValidate={validate}
          onSave={save}
          onOpen={open}
          onRun={start}
          onCancel={() => run && api.cancel(run.id).then(setRun)}
          onClear={() => {
            setNodes([]);
            setEdges([]);
            setRun(null);
            setStatus({ kind: "idle", message: "" });
          }}
        />

        <div className="canvas">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={NODE_TYPES}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            isValidConnection={isValidConnection}
            onNodeClick={(_, node) => setSelected(node.id)}
            onPaneClick={() => setSelected(null)}
            fitView
            // Without a cap, two nodes fill the viewport at 4x and the
            // editor opens looking like a zoomed-in screenshot.
            fitViewOptions={{ maxZoom: 1, padding: 0.25 }}
            minZoom={0.2}
            maxZoom={1.75}
            proOptions={{ hideAttribution: false }}
          >
            <Background gap={18} size={1} />
            <Controls />
            <MiniMap pannable zoomable nodeColor={minimapColor} />
          </ReactFlow>

          {nodes.length === 0 && (
            <div className="hint">
              <p>Pick a block on the left to start a pipeline.</p>
              <p className="sub">
                Wire one block's output into the next. Ports only connect when
                their types match.
              </p>
            </div>
          )}
        </div>

        <StatusBar status={status} run={run} />
      </main>

      {selectedNode ? (
        <Inspector
          block={(selectedNode.data as BlockNodeData).block}
          inputs={(selectedNode.data as BlockNodeData).inputs}
          parameters={(selectedNode.data as BlockNodeData).parameters}
          wired={(selectedNode.data as BlockNodeData).wired}
          onInput={(name, value) =>
            patch(selectedNode.id, {
              inputs: {
                ...(selectedNode.data as BlockNodeData).inputs,
                [name]: value,
              },
            })
          }
          onParameter={(name, value) =>
            patch(selectedNode.id, {
              parameters: {
                ...(selectedNode.data as BlockNodeData).parameters,
                [name]: value,
              },
            })
          }
        />
      ) : (
        <aside className="inspector empty-panel">
          <div className="panel-head">
            <h2>Nothing selected</h2>
            <p className="sub">Click a block to edit its settings.</p>
          </div>
        </aside>
      )}
    </div>
  );
}

// ---------- small pieces ----------

interface Status {
  kind: "idle" | "ok" | "warn" | "error";
  message: string;
}

function Toolbar({
  running,
  empty,
  onValidate,
  onSave,
  onOpen,
  onRun,
  onCancel,
  onClear,
}: {
  running: boolean;
  empty: boolean;
  onValidate: () => void;
  onSave: () => void;
  onOpen: (file: File) => void;
  onRun: () => void;
  onCancel: () => void;
  onClear: () => void;
}) {
  return (
    <header className="toolbar">
      <strong className="brand">molforge</strong>
      <span className="sep" />
      <button onClick={onValidate} disabled={empty}>
        Check
      </button>
      <button onClick={onSave} disabled={empty}>
        Save
      </button>
      <label className="button-like">
        Open…
        <input
          type="file"
          accept=".yaml,.yml,.json"
          hidden
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onOpen(file);
            e.target.value = "";
          }}
        />
      </label>
      <span className="sep" />
      {running ? (
        <button onClick={onCancel}>Cancel</button>
      ) : (
        <button className="primary" onClick={onRun} disabled={empty}>
          Run
        </button>
      )}
      <span className="spacer" />
      <button onClick={onClear} disabled={empty || running}>
        Clear
      </button>
    </header>
  );
}

function StatusBar({ status, run }: { status: Status; run: RunRecord | null }) {
  const done = run ? run.steps.filter((s) => s.state === "succeeded").length : 0;
  return (
    <footer className="statusbar" data-kind={status.kind}>
      <span className="message">{status.message || "Ready."}</span>
      {run && (
        <span className="progress">
          {done}/{run.steps.length} steps
        </span>
      )}
    </footer>
  );
}

// ---------- helpers ----------

type SetNodes = React.Dispatch<React.SetStateAction<Node[]>>;

function clearProblems(setNodes: SetNodes) {
  setNodes((current) =>
    current.map((node) => ({ ...node, data: { ...node.data, problem: undefined } })),
  );
}

function blame(setNodes: SetNodes, nodeId: string, message: string) {
  if (!nodeId) return;
  setNodes((current) =>
    current.map((node) =>
      node.id === nodeId
        ? { ...node, data: { ...node.data, problem: message } }
        : node,
    ),
  );
}

function report(
  error: unknown,
  setStatus: (status: Status) => void,
  setNodes: SetNodes,
) {
  const message =
    error instanceof Error ? error.message : "something went wrong";
  setStatus({ kind: "error", message });
  if (error instanceof ApiError) blame(setNodes, error.nodeId, error.message);
}

function download(filename: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/plain" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

/** Paint minimap nodes by what they produce, matching the canvas. */
function minimapColor(node: Node): string {
  return portColor((node.data as BlockNodeData).block.output.type);
}

function same(a: string[], b: string[]): boolean {
  return a.length === b.length && a.every((value, i) => value === b[i]);
}
