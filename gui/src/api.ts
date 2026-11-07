/**
 * The shapes the Python side sends, and the calls that fetch them.
 *
 * These interfaces mirror the `to_dict()` methods in `molforge.gui` — the
 * catalog's BlockSpec, the graph's nodes and edges, the runner's StepRun.
 * They are written out rather than generated because the duplication is
 * small, visible, and caught immediately: every field here is read by a
 * component, so a backend rename surfaces as a type error at build time
 * rather than as an undefined at 3am.
 */

export type PortType =
  | "sequence"
  | "structure"
  | "structure_ensemble"
  | "ligand"
  | "alignment"
  | "docking_result"
  | "pockets"
  | "designs"
  | "pdb_id"
  | "complex_spec"
  | "unknown";

export interface BlockPort {
  name: string;
  type: PortType;
  required: boolean;
}

/** Widget hints the catalog derives from a parameter's annotation. */
export type ParameterKind =
  | "choice"
  | "bool"
  | "int"
  | "float"
  | "vector3"
  | "path"
  | "text";

export interface BlockParameter {
  name: string;
  kind: ParameterKind;
  annotation: string;
  default: unknown;
  choices?: string[];
  /** `X | None` — the engine decides. Distinct from sitting on a default. */
  optional: boolean;
  /** No default at all; the graph will not run without it. */
  required: boolean;
}

export interface BlockSpec {
  id: string;
  engine: string;
  operation: string;
  label: string;
  category: string;
  inputs: BlockPort[];
  output: BlockPort;
  parameters: BlockParameter[];
  available: boolean;
  detail: string;
  version: string;
}

export interface Catalog {
  blocks: BlockSpec[];
  port_types: PortType[];
}

export interface GraphNodePayload {
  id: string;
  block_id: string;
  parameters: Record<string, unknown>;
  inputs: Record<string, unknown>;
  position: [number, number];
}

export interface GraphEdgePayload {
  source: string;
  target: string;
  target_port: string;
}

export interface GraphPayload {
  nodes: GraphNodePayload[];
  edges: GraphEdgePayload[];
}

export type RunState =
  | "pending"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled";

export interface StepRun {
  node_id: string;
  step: number;
  engine: string;
  operation: string;
  state: RunState;
  elapsed: number;
  error: string;
  traceback: string;
  output: string;
}

export interface RunRecord {
  id: string;
  state: RunState;
  error: string;
  result: string;
  steps: StepRun[];
}

export interface ValidationResult {
  /** A coherent pipeline — savable even if nothing here can run it. */
  valid: boolean;
  /** Every engine is installed on the machine serving this page. */
  runnable: boolean;
  error: string;
  node_id: string;
}

/** A 4xx from the API, carrying the node the editor should highlight. */
export class ApiError extends Error {
  nodeId: string;

  constructor(message: string, nodeId = "") {
    super(message);
    this.name = "ApiError";
    this.nodeId = nodeId;
  }
}

async function request<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    // FastAPI nests our {error, node_id} under `detail`; anything else
    // (a proxy error, a crash) has no such shape and must still read as
    // something rather than "[object Object]".
    let message = `${response.status} ${response.statusText}`;
    let nodeId = "";
    try {
      const payload = await response.json();
      const detail = payload?.detail ?? payload;
      if (typeof detail === "string") message = detail;
      else if (detail?.error) {
        message = detail.error;
        nodeId = detail.node_id ?? "";
      }
    } catch {
      /* a non-JSON body: the status line is all there is */
    }
    throw new ApiError(message, nodeId);
  }
  return (await response.json()) as T;
}

export const api = {
  blocks: () => request<Catalog>("/api/blocks"),

  validate: (graph: GraphPayload) =>
    request<ValidationResult>("/api/graph/validate", { graph }),

  save: (graph: GraphPayload) =>
    request<{ yaml: string; json: string; context: Record<string, unknown> }>(
      "/api/pipeline/save",
      { graph },
    ),

  load: (text: string) =>
    request<{ graph: GraphPayload }>("/api/pipeline/load", { text }),

  startRun: (graph: GraphPayload) => request<RunRecord>("/api/runs", { graph }),

  run: (id: string) => request<RunRecord>(`/api/runs/${id}`),

  cancel: (id: string) => request<RunRecord>(`/api/runs/${id}/cancel`, {}),
};
