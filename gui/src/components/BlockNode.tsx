/**
 * One block on the canvas.
 *
 * Carries four things at a glance: what it is, what it takes, what it
 * produces, and how its last run went. Ports are labelled as well as
 * coloured — the colour is the fast channel, the name is the one that
 * works in greyscale, at 2am, and for the ~8% of men who will not read
 * the hues apart.
 */

import { Handle, Position } from "@xyflow/react";
import type { NodeProps } from "@xyflow/react";
import type { BlockSpec, RunState, StepRun } from "../api";
import { portColor, portLabel } from "../ports";

export interface BlockNodeData {
  block: BlockSpec;
  /** Literal values for ports with no wire. */
  inputs: Record<string, unknown>;
  parameters: Record<string, unknown>;
  /** Which input ports currently have a wire, so the node can say so. */
  wired: string[];
  run?: StepRun;
  /** Set when validation blamed this node. */
  problem?: string;
  [key: string]: unknown;
}

const STATE_LABEL: Record<RunState, string> = {
  pending: "queued",
  running: "running",
  succeeded: "done",
  failed: "failed",
  cancelled: "cancelled",
};

const STATE_COLOR: Record<RunState, string> = {
  pending: "var(--text-muted)",
  running: "var(--status-running)",
  succeeded: "var(--status-good)",
  failed: "var(--status-critical)",
  cancelled: "var(--text-muted)",
};

/** A glyph beside the word, so state never rests on colour alone. */
const STATE_GLYPH: Record<RunState, string> = {
  pending: "·",
  running: "▸",
  succeeded: "✓",
  failed: "✕",
  cancelled: "⊘",
};

export function BlockNode({ data, selected }: NodeProps) {
  const { block, inputs, wired, run, problem } = data as BlockNodeData;
  const missing = block.parameters.filter(
    (p) => p.required && isEmpty((data as BlockNodeData).parameters[p.name]),
  );

  return (
    <div
      className="block-node"
      data-selected={selected || undefined}
      data-state={run?.state}
      data-problem={problem ? true : undefined}
      style={{ borderTopColor: portColor(block.output.type) }}
    >
      <header>
        <div className="engine">
          {block.engine}
          {!block.available && (
            <span className="pill" title={block.detail}>
              not installed
            </span>
          )}
        </div>
        <div className="action">{block.label}</div>
      </header>

      {block.inputs.length > 0 && (
        <ul className="ports">
          {block.inputs.map((port, index) => {
            const connected = wired.includes(port.name);
            const literal = inputs[port.name];
            return (
              <li key={port.name}>
                <Handle
                  id={port.name}
                  type="target"
                  position={Position.Left}
                  className="port"
                  data-many={portLabel(port.type).includes("(") || undefined}
                  style={{
                    top: HEADER + index * PORT_GAP,
                    background: portColor(port.type),
                  }}
                  title={`${port.name} — ${portLabel(port.type)}`}
                />
                <span className="name">{port.name}</span>
                <span className="value">
                  {connected ? (
                    <em>wired</em>
                  ) : isEmpty(literal) ? (
                    <em className={port.required ? "needed" : "optional"}>
                      {port.required ? "needs a value" : "optional"}
                    </em>
                  ) : (
                    String(literal)
                  )}
                </span>
              </li>
            );
          })}
        </ul>
      )}

      <footer>
        <span className="out" title={`outputs ${portLabel(block.output.type)}`}>
          <i style={{ background: portColor(block.output.type) }} />
          {portLabel(block.output.type)}
        </span>
        {run && (
          <span className="state" style={{ color: STATE_COLOR[run.state] }}>
            {STATE_GLYPH[run.state]} {STATE_LABEL[run.state]}
            {run.elapsed > 0 && ` · ${run.elapsed.toFixed(1)}s`}
          </span>
        )}
      </footer>

      {run?.output && <div className="result">{run.output}</div>}
      {run?.error && <div className="error">{run.error}</div>}
      {problem && !run?.error && <div className="error">{problem}</div>}
      {missing.length > 0 && !problem && (
        <div className="error">
          set {missing.map((p) => p.name).join(", ")}
        </div>
      )}

      <Handle
        id="output"
        type="source"
        position={Position.Right}
        className="port"
        style={{ background: portColor(block.output.type) }}
        title={`output — ${portLabel(block.output.type)}`}
      />
    </div>
  );
}

/** Vertical geometry, shared with the CSS so handles line up with rows. */
const HEADER = 54;
const PORT_GAP = 22;

function isEmpty(value: unknown): boolean {
  return value === undefined || value === null || value === "";
}
