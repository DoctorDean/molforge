/**
 * The selected block's settings.
 *
 * Every widget is chosen by the `kind` the catalog derived from the
 * wrapper's own annotation, so a dropdown's options are the engine's
 * `Literal` and cannot drift from it. Three states are kept distinct
 * because they mean different things to an engine: a value that is set,
 * a value left at its default, and a value deliberately unset so the
 * engine decides for itself.
 */

import type { BlockParameter, BlockPort, BlockSpec } from "../api";
import { portColor, portLabel } from "../ports";

interface Props {
  block: BlockSpec;
  inputs: Record<string, unknown>;
  parameters: Record<string, unknown>;
  wired: string[];
  onInput: (name: string, value: unknown) => void;
  onParameter: (name: string, value: unknown) => void;
}

export function Inspector({
  block,
  inputs,
  parameters,
  wired,
  onInput,
  onParameter,
}: Props) {
  return (
    <aside className="inspector">
      <div className="panel-head">
        <h2>{block.engine}</h2>
        <p className="sub">
          {block.label}
          {block.version && ` · v${block.version}`}
        </p>
      </div>

      <div className="scroll">
        {!block.available && (
          <p className="notice warn">
            Not installed here: {block.detail}. You can still draw and save this
            pipeline — it just won't run on this machine.
          </p>
        )}

        {block.inputs.length > 0 && (
          <section>
            <h3>Inputs</h3>
            {block.inputs.map((port) => (
              <InputField
                key={port.name}
                port={port}
                wired={wired.includes(port.name)}
                value={inputs[port.name]}
                onChange={(v) => onInput(port.name, v)}
              />
            ))}
          </section>
        )}

        <section>
          <h3>Output</h3>
          <p className="port-line">
            <i style={{ background: portColor(block.output.type) }} />
            {portLabel(block.output.type)}
          </p>
        </section>

        {block.parameters.length > 0 && (
          <section>
            <h3>Settings</h3>
            {block.parameters.map((parameter) => (
              <ParameterField
                key={parameter.name}
                parameter={parameter}
                value={parameters[parameter.name]}
                onChange={(v) => onParameter(parameter.name, v)}
              />
            ))}
          </section>
        )}
      </div>
    </aside>
  );
}

function InputField({
  port,
  wired,
  value,
  onChange,
}: {
  port: BlockPort;
  wired: boolean;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  return (
    <label className="field">
      <span className="label">
        <i style={{ background: portColor(port.type) }} />
        {port.name}
        <em>{portLabel(port.type)}</em>
      </span>
      {wired ? (
        <p className="wired">Comes from the block wired into this port.</p>
      ) : (
        <input
          value={value === undefined || value === null ? "" : String(value)}
          placeholder={port.required ? "required" : "optional"}
          onChange={(e) => onChange(e.target.value)}
        />
      )}
    </label>
  );
}

function ParameterField({
  parameter,
  value,
  onChange,
}: {
  parameter: BlockParameter;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  const current = value ?? parameter.default;

  return (
    <label className="field">
      <span className="label">
        {parameter.name}
        {parameter.required && <strong className="req">required</strong>}
        <em title={parameter.annotation}>{parameter.annotation}</em>
      </span>
      {renderWidget(parameter, current, onChange)}
    </label>
  );
}

function renderWidget(
  parameter: BlockParameter,
  current: unknown,
  onChange: (value: unknown) => void,
) {
  switch (parameter.kind) {
    case "choice":
      return (
        <select
          value={current === undefined || current === null ? "" : String(current)}
          onChange={(e) => onChange(e.target.value || null)}
        >
          {/* An optional choice can be left to the engine, which is not
              the same as any of its options. */}
          {parameter.optional && <option value="">— engine default —</option>}
          {(parameter.choices ?? []).map((choice) => (
            <option key={choice} value={choice}>
              {choice}
            </option>
          ))}
        </select>
      );

    case "bool":
      return (
        <input
          type="checkbox"
          className="checkbox"
          checked={Boolean(current)}
          onChange={(e) => onChange(e.target.checked)}
        />
      );

    case "int":
    case "float":
      return (
        <input
          type="number"
          step={parameter.kind === "int" ? 1 : "any"}
          value={current === undefined || current === null ? "" : Number(current)}
          placeholder={parameter.optional ? "engine default" : ""}
          onChange={(e) =>
            onChange(
              e.target.value === ""
                ? null
                : parameter.kind === "int"
                  ? parseInt(e.target.value, 10)
                  : parseFloat(e.target.value),
            )
          }
        />
      );

    case "vector3": {
      // The docking box. Three fields rather than one comma-separated
      // string: this is the argument that decides whether a run docks to
      // the pocket or to empty space, and a typo in a free-text triple is
      // invisible until the results make no sense.
      const triple: (number | undefined)[] = Array.isArray(current)
        ? (current as (number | undefined)[])
        : [];
      return (
        <span className="vector3">
          {["x", "y", "z"].map((axis, index) => (
            <input
              key={axis}
              type="number"
              step="any"
              aria-label={`${parameter.name} ${axis}`}
              placeholder={axis}
              value={triple[index] ?? ""}
              onChange={(e) => {
                const next = [triple[0], triple[1], triple[2]];
                next[index] =
                  e.target.value === "" ? undefined : parseFloat(e.target.value);
                onChange(next.every((v) => v === undefined) ? null : next);
              }}
            />
          ))}
        </span>
      );
    }

    default:
      return (
        <input
          value={current === undefined || current === null ? "" : String(current)}
          placeholder={parameter.optional ? "engine default" : ""}
          onChange={(e) => onChange(e.target.value === "" ? null : e.target.value)}
        />
      );
  }
}
