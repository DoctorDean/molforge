/**
 * What a step produced: the structure in 3D, and the numbers beside it.
 *
 * A docking run's answer is not "a DockingResult" — it is nine poses with
 * scores, and the one you care about is the one you can see in the
 * pocket. The panel puts the table and the view next to each other and
 * links them, so selecting a row shows that pose and dims the rest.
 */

import { useCallback, useMemo, useState } from "react";
import type { ResultView, StepRun, StructureRef } from "../api";
import { StructureViewer } from "./StructureViewer";

interface Props {
  runId: string;
  step: StepRun;
  onClose: () => void;
}

export function ResultsPanel({ runId, step, onClose }: Props) {
  const result: ResultView | null = step.result;
  const [selected, setSelected] = useState<string | undefined>(undefined);

  const urlFor = useCallback(
    (ref: StructureRef) =>
      `/api/runs/${runId}/steps/${step.step}/structure/${ref.key}`,
    [runId, step.step],
  );

  // The viewer re-fetches when its props change identity, so these are
  // memoised — otherwise every parent render reloads the structures.
  const structures = useMemo(() => result?.structures ?? [], [result]);
  const centers = useMemo(
    () => (result?.detail?.centers as number[][] | undefined) ?? [],
    [result],
  );

  if (!result || result.kind === "none") {
    return (
      <section className="results">
        <Head step={step} onClose={onClose} />
        <p className="results-empty">
          {result?.error || "This step produced nothing to preview."}
        </p>
      </section>
    );
  }

  const rowKey = rowStructureKey(result.kind);

  return (
    <section className="results">
      <Head step={step} onClose={onClose} />

      <div className="results-body">
        {structures.length > 0 && (
          <StructureViewer
            urlFor={urlFor}
            structures={structures}
            centers={centers}
            highlight={selected}
          />
        )}

        <div className="results-side">
          <dl className="facts">
            {Object.entries(result.detail)
              .filter(([key]) => key !== "centers")
              .map(([key, value]) => (
                <div key={key}>
                  <dt>{key}</dt>
                  <dd>{formatFact(value)}</dd>
                </div>
              ))}
          </dl>

          {result.rows.length > 0 && (
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    {result.columns.map((c) => (
                      <th key={c}>{c}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.rows.map((row, index) => {
                    const key = rowKey ? `${rowKey}-${index + 1}` : undefined;
                    const selectable = Boolean(
                      key && structures.some((s) => s.key === key),
                    );
                    return (
                      <tr
                        key={index}
                        data-selected={selected === key || undefined}
                        data-selectable={selectable || undefined}
                        onClick={() =>
                          selectable &&
                          setSelected(selected === key ? undefined : key)
                        }
                      >
                        {row.map((cell, i) => (
                          <td key={i}>{formatCell(cell)}</td>
                        ))}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          {result.truncated && (
            <p className="results-note">
              Showing the first {result.rows.length || structures.length}. The
              rest are in the run, not lost — replay the saved pipeline to get
              them all.
            </p>
          )}
        </div>
      </div>
    </section>
  );
}

function Head({ step, onClose }: { step: StepRun; onClose: () => void }) {
  return (
    <header className="results-head">
      <strong>{step.engine}</strong>
      <span className="sub">{step.result?.summary || step.output}</span>
      {step.elapsed > 0 && (
        <span className="sub mono">{step.elapsed.toFixed(1)}s</span>
      )}
      <button className="close" onClick={onClose} aria-label="Close results">
        ✕
      </button>
    </header>
  );
}

/**
 * Which structure a table row corresponds to, if any.
 *
 * Only poses and ensembles have one structure per row; a pocket's row is
 * a measurement of a site on a structure that is already shown, and
 * wiring it to a layer that doesn't exist would make rows look clickable
 * when they aren't.
 */
function rowStructureKey(kind: string): string | null {
  if (kind === "poses") return "pose";
  if (kind === "ensemble") return "model";
  return null;
}

function formatCell(cell: unknown): string {
  if (cell === null || cell === undefined) return "—";
  if (typeof cell === "number") return String(cell);
  const text = String(cell);
  return text.length > 60 ? `${text.slice(0, 57)}…` : text;
}

function formatFact(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (Array.isArray(value)) return `${value.length}`;
  return String(value);
}
