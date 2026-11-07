/**
 * The block palette: everything molforge can drive, grouped by what it does.
 *
 * Blocks whose engine isn't installed are listed and dimmed rather than
 * hidden, with the reason on hover. Knowing molforge can drive Gnina is
 * worth more than a palette that pretends it can't — and a user who sees
 * the block can go install the binary, where one who sees nothing
 * concludes the feature doesn't exist.
 */

import { useMemo, useState } from "react";
import type { BlockSpec } from "../api";
import { portColor, portLabel } from "../ports";

interface Props {
  blocks: BlockSpec[];
  onAdd: (block: BlockSpec) => void;
}

const CATEGORY_ORDER = [
  "io",
  "msa",
  "folding",
  "pockets",
  "docking",
  "generative",
];

const CATEGORY_LABEL: Record<string, string> = {
  io: "Sources",
  msa: "Homology search",
  folding: "Structure prediction",
  pockets: "Pocket detection",
  docking: "Docking",
  generative: "Design",
};

export function Palette({ blocks, onAdd }: Props) {
  const [query, setQuery] = useState("");

  const groups = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const matching = blocks.filter(
      (b) =>
        !needle ||
        b.engine.toLowerCase().includes(needle) ||
        b.label.toLowerCase().includes(needle) ||
        b.category.toLowerCase().includes(needle),
    );
    const byCategory = new Map<string, BlockSpec[]>();
    for (const block of matching) {
      const list = byCategory.get(block.category) ?? [];
      list.push(block);
      byCategory.set(block.category, list);
    }
    return [...byCategory.entries()].sort(
      (a, b) => rank(a[0]) - rank(b[0]) || a[0].localeCompare(b[0]),
    );
  }, [blocks, query]);

  return (
    <aside className="palette">
      <div className="panel-head">
        <h2>Blocks</h2>
        <input
          type="search"
          placeholder="Filter…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Filter blocks"
        />
      </div>

      <div className="scroll">
        {groups.map(([category, items]) => (
          <section key={category}>
            <h3>{CATEGORY_LABEL[category] ?? category}</h3>
            {items.map((block) => (
              <button
                key={block.id}
                className="block-chip"
                onClick={() => onAdd(block)}
                data-unavailable={!block.available || undefined}
                title={
                  block.available
                    ? `${block.label} → ${portLabel(block.output.type)}`
                    : block.detail
                }
              >
                <i
                  className="swatch"
                  style={{ background: portColor(block.output.type) }}
                />
                <span className="engine">{block.engine}</span>
                <span className="action">{block.label}</span>
                {!block.available && <span className="pill">missing</span>}
              </button>
            ))}
          </section>
        ))}
        {groups.length === 0 && <p className="empty">Nothing matches that.</p>}
      </div>
    </aside>
  );
}

function rank(category: string): number {
  const index = CATEGORY_ORDER.indexOf(category);
  return index === -1 ? CATEGORY_ORDER.length : index;
}
