/**
 * The 3D view of what a step produced.
 *
 * 3Dmol.js, bundled rather than loaded from a CDN: the editor is a local
 * tool that should keep working on a laptop with no network, which is
 * exactly the situation someone is in when they are waiting on a long
 * local run.
 *
 * Styling is by *role*, not by guesswork. A docked pose and its receptor
 * arrive as two separate structures and want different representations —
 * cartoon for the protein, sticks for the ligand — and inferring that
 * from atom counts gets peptide ligands wrong.
 */

import { useEffect, useRef, useState } from "react";
import type { StructureRef } from "../api";

interface Props {
  /** Where to fetch each structure's PDB text from. */
  urlFor: (ref: StructureRef) => string;
  structures: StructureRef[];
  /** Pocket centres to mark, when the step found some. */
  centers?: number[][];
  /** Which structure to emphasise — a pose selected in the table. */
  highlight?: string;
}

type Viewer = {
  addModel: (data: string, format: string) => unknown;
  setStyle: (sel: unknown, style: unknown) => void;
  addSphere: (spec: unknown) => void;
  addLabel: (text: string, spec: unknown) => void;
  removeAllModels: () => void;
  removeAllShapes: () => void;
  removeAllLabels: () => void;
  zoomTo: () => void;
  render: () => void;
  resize: () => void;
  clear: () => void;
};

export function StructureViewer({
  urlFor,
  structures,
  centers = [],
  highlight,
}: Props) {
  const host = useRef<HTMLDivElement>(null);
  const viewer = useRef<Viewer | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  // 3Dmol is imported lazily, so the viewer does not exist on the first
  // render. Without something to depend on, the load effect below runs
  // once against a null viewer, bails, and never retries — the panel
  // then shows an empty canvas forever.
  const [ready, setReady] = useState(false);

  // Create the viewer once. 3Dmol attaches a WebGL canvas to the host and
  // keeps its own state, so re-creating it per render would leak contexts
  // — browsers cap those, and the cap is low enough to hit in a session.
  useEffect(() => {
    let disposed = false;
    (async () => {
      const $3Dmol = await import("3dmol");
      if (disposed || !host.current) return;
      viewer.current = $3Dmol.createViewer(host.current, {
        backgroundColor: readSurface(),
      }) as unknown as Viewer;
      viewer.current.render();
      setReady(true);
    })();
    return () => {
      disposed = true;
      viewer.current?.clear();
      viewer.current = null;
      setReady(false);
    };
  }, []);

  useEffect(() => {
    const observer = new ResizeObserver(() => viewer.current?.resize());
    if (host.current) observer.observe(host.current);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      const view = viewer.current;
      if (!view) return;
      // The host can be laid out after the viewer is built; 3Dmol sizes
      // its projection from the container, so tell it the real size
      // before drawing rather than rendering into a stale one.
      view.resize();
      setError("");
      setLoading(true);
      view.removeAllModels();
      view.removeAllShapes();
      view.removeAllLabels();

      try {
        for (const ref of structures) {
          const response = await fetch(urlFor(ref));
          if (!response.ok) throw new Error(`${ref.label}: ${response.status}`);
          const text = await response.text();
          if (cancelled) return;

          view.addModel(text, ref.format || "pdb");
          const dimmed = Boolean(highlight) && highlight !== ref.key;
          view.setStyle({ model: -1 }, styleFor(ref.role, dimmed));
        }

        centers.forEach((center, index) => {
          if (center.length !== 3) return;
          const [x, y, z] = center;
          // Hollow, so a pocket marker never hides the residues that
          // make it one.
          view.addSphere({
            center: { x, y, z },
            radius: 2.2,
            color: POCKET,
            alpha: 0.55,
          });
          view.addLabel(`${index + 1}`, {
            position: { x, y, z },
            fontSize: 11,
            backgroundOpacity: 0.65,
          });
        });

        view.zoomTo();
        view.render();
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [structures, centers, highlight, urlFor, ready]);

  return (
    <div className="viewer">
      <div className="viewer-canvas" ref={host} />
      {loading && <p className="viewer-note">Loading…</p>}
      {error && <p className="viewer-note error">{error}</p>}
      {!loading && !error && structures.length === 0 && (
        <p className="viewer-note">Nothing to show in 3D for this step.</p>
      )}
    </div>
  );
}

/** Ligand orange and protein blue, matching the port colours. */
const LIGAND = "#eb6834";
const POCKET = "#008300";

function styleFor(role: string, dimmed: boolean) {
  if (role === "ligand") {
    return {
      stick: { radius: 0.17, color: dimmed ? "grey" : LIGAND, opacity: dimmed ? 0.4 : 1 },
    };
  }
  // Spectrum along the chain: the cheapest way to make a fold's
  // topology readable, and it carries no claim about the data.
  return {
    cartoon: { color: dimmed ? "grey" : "spectrum", opacity: dimmed ? 0.45 : 1 },
  };
}

/** Match the page's surface so the canvas doesn't flash white in dark mode. */
function readSurface(): string {
  const value = getComputedStyle(document.documentElement)
    .getPropertyValue("--surface-1")
    .trim();
  return value || "#ffffff";
}
