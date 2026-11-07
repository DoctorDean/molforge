/**
 * Port identity: colour, shape, and the words that go with them.
 *
 * Nine port types over seven hues. `structure_ensemble` shares
 * `structure`'s hue and `designs` shares `sequence`'s, because each pair
 * is the same kind of thing in single and plural — a secondary encoding
 * (the doubled ring) rather than a new identity. That keeps the
 * categorical palette inside its validated slots instead of inventing an
 * eighth and ninth hue, and leaves red free to mean "invalid".
 */

import type { PortType } from "./api";

interface PortStyle {
  /** CSS variable holding this port's hue. */
  color: string;
  /** Plural kinds get a ring, so identity never rests on colour alone. */
  many: boolean;
  /** What the type is, in words, for the inspector and tooltips. */
  label: string;
}

const STYLES: Record<string, PortStyle> = {
  structure: { color: "--port-structure", many: false, label: "structure" },
  structure_ensemble: {
    color: "--port-structure",
    many: true,
    label: "structures (ensemble)",
  },
  sequence: { color: "--port-sequence", many: false, label: "sequence" },
  designs: { color: "--port-sequence", many: true, label: "designed sequences" },
  ligand: { color: "--port-ligand", many: false, label: "ligand" },
  alignment: { color: "--port-alignment", many: false, label: "alignment" },
  docking_result: {
    color: "--port-docking_result",
    many: false,
    label: "docking result",
  },
  pockets: { color: "--port-pockets", many: true, label: "pockets" },
  pdb_id: { color: "--port-pdb_id", many: false, label: "PDB ID" },
  complex_spec: { color: "--port-pdb_id", many: false, label: "complex spec" },
};

const FALLBACK: PortStyle = {
  color: "--port-unknown",
  many: false,
  label: "unknown",
};

export function portStyle(type: PortType | string): PortStyle {
  return STYLES[type] ?? FALLBACK;
}

export function portColor(type: PortType | string): string {
  return `var(${portStyle(type).color})`;
}

export function portLabel(type: PortType | string): string {
  return portStyle(type).label;
}
