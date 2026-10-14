# Coming from another library

If you already use BioPython, Biotite, or MDAnalysis, you don't have to
switch wholesale. molforge is **connective tissue, not a replacement** — it
reads and writes the same formats those libraries produce, so you can adopt
it for the parts it does well (multi-engine workflows, one data model across
tools) and keep using your existing code for the rest.

This page maps the operations you already know to their molforge
equivalents.

## The one concept to learn

molforge's data model is a single `Protein` holding a flat, NumPy-backed
`AtomArray`, with hierarchical views on top:

```python
from molforge.io import read_pdb

protein = read_pdb("1ubq.pdb")

protein.chains["A"].residues[42]     # hierarchical — like BioPython
protein.atom_array.coords            # linear (N, 3) — like Biotite / for ML
```

Same data, two views, no conversion between them — that's the whole idea.

---

## Coming from BioPython (`Bio.PDB`)

| BioPython | molforge |
| --------- | -------- |
| `PDBParser().get_structure(id, path)` | `molforge.io.read_pdb(path)` / `load(path)` |
| `MMCIFParser()` | `molforge.io.read_cif(path)` |
| fetch from RCSB (manual) | `molforge.io.fetch("1UBQ")` |
| `structure[0]["A"][42]` | `protein.chains["A"].residues[42]` |
| `atom.get_coord()` | `atom.coord` / `protein.atom_array.coords` |
| `Superimposer()` | `molforge.structure.superpose` / `rmsd` |
| `PPBuilder` / `internal_coords` (φ/ψ) | `molforge.structure.ramachandran` / `phi_psi_omega` |
| `calc_dihedral(...)` | `molforge.structure.dihedral(...)` |
| `DSSP(model, path)` | `molforge.structure.dssp(protein)` |
| `Bio.Align.PairwiseAligner` | `molforge.sequence.needleman_wunsch` / `smith_waterman` |
| `Bio.SeqIO` (FASTA) | `molforge.io.read_fasta` / `write_fasta` |

molforge's φ/ψ follow the same IUPAC sign convention as BioPython's
`calc_dihedral` (a right-handed helix is φ ≈ −60°), so Ramachandran results
line up.

## Coming from Biotite

Biotite users will feel at home: molforge's `AtomArray` is the same
flat-array idea. The difference is the workflow layer on top — engine
wrappers, provenance, caching.

| Biotite | molforge |
| ------- | -------- |
| `biotite.structure.io.load_structure(path)` | `molforge.io.load(path)` |
| `AtomArray` (annotations + coord) | `molforge.core.AtomArray` |
| `array[array.chain_id == "A"]` | `protein.select(chain_id="A")` / `atom_array` masks |
| `biotite.structure.rmsd` / `superimpose` | `molforge.structure.rmsd` / `superpose` |
| `biotite.structure.sasa` | `molforge.structure.sasa` |
| substitution-matrix alignment | `molforge.sequence.align` |

Because both use element-wise NumPy arrays, moving coordinates between the
two is a plain array copy — no format round-trip.

## Coming from MDAnalysis

MDAnalysis is trajectory-first; molforge leans on **mdtraj** for
trajectories and adds the cross-tool workflow layer. molforge doesn't aim to
replace MDAnalysis's analysis breadth — it interops.

| MDAnalysis | molforge |
| ---------- | -------- |
| `Universe(topology, trajectory)` | `molforge.io.read_trajectory(...)` (mdtraj-backed) |
| `u.select_atoms("chainID A")` | `protein.select(chain_id="A")` |
| `u.atoms.positions` | `protein.atom_array.coords` |
| iterate `u.trajectory` frames | iterate the molforge `Trajectory` |
| `rms.RMSD(...)` | `molforge.structure.rmsd` / `rmsd_per_residue` |
| running MD | `molforge.wrappers.md` (OpenMM / GROMACS / AMBER) |

If you have an existing MDAnalysis analysis you like, keep it — molforge
reads the same PDB/DCD/XTC files, so the two coexist on the same data.

---

## When *not* to reach for molforge

molforge deliberately doesn't reimplement everything. For deep
single-library work — MDAnalysis's trajectory-analysis breadth, RDKit's
cheminformatics, Biotite's sequence-database tooling — use those libraries
directly. molforge's value shows up when a workflow spans **several** tools
and you're tired of writing the glue: fold with one engine, dock with
another, score, validate, and walk away with a reproducible record — one
`Protein`, one `Provenance`, one cache throughout.
