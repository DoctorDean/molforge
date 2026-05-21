# Quickstart

A five-minute tour of `molforge` covering the parts most workflows
touch: loading a structure, inspecting it through both views of the
data model, running a structural analysis, and saving back to disk.

For longer end-to-end examples (design loops, cross-engine validation,
MD walkthroughs), see the
[walkthrough notebooks](https://github.com/DoctorDean/molforge/tree/main/notebooks/walkthroughs).

## 1. Load a structure

```python
from molforge.io import load

protein = load("1ubq.pdb")
print(protein.name, len(protein), "chains")
```

`load` dispatches on file extension and returns a
[`Protein`](../reference/core.md). PDB, mmCIF, and FASTA all work
out of the box; with the `[io]` extra installed you also get
biotite-backed mmCIF parsing for very large structures.

If you don't have a PDB file handy, fetch one:

```python
from molforge.io import fetch

protein = fetch("1UBQ")   # downloads from the RCSB
```

## 2. Two views, same data

`molforge` exposes the same underlying atoms two ways:

**Hierarchical** — for code that reasons about biology:

```python
chain_a = protein["A"]              # or protein.chains[0]
residue_42 = chain_a.residues[42]
ca = residue_42.atoms["CA"]
print(ca.coord, ca.element)
```

**Linear** — for vectorized analysis and ML:

```python
arr = protein.atom_array          # (N,) struct-of-arrays
print(arr.coords.shape)           # (N, 3) NumPy
print(arr.atom_name[:5])          # ('N', 'CA', 'C', 'O', 'CB')
ca_mask = arr.atom_name == "CA"
print(arr.coords[ca_mask])        # all alpha-carbons
```

Both views read from the same backing arrays — there's no copying or
synchronization layer to worry about.

## 3. Run an analysis

```python
from molforge.structure import rmsd, sasa, dssp

# Compare two structures (auto-aligns alpha-carbons by default)
folded = load("predicted.pdb")
reference = load("crystal.pdb")
print(f"CA-RMSD: {rmsd(folded, reference):.2f} Å")

# Per-residue surface area
sasa_per_res = sasa(protein)

# Secondary structure (8-state or 3-state)
ss = dssp(protein)         # ('H', 'E', 'T', '-', ...)
ss3 = dssp(protein, three_state=True)   # ('H', 'E', 'C', ...)
```

See [`molforge.structure`](../reference/structure.md) for the full
list (contacts, dihedrals, superposition, distance maps).

## 4. Save it

```python
from molforge.io import save

save(protein, "out.pdb")
save(protein, "out.cif")          # mmCIF
save(protein, "out.fasta")        # sequence only
```

`save` dispatches on extension; you don't have to remember per-format
writer names.

## 5. Where to go from here

- **Mutations and alignments** —
  [walkthroughs/01_sequences.ipynb](https://github.com/DoctorDean/molforge/blob/main/notebooks/walkthroughs/01_sequences.ipynb)
- **Geometry deep dive** —
  [walkthroughs/02_structures.ipynb](https://github.com/DoctorDean/molforge/blob/main/notebooks/walkthroughs/02_structures.ipynb)
- **Folding, docking, MD** — [Engine wrappers](../guide/wrappers.md)
- **Adding your own engine** — [Plugins](../guide/plugins.md)
- **Why the data model looks the way it does** —
  [Architecture overview](../architecture/overview.md)
