# File I/O

`molforge.io` provides one-line load/save for the formats structural
biology actually uses, with extension-based dispatch so you don't
have to remember per-format function names.

## The three entry points

```python
from molforge.io import load, save, fetch
```

- [`load(path)`](../reference/io.md) — read a structure or sequence
  from disk. Dispatches on extension.
- [`save(obj, path)`](../reference/io.md) — write a structure or
  sequence. Dispatches on extension.
- [`fetch(pdb_id)`](../reference/io.md) — download a structure from
  the RCSB PDB into a `Protein`.

## Supported formats

### Structure formats

| Extension | Load | Save | Notes                                                              |
| --------- | :--: | :--: | ------------------------------------------------------------------ |
| `.pdb`    |  ✓   |  ✓   | Full PDB parser including altlocs, multi-model.                    |
| `.cif`    |  ✓   |  ✓   | mmCIF; hand-written pure-Python parser, no extra deps.             |
| `.mmcif`  |  ✓   |  ✓   | Alias for `.cif`.                                                  |
| `.fasta`  |  ✓   |  ✓   | One record per chain.                                              |
| `.fa`     |  ✓   |  ✓   | Alias for `.fasta`.                                                |
| `.sdf`    |  ✓   |  ✓   | Small molecules; V2000, multi-molecule.                            |
| `.mol`    |  ✓   |  ✓   | Single-molecule SDF (V2000).                                       |
| `.mol2`   |  ✓   |  ✓   | Tripos MOL2; atom-section coords, elements via type prefix, charges. |
| `.pdbqt`  |  ✓   |  ✓   | AutoDock / Vina; PDB body + charge / AutoDock-type tail.           |
| `.pqr`    |  ✓   |  ✓   | APBS / PDB2PQR; PDB body + whitespace charge / radius tail.        |

### Trajectory formats

Trajectory I/O uses dedicated entry points
([`read_trajectory`](../reference/io.md),
[`iter_trajectory`](../reference/io.md),
[`write_trajectory`](../reference/io.md)) rather than the
`load` / `save` dispatcher — trajectories need an explicit
`topology` argument and return a `molforge.md.Trajectory` rather
than a `Protein`. Backed by [mdtraj](https://www.mdtraj.org/),
available with the `[md]` extra.

| Extension       | Read | Write | Notes                                                |
| --------------- | :--: | :---: | ---------------------------------------------------- |
| `.xtc`          |  ✓   |   ✓   | GROMACS lossy (int16 × 0.001 nm); the common case.   |
| `.trr`          |  ✓   |   ✓   | GROMACS lossless. Velocities / forces dropped.       |
| `.dcd`          |  ✓   |   ✓   | CHARMM / NAMD / OpenMM.                              |
| `.nc`, `.netcdf`|  ✓   |   ✓   | AMBER NetCDF.                                        |
| `.h5`, `.h5md`  |  ✓   |   ✓   | HDF5-based; can carry topology.                      |
| `.pdb` (multi-MODEL) |  ✓   |   ✓   | Text; large and slow, for tiny trajectories only. |

## AlphaFold-aware loading

AlphaFold PDB output stores per-residue pLDDT in the B-factor column.
[`load_alphafold`](../reference/io.md) recognizes this and stores
the values in `protein.metadata["confidence_per_residue"]` instead of
silently mixing them with structural B-factors:

```python
from molforge.io import load_alphafold, is_alphafold_pdb

if is_alphafold_pdb("AF-Q9Y6K9-F1-model_v4.pdb"):
    protein = load_alphafold("AF-Q9Y6K9-F1-model_v4.pdb")
    confidence = protein.metadata["confidence_per_residue"]
```

`load` itself does *not* auto-detect AlphaFold output — it would
require sniffing file contents on every call. Use the explicit
loader when you know you're working with AlphaFold predictions.

## Altloc handling

PDB altloc records (alternate side-chain conformations) are
preserved by default. Pass `altloc=` to choose how to resolve them
at load time:

```python
from molforge.io import load_pdb

p = load_pdb("with_altlocs.pdb", altloc="first")      # default
p = load_pdb("with_altlocs.pdb", altloc="highest")    # by occupancy
p = load_pdb("with_altlocs.pdb", altloc="all")        # keep all rows
p = load_pdb("with_altlocs.pdb", altloc="A")          # by label
```

See [`molforge.io`](../reference/io.md) for the full set of
options and per-format hooks.

## Reference

- [`molforge.io`](../reference/io.md) — full API.
