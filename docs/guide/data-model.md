# Data model

The data model is the heart of `molforge`. Every wrapper, parser, and
analysis function reads from and writes to the same set of types, so
chaining tools doesn't require conversion code.

There are two views of the same data:

- A **hierarchical** view — `Protein` → `Chain` → `Residue` → `Atom`
  — for code that reasons about biology.
- A **linear** view — [`AtomArray`](../reference/core.md) — a
  struct-of-arrays NumPy container for vectorized analysis and ML.

Both views read from the same backing arrays. There is no copy, no
synchronization layer, and no "convert to the other view" step.

## `AtomArray` — the canonical store

`AtomArray` is a flat, NumPy-backed container holding one row per
atom. Columns include:

| Field        | Dtype                  | Notes                                          |
| ------------ | ---------------------- | ---------------------------------------------- |
| `coords`     | `float32` `(N, 3)`     | Cartesian coordinates in Å.                    |
| `atom_name`  | `<U4`                  | Standard PDB atom name, e.g. `"CA"`.           |
| `element`    | `<U2`                  | One- or two-letter symbol.                     |
| `res_name`   | `<U3`                  | Three-letter residue code.                     |
| `res_id`     | `int32`                | Author residue number.                         |
| `chain_id`   | `<U2`                  | Chain identifier.                              |
| `entity_type`| `<U10`                 | `"protein"`, `"ligand"`, `"ion"`, `"water"`.   |
| `b_factor`   | `float32`              | Temperature factor.                            |
| `occupancy`  | `float32`              | Crystallographic occupancy.                    |
| `altloc`     | `<U1`                  | Alternate-location indicator.                  |

(See [`molforge.core.AtomArray`](../reference/core.md) for the
full schema.) Because everything is a NumPy array, masking and
slicing are O(1) and vectorized:

```python
ca = protein.atom_array.coords[protein.atom_array.atom_name == "CA"]
heavy = protein.atom_array.coords[protein.atom_array.element != "H"]
```

## Hierarchical accessors

The hierarchical view is a thin layer of *views* over the same
`AtomArray`. Each `Chain`, `Residue`, and `Atom` holds a `(start,
end)` index pair into the parent array — they don't own data:

```python
protein["A"]                       # Chain (lookup by id)
protein.chains[0]                  # Chain (lookup by position)
protein["A"].residues[42]          # Residue
protein["A"].residues[42].atoms["CA"]   # Atom
```

This means mutating a residue's coordinates mutates the underlying
`AtomArray`, and analyses that operate on the linear view see the
change immediately. It also means `Chain`/`Residue`/`Atom` objects
are cheap to create — they're essentially typed pointers.

## Metadata

`Protein.metadata` is a free-form `dict[str, Any]` for things that
don't fit cleanly into the structural schema: resolution, experimental
method, PDB header lines, prediction confidence (e.g.
`"confidence_per_residue"` set by [`load_alphafold`](../reference/io.md)).

!!! note "API status"
    `metadata` is intentionally untyped today. A typed
    `ProteinMetadata` dataclass is under consideration for a future
    release — see the
    [API audit issue](https://github.com/DoctorDean/molforge/issues).
    Treat keys you set as conventions, not contracts.

## Entity types

The `entity_type` column on `AtomArray` distinguishes protein atoms
from ligands, ions, and waters. PDB and mmCIF parsers set this
automatically; you can also filter manually:

```python
arr = protein.atom_array
protein_atoms = arr[arr.entity_type == "protein"]
ligands       = arr[arr.entity_type == "ligand"]
ions          = arr[arr.entity_type == "ion"]
```

This is what makes heterogeneous content first-class — antibody glycans,
drug-target ligands, structural waters, and metal ions all coexist in
one `Protein` without special-casing.

## Reference

- [`molforge.core`](../reference/core.md) — the full API for the
  data model.
