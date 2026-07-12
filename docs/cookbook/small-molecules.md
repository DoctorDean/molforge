# Work with small molecules

Ligand libraries arrive messy: salts and counter-ions tagged along,
charges written however the vendor felt like it, the same compound drawn
three different ways, and the odd structure that no toolkit will accept.
Before docking, scoring, or featurizing anything you usually want to
bring the set to a consistent form and throw out what you can't use.

molforge reads small molecules as first-class
[`Molecule`](../reference/core.md) objects — chemistry preserved (bonds,
formal charges, aromaticity, stereochemistry), not the bond-less point
cloud the coordinate-only path produces — then cleans, deduplicates, and
filters them. This recipe takes a raw library from a file to a clean,
drug-like, duplicate-free set.

Everything here is RDKit-backed, so it needs the `chem` extra:

```
pip install "molforge[chem]"
```

Every operation is lazy about RDKit — importing molforge never pulls it
in, and an operation without it raises `RDKitNotInstalledError` with an
install hint rather than failing obscurely.

# Read molecules from a file

`read_molecules` reads SDF (`.sdf` / `.mol`) and SMILES (`.smi` /
`.smiles`) into a list of `Molecule`, chemistry intact:

```python
from molforge.io import read_molecules

ligands = read_molecules("vendor_library.sdf")
print(len(ligands), "molecules")

lig = ligands[0]
print(lig.name)              # the SDF title
print(lig.smiles)            # canonical isomeric SMILES
print(lig.formula)           # e.g. "C17H19N3O"
print(lig.molecular_weight)  # g/mol
print(lig.formal_charge, lig.n_heavy_atoms)
```

Each molecule records where it came from in `metadata["source"]`, and
takes its `name` from the record — the SDF title, or the name column of a
SMILES file. Records RDKit can't parse are skipped rather than sinking
the whole read, so one malformed entry in a big SDF doesn't cost you the
rest.

The coordinate-only `read_sdf` (which returns a `Protein`) is untouched
and still there when you want atoms, not chemistry — `read_molecules` is
the chemistry-aware path.

# Build molecules from SMILES

A single SMILES string:

```python
from molforge.core import Molecule

aspirin = Molecule.from_smiles("CC(=O)Oc1ccccc1C(=O)O", name="aspirin")
print(aspirin.inchikey)   # BSYNRYMUTXBXSQ-UHFFFAOYSA-N — a stable structural id
```

Or a whole block, one `SMILES [name]` per line (blanks and `#` comments
skipped):

```python
from molforge.io import read_smiles

mols = read_smiles(
    """
    CCO             ethanol
    CC(=O)O         acetic_acid
    c1ccccc1        benzene
    """
)
```

# Standardize (clean)

`standardize` runs a sensible cleaning pipeline — sanitize, keep the
largest fragment (drops salts and solvents), and neutralize charges where
chemically reasonable:

```python
from molforge.chem import standardize

clean = standardize(aspirin)
print(clean.metadata["standardized"])   # ['cleanup', 'largest_fragment', 'neutralize']
```

The input is never mutated: `standardize` returns a *new* `Molecule`,
preserves its `name`, and records the steps it applied under
`metadata["standardized"]`. Two flags tune the pipeline, and a canonical
tautomer step is available but off by default (it's the slowest and
occasionally surprising):

```python
standardize(mol, desalt=False)          # keep every fragment
standardize(mol, tautomer=True)         # also pick the canonical tautomer
```

When you want just one step, the granular functions are exported too —
`cleanup`, `largest_fragment`, `neutralize`, and `canonical_tautomer` —
each returning a new `Molecule` and recording its step the same way.

# Drop invalid structures and duplicates

`is_valid` reports whether a molecule passes RDKit sanitization (valence,
aromaticity, kekulization), checked on a copy so nothing is mutated — a
predicate you can filter on:

```python
from molforge.chem import is_valid

good = [m for m in ligands if is_valid(m)]
```

`unique` removes structural duplicates, keeping the first occurrence so
input order is preserved. It compares on InChIKey by default (or
`key="smiles"`):

```python
from molforge.chem import unique

distinct = unique(good)                    # by InChIKey
distinct = unique(good, key="smiles")      # or by canonical SMILES
```

# The MoleculeDataset pipeline

`MoleculeDataset` chains these steps into one lazy pipeline. Each
combinator returns a new dataset and nothing runs until you iterate or
`collect`, so a filter that discards most of the library never pays to
standardize what it drops:

```python
from molforge.io import read_molecules
from molforge.chem import MoleculeDataset, standardize
from molforge.validation import Criterion

drug_like = Criterion.lt("molecular_weight", 500) & Criterion.le("formal_charge", 0)

hits = (
    MoleculeDataset(read_molecules("vendor_library.sdf"))
    .map(standardize)     # clean every molecule
    .valid()              # drop anything RDKit rejects
    .dedup()              # remove duplicates by InChIKey, keep first
    .filter(drug_like)    # keep small, non-cationic molecules
    .take(1000)           # at most 1000
    .collect()            # materialize to a list
)
```

The combinators:

- `map(fn)` applies any `Molecule -> Molecule` transform (here
  `standardize`).
- `valid()` keeps only molecules that sanitize cleanly.
- `dedup(key="inchikey")` drops structural duplicates, streaming with a
  running set of seen identities — only the identities, not the
  molecules, stay in memory.
- `filter(criterion)` keeps molecules whose descriptors satisfy a
  criterion (next section).
- `take(n)` keeps the first `n`; it short-circuits, so it's safe over an
  unbounded source.
- `collect()` runs the whole pipeline and returns a `list`.

A dataset is re-iterable exactly when its source is: built over a list it
can be traversed repeatedly; built over a one-shot iterator (see
[Stream a library larger than memory](#stream-a-library-larger-than-memory))
it is single-pass, the same contract as a generator.

# Filter on molecular properties

`filter` reuses a [`Criterion`](../reference/validation.md) — the same
declarative comparison the validation subsystem uses for design metrics —
as a molecule filter. A criterion is built from a metric name, a
comparison, and a threshold, and composes with `&`, `|`, and `~`:

```python
from molforge.validation import Criterion

fragment_like = (
    Criterion.le("molecular_weight", 300)
    & Criterion.le("n_heavy_atoms", 22)
    & Criterion.eq("formal_charge", 0)
)

fragments = MoleculeDataset(ligands).filter(fragment_like).collect()
```

The metric names a criterion may reference are the molecule descriptors
in `DESCRIPTOR_NAMES` — `molecular_weight`, `formal_charge`, `n_atoms`,
and `n_heavy_atoms`. `filter` validates the referenced names up front
(a typo like `Criterion.lt("mw", 500)` raises immediately with the
available names) and computes only the descriptors a given criterion
actually uses.

The mapping is public, so you can score a molecule directly:

```python
from molforge.chem import molecule_descriptors, DESCRIPTOR_NAMES

molecule_descriptors(aspirin)
# {'molecular_weight': 180.16, 'formal_charge': 0, 'n_atoms': 13, 'n_heavy_atoms': 13}

molecule_descriptors(aspirin, names=["molecular_weight"])   # compute just one
sorted(DESCRIPTOR_NAMES)
```

`n_atoms` counts the atoms actually in the graph, so for a freshly parsed
molecule it matches `n_heavy_atoms`; it only rises above the heavy-atom
count once hydrogens are made explicit.

# Stream a library larger than memory

`read_molecules` materializes the whole file; its lazy counterpart
`iter_molecules` yields one `Molecule` at a time (SDF via RDKit's
`ForwardSDMolSupplier`, SMILES line by line), so a multi-gigabyte library
never has to fit in RAM. Feed it straight into a dataset and let `take`
stop the stream once you have enough:

```python
from molforge.io import iter_molecules
from molforge.chem import MoleculeDataset, standardize
from molforge.validation import Criterion

first_hits = (
    MoleculeDataset(iter_molecules("enormous_library.sdf"))
    .map(standardize)
    .valid()
    .filter(Criterion.lt("molecular_weight", 500))
    .take(50)
    .collect()   # reads only far enough into the file to find 50
)
```

Because the source is a one-shot iterator, this dataset is single-pass —
`collect()` it once. Wrap the readers' output in a `list` first if you
need to traverse the set more than once.

`iter_smiles` is the streaming counterpart to `read_smiles` for SMILES
blocks. Both readers resolve the file format eagerly (a bad extension
raises right away) but parse records lazily, so the first molecule comes
back without waiting for the last.
