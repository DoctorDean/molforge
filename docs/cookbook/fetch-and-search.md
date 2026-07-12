# Fetch and search databases

Not every structure or compound starts as a file on disk. molforge can pull
them straight from the public databases — the RCSB Protein Data Bank, the
AlphaFold Protein Structure Database, and ChEMBL — and hand the results to the
rest of the library. The structure side is pure standard library (no extra
dependency); the ChEMBL side builds `Molecule` objects and so needs RDKit.

This recipe covers fetching by ID, fetching in bulk, searching the PDB, and
pulling compounds from ChEMBL.

The structure functions (`fetch`, `fetch_many`, `search_rcsb`) work out of the
box. `fetch_chembl` builds molecules, so it needs the `chem` extra:

```
pip install "molforge[chem]"
```

Everything here does live network I/O: a missing ID or a dropped connection
surfaces as `OSError`, and every call takes a `timeout` (seconds, default 30).

# Fetch a structure by ID

`fetch` downloads one structure and parses it into a `Protein`:

```python
from molforge.io import fetch

protein = fetch("1UBQ")                       # RCSB, PDB format
haemoglobin = fetch("4HHB", format="cif")     # same source, mmCIF
model = fetch("P00520", source="alphafold")   # AlphaFold DB, by UniProt accession
```

RCSB IDs are 4-character PDB codes (case-insensitive); AlphaFold takes a
UniProt accession. Both `"pdb"` and `"cif"` formats are supported.

# Fetch a whole set

`fetch_many` pulls a list of IDs in one call, returning `Protein`s in the
order you asked for them:

```python
from molforge.io import fetch_many

kinases = fetch_many(["1ATP", "2CPK", "1STC"])
```

Downloads are sequential (the servers rate-limit, and it keeps the dependency
to the standard library). By default a single bad ID raises and stops the
batch; pass `on_error="skip"` to tolerate the odd 404 in a large list and get
back whatever resolved:

```python
structures = fetch_many(ids, on_error="skip")   # failed IDs simply don't appear
```

# Search the PDB

`search_rcsb` turns a free-text query into a relevance-ranked list of PDB IDs
via the RCSB Search API. Chain it into `fetch_many` to go from a keyword to
structures:

```python
from molforge.io import search_rcsb, fetch_many

ids = search_rcsb("SARS-CoV-2 main protease", limit=10)
# -> ['6LU7', '6M03', ...]  most relevant first

structures = fetch_many(ids, on_error="skip")
```

`limit` caps how many IDs come back (the top hits); a query that matches
nothing returns an empty list.

# Pull compounds from ChEMBL

`fetch_chembl` downloads a compound by ChEMBL ID and returns it as a
chemistry-aware `Molecule`, built from ChEMBL's canonical SMILES:

```python
from molforge.io import fetch_chembl

aspirin = fetch_chembl("CHEMBL25")
print(aspirin.name)                 # 'ASPIRIN' — ChEMBL's preferred name
print(aspirin.smiles)
print(aspirin.metadata["source"])   # 'chembl'
print(aspirin.metadata["chembl_id"])
```

Entries that carry no small-molecule structure — a biotherapeutic such as an
antibody — raise `ValueError`. For a set, `fetch_chembl_many` mirrors
`fetch_many`'s error policy, so `on_error="skip"` drops IDs that fail to
download *or* have no structure:

```python
from molforge.io import fetch_chembl_many

actives = fetch_chembl_many(["CHEMBL25", "CHEMBL521", "CHEMBL1201585"], on_error="skip")
```

# From a database straight into a pipeline

Because `MoleculeDataset` accepts any iterable of `Molecule`, a ChEMBL pull
drops straight into the ingest → clean → filter pipeline from
[Work with small molecules](small-molecules.md):

```python
from molforge.io import fetch_chembl_many
from molforge.chem import MoleculeDataset, standardize
from molforge.validation import Criterion

hits = (
    MoleculeDataset(fetch_chembl_many(ids, on_error="skip"))
    .map(standardize)
    .valid()
    .dedup()
    .filter(Criterion.lt("molecular_weight", 500) & Criterion.le("formal_charge", 0))
    .collect()
)
```

The structure side composes just as directly: a fetched `Protein` is an
ordinary `Protein`, ready for validation, prep, docking, or anything else in
the library — so `fetch_many(search_rcsb(...))` is a one-liner from a search
term to a set of structures to work on.

# Network, timeouts, and errors

Every call is a real HTTP request, so plan for the network:

- A non-existent ID or an unreachable server raises `OSError` with a message
  naming the source and, for HTTP errors, the status code.
- `timeout` bounds each request; raise it for slow links or large downloads.
- In the bulk calls, `on_error="skip"` is the difference between "one bad ID
  sinks the batch" and "give me whatever resolves." Skipped IDs are dropped
  from the result, so check its length if you need to know how many came back.
