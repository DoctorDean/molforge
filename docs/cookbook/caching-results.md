# Caching results

molforge caches results from expensive engine calls automatically.
Folding (ESMFold, Boltz, Chai-1), sequence design (ProteinMPNN,
ESM-IF1), and docking (Vina, Gnina, DiffDock) all participate. When
you re-run a computation with identical inputs and parameters, the
cached result returns in milliseconds instead of the engine running
again.

The cache key is derived from `Provenance` — the same record that
captures engine + parameters + inputs + parent chain — so it
invalidates correctly when *anything* about the computation
changes, and chains correctly through multi-step pipelines.

## Default behaviour

The cache is on by default and lives at `~/.cache/molforge/`. The
first call to any wrapped engine runs the compute and stores the
result; the second identical call returns the cached result
without touching the model:

```python
from molforge.wrappers.folding import ESMFold

engine = ESMFold()
p1 = engine.predict("MKQHKAMIVAL...")  # runs the model
p2 = engine.predict("MKQHKAMIVAL...")  # cached — milliseconds
```

A "second identical call" means: same engine, same constructor
parameters, same inputs (sequence or `ComplexSpec`), same parent
provenance. Change *any* of those and the cache key changes:

```python
from molforge.wrappers.folding import Boltz
from molforge.folding import ComplexSpec

spec = ComplexSpec.protein_ligand(
    protein_sequence="MVTPEG...",
    ligand_smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
)
Boltz(use_msa_server=False).predict_complex(spec)  # cache slot A
Boltz(use_msa_server=True).predict_complex(spec)   # cache slot B (different params)
```

The cache key includes molforge's major.minor version, so a `0.4.x → 0.5.0`
upgrade invalidates everything transparently. No manual cleanup
needed for routine version bumps.

## Cascading invalidation

When you chain engines together (RFdiffusion → ProteinMPNN, fold →
dock, etc.), upstream changes invalidate downstream caches
automatically — the upstream `Provenance` is the downstream
`Provenance.parent`, and the parent chain participates in the key:

```python
from molforge.wrappers.generative import RFdiffusion, ProteinMPNN

# Run 1: produces RFdiffusion(seed=42) → ProteinMPNN(seed=7).
#   Both calls hit empty cache slots; results cached.
backbones = RFdiffusion(seed=42).generate(...)
designs1 = ProteinMPNN(seed=7).generate(backbones[0])

# Run 2: same upstream + downstream → both cached.
backbones = RFdiffusion(seed=42).generate(...)
designs2 = ProteinMPNN(seed=7).generate(backbones[0])  # cache hit

# Run 3: upstream changes → downstream invalidates.
backbones = RFdiffusion(seed=99).generate(...)         # new backbone
designs3 = ProteinMPNN(seed=7).generate(backbones[0])  # cache miss
```

You don't have to think about this. The parent chain plumbs through.

## Docking

Docking engines cache the whole `DockingResult` — every pose, the
receptor, and the run's `Provenance`. The same key rules apply: the
search box (`center`, `box_size`), `exhaustiveness`, scoring choice,
and the receptor/ligand identity all participate, so re-docking the
same ligand into the same box is a hit:

```python
from molforge.wrappers.docking import Vina

vina = Vina(seed=42)
box = {"center": (12.0, 4.5, -8.0), "box_size": (20.0, 20.0, 20.0)}

r1 = vina.dock("receptor.pdbqt", "ligand.sdf", **box)  # runs Vina
r2 = vina.dock("receptor.pdbqt", "ligand.sdf", **box)  # cached
```

The cache is consulted *before* the engine's binary is even
required, so a machine without Vina / gnina / DiffDock installed
can still replay a result that was computed elsewhere and shared
(copy the entry directory into `~/.cache/molforge/`).

Because the receptor's `Provenance` is the docking result's
`Provenance.parent`, a fold-then-dock pipeline invalidates correctly:
re-fold the receptor with a different engine or sequence and the
downstream dock is a miss, even with an identical box.

```python
from molforge.wrappers.folding import ESMFold
from molforge.wrappers.docking import Gnina

receptor = ESMFold().predict("MVTPEG...")   # cached folding result
poses = Gnina().dock(receptor, "ligand.sdf", center=(10.0, 5.0, -2.0))
poses2 = Gnina().dock(receptor, "ligand.sdf", center=(10.0, 5.0, -2.0))  # hit
```

MD trajectories remain the one deliberately-uncached engine output
(see the bottom of this page).

## What's in the cache directory

```
~/.cache/molforge/
├── 49d8f89cd25af146.../   # one dir per entry, named by SHA-256 of the key
│   ├── type               # "protein", "designed_sequences", or "docking_result"
│   ├── meta.json          # name + metadata (provenance, scalars, ...)
│   ├── structure.cif      # the AtomArray as mmCIF (for Protein)
│   └── arrays.npz         # numpy arrays from metadata
├── 7c9c4317f1e0a4b2.../
│   ├── type
│   ├── payload.json       # for designed_sequences entries
│   └── arrays.npz
├── a1b2c3d4e5f60718.../   # a docking_result entry
│   ├── type
│   ├── payload.json       # engine, per-pose score/rank/rmsd, metadata
│   ├── receptor.cif       # receptor structure (if present)
│   ├── pose_0.cif         # one cif per pose ligand
│   ├── pose_1.cif
│   └── arrays.npz
└── ...
```

Entries are inspectable with standard tools: `cat meta.json`,
`molforge view structure.cif`, `python -c "import numpy as np; print(np.load('arrays.npz').files)"`,
etc. No pickle, no binary blobs you can't read.

## Disabling the cache

Three ways, from coarsest to finest:

**Global, via env var:**

```bash
export MOLFORGE_CACHE=disabled
```

The string is also matched against `"0"`, `"false"`, `"off"`, `"no"`
(case-insensitive). Anything else is treated as enabled.

**Custom location**, via env var:

```bash
export MOLFORGE_CACHE_DIR=/scratch/molforge-cache
```

Useful on shared clusters where `~/` has tight quotas. Respects
`$XDG_CACHE_HOME` if set.

**Per-call**, by constructing a disabled `Cache` instance — the
engines consult `get_default_cache()` so to override per-call,
override the singleton:

```python
import molforge.cache as cache
cache._default_cache = cache.Cache(enabled=False)
# Any engine call now bypasses the cache.
```

For tests, the project's `tests/conftest.py` has an `_isolate_cache`
fixture that points every test at a per-test temp dir, so test runs
never pollute (or read from) the real cache.

## Clearing the cache

```python
from molforge.cache import get_default_cache

n_removed = get_default_cache().clear()
print(f"Cleared {n_removed} entries")
```

Or just `rm -rf ~/.cache/molforge/` if you prefer. `clear()` only
removes entries named with the 64-character hex pattern — anything
else you've put in the cache directory (notes, sub-projects,
whatever) is left alone.

## When the cache *isn't* what you want

- **MD trajectories**: deliberately uncached. Multi-GB per
  simulation; use the upstream MD framework's checkpointing instead.
- **Profiling / benchmarking**: a cached call returns near-instantly,
  which skews timing measurements. Disable with
  `MOLFORGE_CACHE=disabled` for benchmark runs.
- **Diagnosing nondeterminism**: if you suspect an engine isn't
  reproducible (different output across identical-input calls),
  the cache will hide that from you. Disable temporarily, run the
  computation twice, compare.

## Safety

- **Corrupted entries** (missing files, parse errors) are silently
  treated as cache misses. Your computation runs as if there was
  no cache entry; no exceptions propagate from cache lookups.
- **Writes are atomic**: serialization goes to a `.tmp` directory
  and renames atomically at the end. A `Ctrl-C` during writing
  leaves the cache directory clean.
- **Concurrent processes** writing the same key are safe in the
  sense that the result is correct. Two processes computing the
  same thing will both succeed, with one of their writes "winning";
  the content is deterministic so it doesn't matter which.
