# Architecture overview

This document describes the high-level architecture of `biocore`.

## Design goals

1. **One canonical data model** for protein structures, shared across all subpackages.
2. **Two complementary views** of the same data:
   - *Hierarchical* (`Protein` → `Chain` → `Residue` → `Atom`) — for biology-aware reasoning.
   - *Linear* (`AtomArray`) — flat NumPy arrays for vectorized analysis and ML.
3. **Composability over orchestration** — `biocore` is a library; users compose modules in their own scripts/notebooks. There is no runtime or DAG engine.
4. **Wrappers, not reimplementations** — external engines (folding, docking, MD) are wrapped behind small, typed interfaces. The wrappers do conversion to/from the canonical data model; they do not reimplement the underlying science.
5. **Plugin-friendly** — third parties can extend `biocore` without forking it by exposing entry points under the `biocore.plugins` group.

## Module map

```
biocore
├── core          # Data model: Protein, Chain, Residue, Atom, AtomArray
├── io            # Parsers/writers: PDB, mmCIF, FASTA, MOL2, SDF, trajectories
├── sequence      # Sequence-level ops: align, mutate, composition
├── structure     # Geometry & analysis: RMSD, SASA, contacts, DSSP
├── md            # Trajectory I/O and simulation interface
├── docking       # Pose handling and engine-agnostic docking abstractions
├── ml            # Featurizers and tensor views
├── metrics       # Task-level scoring: TM-score, lDDT, GDT-TS
├── plugins       # Plugin registry and entry-point discovery
└── wrappers      # Thin wrappers around external engines
    ├── folding   # AlphaFold, ESMFold, Boltz, (Py)Rosetta
    ├── docking   # AutoDock Vina, DiffDock
    └── md        # OpenMM, GROMACS
```

## Data model

```
Protein
├── name: str                        # e.g. "1UBQ"
├── metadata: dict                   # resolution, method, header
├── chains: list[Chain]
│   └── Chain
│       ├── chain_id: str            # "A"
│       └── residues: list[Residue]
│           └── Residue
│               ├── name: str        # "ALA"
│               ├── seq_id: int
│               ├── insertion_code: str
│               └── atoms: list[Atom]
│                   └── Atom
│                       ├── name: str   # "CA"
│                       ├── element: str
│                       ├── coord: ndarray(3,)
│                       └── ...
├── atom_array: AtomArray            # linear/flat view
└── sequence: str                    # one-letter code
```

The linear view (`AtomArray`) holds the same atoms as a set of parallel
NumPy arrays of shape `(N,)` (and `(N, 3)` for coords), enabling
vectorized operations, zero-copy hand-off to PyTorch, and SIMD-friendly
analysis. The hierarchical and linear views are kept consistent.

## Plugin architecture

The plugin registry is a simple mapping keyed by `(kind, name)` pairs.
Discovery walks `importlib.metadata` entry points under the
`biocore.plugins` group. A third-party package looks like:

```toml
# external_pkg/pyproject.toml
[project.entry-points."biocore.plugins"]
my_docker = "external_pkg:register"
```

```python
# external_pkg/__init__.py
from biocore.plugins import register_engine
from .my_docker import MyDocker

def register() -> None:
    register_engine("my_docker", MyDocker)
```

After `pip install external_pkg`, users get:

```python
from biocore.plugins import discover, get
discover()
engine_cls = get("engine", "my_docker")
```

## Stability and versioning

`biocore` follows SemVer. While we are pre-1.0, expect breaking changes
between minor releases — they will be documented in `CHANGELOG.md`. The
public API is everything not prefixed with `_`.
