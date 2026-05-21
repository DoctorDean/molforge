# Engine wrappers

`molforge` doesn't reimplement folding models, MD engines, or docking
binaries. It wraps them behind small, typed interfaces so engines are
swappable from user code.

There are four wrapper categories, each with a stable base class and
one or more concrete engines:

| Category    | Base class           | Engines                                    |
| ----------- | -------------------- | ------------------------------------------ |
| Folding     | `FoldingEngine`      | `ESMFold`, `AlphaFold` *(stubs: Boltz, Rosetta)* |
| Docking     | `DockingEngine`      | `Vina` *(stub: DiffDock)*                  |
| MD          | `MDEngine`           | `OpenMM` *(stub: GROMACS)*                 |
| Generative  | `GenerativeEngine`   | `RFdiffusion`, `ProteinMPNN`               |

Each base class defines a small `predict` / `dock` / `simulate` /
`generate` method that returns a `molforge` type (`Protein`,
`DockingResult`, `Trajectory`, etc.) — so a pipeline that uses one
engine can swap to another with a one-line change.

## Lazy heavy imports

Wrappers never import their heavy dependencies (`torch`,
`transformers`, `vina`, `openmm`, `colabfold`) at module load time.
Heavy imports happen inside method bodies, so `import molforge.wrappers`
stays fast and the dependencies are only required when the
corresponding engine is actually used.

If you try to call a wrapper whose dependency is missing, the wrapper
raises an `*EngineNotInstalledError` with a clear message:

```python
from molforge.wrappers.folding import ESMFold

try:
    folded = ESMFold().predict("MKTV...")
except FoldingEngineNotInstalledError as e:
    print(e)   # "ESMFold requires torch + transformers; pip install 'molforge[ml]'"
```

## A worked example — folding

```python
from molforge.wrappers.folding import ESMFold
from molforge.io import save

engine = ESMFold(device="cuda")          # or "cpu"
folded = engine.predict("MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVS...")

# Per-residue pLDDT lives in metadata (same convention across all
# folding engines — see molforge.wrappers.folding docstring)
plddt = folded.metadata["confidence_per_residue"]

save(folded, "candidate.pdb")
```

To switch to AlphaFold, replace one line:

```python
from molforge.wrappers.folding import AlphaFold

engine = AlphaFold(...)
folded = engine.predict("MKTV...")    # same return type, same metadata key
```

That's the whole point of the wrapper pattern.

## The escape hatch

For the rare case where you need engine-specific functionality not in
the base interface, `MDEngine`-derived simulations expose
`Simulation.engine_handle: object | None` — the raw underlying object
(an `openmm.Simulation`, a `gromacs.Process`, etc.). This is an
explicit "you're going off the supported surface" handle; use it
sparingly, and only when the wrapper genuinely doesn't expose what
you need.

!!! note "API status"
    `engine_handle` is intentionally typed as `object | None` to
    signal that the contents are engine-specific. The handle is
    retained for the lifetime of the `Simulation`; it's never
    invalidated underneath you.

## Reference

- [`molforge.wrappers`](../reference/wrappers.md) — all engine
  wrappers with full signatures.
- Generative wrappers used together (RFdiffusion → ProteinMPNN →
  ESMFold): see the
  [`de_novo_design.ipynb`](https://github.com/DoctorDean/molforge/blob/main/notebooks/examples/de_novo_design.ipynb)
  notebook.
