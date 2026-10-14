# Performance benchmarks

molforge's analysis stack is written from scratch in NumPy (no C extensions
to build), so it's worth knowing what's fast and what isn't. The numbers
below come from molforge's own benchmark suite
([`tests/benchmarks/`](https://github.com/DoctorDean/molforge/tree/master/tests/benchmarks)),
run on a **200-residue** idealized structure.

!!! note "Read these as orders of magnitude"
    Absolute times depend on your machine, Python build, and NumPy/BLAS.
    These were measured single-threaded on an Apple-silicon laptop
    (Python 3.12). What's stable across machines is the *ranking* — which
    operations are microseconds and which are hundreds of milliseconds.
    Reproduce them yourself with the command at the bottom.

## Analysis stack (200 residues)

| Operation | Function | Mean time | ≈ ops/sec |
| --------- | -------- | --------: | --------: |
| RMSD (no alignment) | `rmsd(..., align=False)` | **0.017 ms** | 58,000 |
| RMSD (Kabsch superposition) | `rmsd` | **0.067 ms** | 15,000 |
| Distance map | `distance_map` | **1.3 ms** | 740 |
| Contact map | `contact_map` | **1.4 ms** | 720 |
| lDDT | `lddt` | **4.2 ms** | 240 |
| Smith-Waterman (local align) | `align(mode="local")` | **104 ms** | 10 |
| Needleman-Wunsch (global align) | `align(mode="global")` | **107 ms** | 9 |
| DSSP | `dssp` | **251 ms** | 4 |

## What this tells you

- **The vectorized geometry is effectively free.** RMSD, distance/contact
  maps, and lDDT are sub-millisecond to low-single-digit milliseconds —
  fine to call in a tight loop (per docking pose, per MD frame, per design
  candidate).
- **Pairwise alignment and DSSP are the slow paths** (~0.1–0.25 s each).
  They're pure-Python dynamic-programming / hydrogen-bond loops over the
  whole structure. Still fast enough for interactive use and per-structure
  gating, but if you're processing tens of thousands of structures, batch
  them (see [`molforge.parallel`](../reference/parallel.md)) rather than
  calling them in the innermost loop.
- These two are exactly the **Numba/Rust hot-path candidates** flagged in
  the [roadmap](roadmap.md) — deferred until a real workload needs them,
  because correctness (validated against independent oracles) came first.

## Engine wrappers

Wrapped engines (ESMFold, Vina, OpenMM, …) are dominated by the engine
itself — model inference, the docking search, the MD integrator — not by
molforge's thin wrapper. molforge's overhead there is the format conversion
in and out (typically milliseconds), negligible against a fold that takes
seconds to minutes. The content-addressed [cache](../reference/cache.md)
turns a repeat call into a millisecond lookup, and `pipeline.yaml`
[replay](../guide/reproducibility.md) skips a step entirely on a cache hit.

## Reproduce

```bash
pip install pytest-benchmark
pytest tests/benchmarks/ --benchmark-only --benchmark-columns=mean,median,ops
```

The inputs are synthesized parametrically (an idealized helix with full
backbone atoms), so no large PDB files are needed and the suite is
deterministic. Pass a different residue count by editing the fixtures in
`tests/benchmarks/conftest.py`.
