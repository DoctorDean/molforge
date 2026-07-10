# Binding free energy

`molforge` estimates protein–ligand binding free energy through several
methods that span two orders of magnitude in cost — from endpoint
MM/GBSA to rigorous alchemical FEP. This page explains how they fit
together. The unifying idea is that **every method, however cheap or
expensive, speaks the same value types**, so the code that ranks,
compares, and reports affinities is identical no matter which method
produced the numbers.

## One currency: the value types

`molforge.freeenergy` holds the shared vocabulary — no engines, no
parsing, just the types every producer returns and every consumer reads:

```
molforge.freeenergy
├── FreeEnergyResult        # one ΔG: value, uncertainty, method, provenance,
│                           #   optional components / decomposition
├── FreeEnergyComponents    # per-term breakdown (vdw, elec, polar, nonpolar, entropy)
├── DeltaDeltaG             # a signed difference between two ligands (ΔΔG)
├── FreeEnergyRanking       # orders results; computes pairwise ΔΔG
├── ResidueContribution     # one residue's share of ΔG_bind
├── Decomposition           # per-residue hotspot map (.hotspots(...))
└── MMGBSAEngine (ABC)      # the endpoint-engine interface
    └── MMGBSAEngineNotInstalledError
```

A `FreeEnergyResult` is the atom of the subsystem: a ΔG with an
**uncertainty that is never optional**, the `method` that produced it, a
`Provenance` recording how, and optional extras — a `FreeEnergyComponents`
breakdown and, for MM/GBSA runs, a per-residue `Decomposition`. Because a
relative-FEP ΔΔG and an MM/GBSA endpoint ΔG both surface as the same
types, they rank through one `FreeEnergyRanking`.

## Two families of producer

Everything that *makes* a result is either an **engine** or an **ingest
function**, and which one it is follows directly from cost.

**Engines** run an external tool end to end. Endpoint methods are cheap
enough to launch from `molforge` — they post-process a trajectory you
already have — so they get a full engine that invokes the binary, parses
its output, caches, and returns a `FreeEnergyResult`:

```
MMGBSAEngine (ABC)          run(trajectory, *, receptor, ligand, solvent_model=...)
├── AmberMMGBSA             MMPBSA.py
└── GromacsMMGBSA           gmx_MMPBSA
```

**Ingest functions** read an analysis you produced elsewhere. Alchemical
methods need dedicated simulations and specialized samplers, so `molforge`
does *not* reimplement them — it ingests their results and does the
thermodynamic bookkeeping:

```
from_alchemlyb / from_delta_f     an FEP/TI leg (alchemlyb estimator) -> FreeEnergyResult
from_cinnabar                     a solved ΔΔG network -> {ligand: FreeEnergyResult}
relative_binding_free_energy      two legs -> DeltaDeltaG   (RBFE cycle)
absolute_binding_free_energy      two legs + restraint -> FreeEnergyResult  (ABFE cycle)
```

This split is the central design decision: **wrap what is cheap to run,
ingest what is not.** It keeps `molforge` free of heavy simulation
dependencies while still covering the rigorous end of the spectrum.

## The three tiers

| Tier | Method | Producer | Yields | Use when |
|------|--------|----------|--------|----------|
| Endpoint | MM/GBSA, MM/PBSA | `AmberMMGBSA`, `GromacsMMGBSA` | absolute-ish ΔG (+ hotspots) | triaging a large set from trajectories you have |
| Relative alchemical | RBFE (FEP/TI) | `from_alchemlyb` + `relative_binding_free_energy` | `DeltaDeltaG` per edge | a congeneric series, rigorously |
| Absolute alchemical | ABFE (double decoupling) | `from_alchemlyb` + `absolute_binding_free_energy` | absolute ΔG_bind | one ligand, no congeneric reference |

They compose: a common workflow triages hundreds of poses with MM/GBSA,
then spends FEP on the shortlist — and because both land in
`FreeEnergyResult` / `FreeEnergyRanking`, the ranking and reporting code
doesn't change.

## How a result flows

```
   producers                          value types                consumers
 ┌────────────────────────┐
 │ AmberMMGBSA.run()      ─┼──┐   (may attach a Decomposition)
 │ GromacsMMGBSA.run()    ─┼──┤
 ├────────────────────────┤  │
 │ from_alchemlyb()       ─┼──┼──▶  FreeEnergyResult ──┐
 │ from_delta_f()         ─┼──┤                         │
 │ from_cinnabar()        ─┼──┤                         ├──▶ FreeEnergyRanking
 │ absolute_binding_…()   ─┼──┘                         │      .ranked  .best
 ├────────────────────────┤                             │      .delta_delta_g() ─┐
 │ relative_binding_…()   ─┼───────▶  DeltaDeltaG ───────┘                        │
 └────────────────────────┘             ▲                                         │
                                        └─────────────────────────────────────────┘
```

`FreeEnergyRanking` both *consumes* a set of `FreeEnergyResult`s and
*produces* a `DeltaDeltaG` on demand (`.delta_delta_g(a, b)`), so the two
value types are the whole surface a caller needs.

## Module layout

```
molforge.freeenergy               # value types + the MMGBSAEngine ABC (above)

molforge.wrappers.freeenergy      # the producers
├── _common.py                    # shared: input building, result/decomp parsing,
│                                 #   selection→mask, provenance helpers
├── amber.py                      # AmberMMGBSA, parse_mmpbsa_dat, parse_mmpbsa_decomp
├── gromacs.py                    # GromacsMMGBSA, parse_gmx_mmpbsa_dat, parse_gmx_mmpbsa_decomp
├── alchemlyb.py                  # from_alchemlyb, from_delta_f, the two cycle helpers
└── cinnabar.py                   # from_cinnabar
```

The endpoint engines share more than they differ — both drive a
`MMPBSA.py`-family tool — so the input builder, the results and
decomposition parsers, and the provenance helpers live once in `_common`;
`amber.py` and `gromacs.py` hold only what is genuinely tool-specific
(mask vs. index-group selection, the Δ-prefixed 5-column results layout,
the gmx `Location` column).

## Cross-cutting choices

**No heavy dependencies.** `alchemlyb`, `cinnabar`, and `pandas` are never
imported — the ingest functions duck-type through the small surface they
need (`numpy.asarray`, `.attrs`, `.columns` / `.to_dict`). Engines lazy
-check for their binary and raise `MMGBSAEngineNotInstalledError` if it's
absent, rather than at import time. You install only what your chosen
method needs.

**Uncertainty is structural.** Every `FreeEnergyResult` carries an
uncertainty (construction rejects a negative one); the cycle helpers
propagate in quadrature; and `FreeEnergyRanking` is meant to be read with
the error bars in mind — a difference within its uncertainty is a tie, not
a win. The subsystem is built to discourage over-reading a single number.

**Provenance and caching.** Engines attach a `Provenance` (engine,
parameters, inputs, parent) and cache keyed on it, so an identical re-run
returns instantly without touching the tool. Parameters that change the
result — `solvent_model`, the masks, `idecomp` — are part of the key;
`idecomp` and `print_res` enter it only when a decomposition is requested,
so turning decomposition on doesn't invalidate a plain run's cache.

**Decomposition rides along.** When an engine is asked for a per-residue
decomposition (`idecomp=…`), the `Decomposition` is attached to the same
`FreeEnergyResult` and cached with it — the affinity and the map of which
residues produce it stay together.

## Choosing a method

- **Triaging many candidates** from trajectories you already have →
  MM/GBSA. Cheap, and it answers *where* the affinity comes from via
  decomposition. See [Rank binders with MM/GBSA](../cookbook/ranking-binders.md).
- **A congeneric series, rigorously** → relative FEP. Calibrated ~1
  kcal/mol ΔΔG between similar ligands. See
  [Rank binders with FEP](../cookbook/rank-binders-fep.md).
- **A single ligand with no analog** (fragment, scaffold hop, or a number
  to compare with K_d) → absolute FEP. See
  [Absolute binding free energy with FEP](../cookbook/absolute-binding-fep.md).
- **Both** → triage with MM/GBSA, then FEP the shortlist; the shared types
  make the hand-off seamless.

## Reference

- [`molforge.freeenergy`](../reference/freeenergy.md) — the value types and
  the engine ABC.
- [`molforge.wrappers.freeenergy`](../reference/wrappers/freeenergy.md) — the
  engines, ingest functions, and cycle helpers.
