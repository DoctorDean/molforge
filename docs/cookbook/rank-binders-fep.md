# Rank binders with FEP

When you need rigorous relative affinities — is analog B really ~1
kcal/mol tighter than the lead, or is that MM/GBSA rank just noise? —
relative free energy perturbation (**RFEP**) is the tool. molforge
doesn't run the alchemical simulations or fit the estimators; it
**ingests the analysis**. You run FEP and analyze each leg with
[alchemlyb](https://alchemlyb.readthedocs.io); molforge turns those
estimators into the same `FreeEnergyResult` and `DeltaDeltaG` types the
MM/GBSA path uses, closes the thermodynamic cycle, and ranks.

This is the rigorous sibling of [Rank binders with
MM/GBSA](ranking-binders.md): slower and more setup, but calibrated
relative affinities rather than a triage signal.

## Requirements

```bash
pip install "molforge"        # the ingest adds no deps of its own
pip install alchemlyb pymbar  # you bring these to analyze the FEP legs
```

molforge never imports alchemlyb — you fit the estimators, molforge reads
their results — so it isn't a molforge dependency.

## The cycle

A relative FEP edge perturbs one ligand into another (`lead` → `analog`)
along two legs: bound to the receptor (the **complex** leg) and free in
solution (the **solvent** leg). The thermodynamic cycle gives the
relative binding free energy:

```
ΔΔG_bind(lead → analog) = ΔG_complex − ΔG_solvent
```

A single leg's ΔG is *not* a binding affinity — only the cycle is. So the
unit of work is a pair of legs per edge.

## The recipe

For each analog you've run an FEP edge against the lead and analyzed both
legs with alchemlyb (an MBAR/BAR/TI estimator per leg):

```python
from molforge.wrappers.freeenergy import from_alchemlyb, relative_binding_free_energy

# edges: {analog_name: (complex_estimator, solvent_estimator)}, each a
# fitted alchemlyb estimator for that leg.
ddgs = {
    name: relative_binding_free_energy(
        from_alchemlyb(complex_est),   # ΔG of lead→analog in the complex
        from_alchemlyb(solvent_est),   # ΔG of lead→analog in solvent
        reference="lead",
        other=name,
    )
    for name, (complex_est, solvent_est) in edges.items()
}
```

`from_alchemlyb` reads the estimator's `delta_f_` / `d_delta_f_`, takes
the full first-state → last-state transformation, and converts to
kcal/mol using the temperature and unit alchemlyb records in the
DataFrame's `.attrs`. If those attributes were stripped (a parquet round
trip, say), pass `temperature=` explicitly.

## Reading an edge

Each `DeltaDeltaG` is the signed relative affinity with propagated error:

```python
ddg = ddgs["analog_3"]
print(f"ΔΔG(analog_3 – lead) = {ddg.value:+.2f} ± {ddg.uncertainty:.2f} kcal/mol")
print("tighter:", ddg.tighter)
```

`value` is `ΔG_bind(analog) − ΔG_bind(lead)`; negative means the analog
binds more tightly. The error is the two legs' standard errors combined
in quadrature. As everywhere in molforge, treat a difference within its
uncertainty as a tie — RFEP's useful resolution is around 1 kcal/mol, so
a 0.3 ± 0.4 edge is not a win.

## Ranking a star map

If every edge shares the lead as reference (a *star map*), the ΔΔGs are
already on one scale — the lead's. Anchor the lead at zero and rank:

```python
from molforge.freeenergy import FreeEnergyRanking, FreeEnergyResult

results = {"lead": FreeEnergyResult(delta_g=0.0, uncertainty=0.0, method="FEP (ΔΔG)")}
for name, ddg in ddgs.items():
    results[name] = FreeEnergyResult(
        delta_g=ddg.value, uncertainty=ddg.uncertainty, method="FEP (ΔΔG)"
    )

ranking = FreeEnergyRanking(results)
for name, r in ranking.ranked:
    print(f"{name:10s} ΔΔG = {r.delta_g:+.2f} ± {r.uncertainty:.2f} kcal/mol")
```

These are binding free energies *relative to the lead*, not absolute
ΔG_bind — the whole map floats on the lead's unknown baseline, which is
exactly what a congeneric optimization cares about. For a non-star
network (edges between arbitrary pairs) you'd solve the graph for
per-ligand estimates first; molforge gives you the per-edge `DeltaDeltaG`
inputs, and a tool like [cinnabar](https://github.com/OpenFreeEnergy/cinnabar)
does the network fit.

## Absolute FEP and other estimators

`from_alchemlyb` works with any alchemlyb estimator, so TI or BAR legs
ingest the same way (the `method` label follows the estimator's name).

Absolute binding FEP (double decoupling) has its own cycle helper,
`absolute_binding_free_energy`, which combines the two decoupling legs and
a standard-state restraint correction into an *absolute* ΔG_bind:

```python
from molforge.wrappers.freeenergy import absolute_binding_free_energy, from_alchemlyb

dg_bind = absolute_binding_free_energy(
    from_alchemlyb(complex_decoupling_est),   # ligand decoupled in the complex
    from_alchemlyb(solvent_decoupling_est),   # ligand decoupled in solvent
    restraint_correction=-1.6,                # signed, from your Boresch/analytical term
)
# ΔG_bind = ΔG_solvent − ΔG_complex + restraint_correction
```

It returns a `FreeEnergyResult` (not a `DeltaDeltaG`), so absolute ΔG_bind
values rank directly in `FreeEnergyRanking`. The legs must be *decoupling*
free energies and the restraint term carries the sign your protocol uses —
see the function's docstring for the convention.

## FEP or MM/GBSA

- **MM/GBSA** ([recipe](ranking-binders.md)): cheap, post-processes
  trajectories you already have, good for triaging a large series.
- **FEP**: expensive, needs dedicated alchemical simulations, gives
  calibrated ~1 kcal/mol relative affinities for the shortlist.

A common workflow triages with MM/GBSA, then spends FEP on the top
handful — and because both land in `FreeEnergyResult` and
`FreeEnergyRanking`, the ranking code is identical.
