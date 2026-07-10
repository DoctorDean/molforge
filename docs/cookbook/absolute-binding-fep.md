# Absolute binding free energy with FEP

Relative FEP tells you that analog B binds ~1 kcal/mol tighter than the
lead ([that recipe](rank-binders-fep.md)). Absolute binding FEP (**ABFE**)
asks the harder question — *how* tightly does this ligand bind, on an
absolute scale — with no congeneric reference at all. That makes it the
tool when your candidates aren't a series (fragment hits, scaffold hops,
a ligand with no measured analog) or when you want a number to compare
against an experimental K_d.

molforge doesn't run the alchemical simulations; it **ingests the
analysis** and closes the thermodynamic cycle. You run ABFE and analyze
each leg with [alchemlyb](https://alchemlyb.readthedocs.io); molforge
combines the legs and the restraint correction into a `FreeEnergyResult`.

## The double-decoupling cycle

ABFE by double decoupling annihilates the ligand's interactions with its
environment in two places and adds a restraint correction:

```
ΔG_bind = ΔG_solvent − ΔG_complex + restraint_correction
```

Three ingredients:

- **ΔG_complex** — the free energy of decoupling the ligand from the
  *complex*, with restraints holding it in the pocket.
- **ΔG_solvent** — the free energy of decoupling the same ligand in
  *solvent*.
- **restraint_correction** — the standard-state term for those restraints
  (below). Without it, ΔG_bind isn't referenced to 1 M.

Both legs are *decoupling* free energies (coupled → non-interacting). A
strong binder is hard to pull out of the complex (large positive
`ΔG_complex`), so `ΔG_solvent − ΔG_complex` comes out negative —
favorable — as it should.

## Requirements

```bash
pip install "molforge"        # the ingest adds no deps of its own
pip install alchemlyb pymbar  # you bring these to analyze each leg
```

## Ingest the two legs

Analyze each decoupling leg with alchemlyb (an MBAR/BAR/TI estimator per
leg), then ingest:

```python
from molforge.wrappers.freeenergy import absolute_binding_free_energy, from_alchemlyb

complex_leg = from_alchemlyb(complex_decoupling_estimator)  # restrained, in the complex
solvent_leg = from_alchemlyb(solvent_decoupling_estimator)  # in solvent
```

Each is a `FreeEnergyResult` in kcal/mol. If your λ schedule runs the
other way (coupling rather than decoupling), reverse the λ order before
`from_alchemlyb`, or negate the leg.

## The restraint correction

To decouple the ligand in the complex you first tether it to the
receptor, or it drifts out of the pocket and the calculation is
meaningless. The standard choice is a set of **Boresch orientational
restraints** — one distance, two angles, and three dihedrals between
three receptor atoms and three ligand atoms — which pin the ligand's
position *and* orientation.

Those restraints have to be accounted for. The free energy of releasing
them to the standard-state volume (V° ≈ 1660 Å³ at 1 M) has a closed-form
analytical expression under the rigid-rotor/harmonic-oscillator
approximation, so your ABFE tooling reports it directly; you don't
simulate it. Pass it as `restraint_correction`:

```python
dg_bind = absolute_binding_free_energy(
    complex_leg,
    solvent_leg,
    restraint_correction=-6.4,   # signed, from your Boresch analytical term
)
print(f"ΔG_bind = {dg_bind.delta_g:+.2f} ± {dg_bind.uncertainty:.2f} kcal/mol")
```

`restraint_correction` is added *as-is* — molforge takes it with whatever
sign your protocol uses, because conventions genuinely differ (some tools
report the free energy of *releasing* the restraint, others the cost of
*imposing* it, and some already fold the restrain-on step into the
complex leg). It can be a plain float, or a `FreeEnergyResult` if the term
carries its own uncertainty (which then propagates in quadrature with the
two legs).

**Get the sign right by sanity-checking, not by trusting a convention.**
A real binder must come out negative. And leaving orientational restraints
out entirely inflates affinity by up to ~4 kcal/mol — so a suspiciously
favorable ΔG_bind usually means a restraint/standard-state term with the
wrong sign or magnitude, not a wonder-drug.

## Reading and ranking

The result is an *absolute* ΔG_bind — a `FreeEnergyResult`, not a
`DeltaDeltaG` — with the three input terms in its metadata:

```python
print(dg_bind.metadata)
# {'complex_leg': ..., 'solvent_leg': ..., 'restraint_correction': -6.4}
```

Because it's absolute, several ligands rank directly, no shared reference
needed — and on the same scale as an MM/GBSA triage:

```python
from molforge.freeenergy import FreeEnergyRanking

# legs_by_ligand: {name: (complex_estimator, solvent_estimator, restraint)}
results = {
    name: absolute_binding_free_energy(
        from_alchemlyb(complex_est),
        from_alchemlyb(solvent_est),
        restraint_correction=rc,
    )
    for name, (complex_est, solvent_est, rc) in legs_by_ligand.items()
}
ranking = FreeEnergyRanking(results)
for name, r in ranking.ranked:
    print(f"{name:10s} ΔG_bind = {r.delta_g:+.2f} ± {r.uncertainty:.2f} kcal/mol")
```

Converting to an affinity for a gut check: ΔG_bind ≈ RT ln K_d, so at 298 K
a K_d of 1 µM is about −8.2 kcal/mol and each ~1.36 kcal/mol is a decade
in K_d.

## What to watch

- **Accuracy.** Well-converged ABFE lands around 1–2 kcal/mol of
  experiment (better in favorable cases); treat a single ΔG_bind as an
  estimate with real error bars, not a precise K_d.
- **Convergence and orientation.** In the fully decoupled state the
  ligand can reorient freely, and sampling that well is the usual failure
  mode — check that each leg's estimate is stable before trusting the
  cycle.
- **The restraint is load-bearing.** The choice of anchor atoms and force
  constants, and the matching standard-state correction, are where ABFE
  most often goes wrong; the sign check above is your first line of
  defense.

## See also

- [Rank binders with FEP](rank-binders-fep.md) — relative FEP (ΔΔG) for a
  congeneric series, the cheaper and more common case.
- [Rank binders with MM/GBSA](ranking-binders.md) — an endpoint method to
  triage a large set before spending FEP on the shortlist.
