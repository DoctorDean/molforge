# Rank binders with MM/GBSA

You have a congeneric series — a lead compound and a handful of analogs —
and you want to know which ones bind your target tightest, to decide what
to make next. Endpoint free energy (**MM/GBSA**) is the cheap, standard
tool for this: run a short MD simulation of each complex, post-process it
into a binding ΔG, and rank.

The honest framing up front: MM/GBSA is poor at *absolute* affinities but
useful for *ranking* a series of similar ligands. molforge leans into
that — every result carries an uncertainty, and the ranking exposes
pairwise ΔΔG with propagated error so ties don't masquerade as wins.

## Requirements

```bash
pip install "molforge[md]"        # for the MD step (OpenMM/AMBER/GROMACS)
# plus the endpoint tool on PATH, matching your MD backend:
#   AmberTools (MMPBSA.py + ante-MMPBSA.py):  conda install -c conda-forge ambertools
#   or gmx_MMPBSA (for GROMACS trajectories):  conda install -c conda-forge gmx_mmpbsa
```

## The recipe

Given one MD trajectory per complex — the target bound to each analog —
score and rank them:

```python
from molforge.wrappers.freeenergy import AmberMMGBSA
from molforge.freeenergy import FreeEnergyRanking

engine = AmberMMGBSA()          # drives MMPBSA.py + ante-MMPBSA.py

# complexes: {name: Trajectory}, each an MD run of the target + that
# analog. See "Where the trajectories come from" below.
results = {
    name: engine.run(
        traj,
        receptor={"entity_type": "protein"},
        ligand={"entity_type": "ligand"},
        solvent_model="gb",         # MM/GBSA; "pb" for MM/PBSA
    )
    for name, traj in complexes.items()
}

ranking = FreeEnergyRanking(results)
```

`receptor` and `ligand` are molforge selections — the same field filters
you'd pass to `.select()` — resolved against the complex topology, not
chain IDs (which drift across files). Here the target is the protein and
the analog is the one non-polymer entity.

The engine is the only thing that changes between MD backends:
`GromacsMMGBSA` (driving `gmx_MMPBSA`) takes the same `run(traj, *,
receptor, ligand, solvent_model)` call and returns the same
`FreeEnergyResult`, so a GROMACS series ranks with identical downstream
code — only the constructor and the trajectory source differ.

## Reading the ranking

The ranking orders ligands tightest-first, but the number that matters
for a decision is the *difference* between two ligands and whether it
clears the noise:

```python
# Tightest first.
for name, r in ranking.ranked:
    print(f"{name:10s} ΔG = {r.delta_g:6.1f} ± {r.uncertainty:.1f} kcal/mol")

best_name, _ = ranking.best
print("tightest binder:", best_name)

# Is analog_B actually better than the current lead, or a wash?
ddg = ranking.delta_delta_g("lead", "analog_B")
print(f"ΔΔG(B – lead) = {ddg.value:+.1f} ± {ddg.uncertainty:.1f} kcal/mol")
print("tighter:", ddg.tighter)
```

`delta_delta_g(reference, other)` returns `other − reference` with the
two errors combined in quadrature (the runs are independent). Use it, not
the bare rank: if `abs(ddg.value)` is within roughly its `uncertainty`,
the two are tied and the rank order between them is noise. molforge
deliberately does **not** hand you a significance verdict — the right test
depends on assumptions (frame correlation, Gaussianity) it shouldn't bake
in — so you apply the threshold your project uses.

## What's in a result

Each `FreeEnergyResult` is more than a number:

```python
r = ranking.best[1]

r.delta_g          # binding ΔG, kcal/mol (lower = tighter)
r.uncertainty      # standard error across frames
r.method           # "MM/GBSA"

c = r.components    # the per-term breakdown
c.vdw, c.electrostatic, c.polar_solvation, c.nonpolar_solvation
c.enthalpy         # sum of the four — the interaction enthalpy
c.entropy          # None unless you ran an entropy calculation
```

When a rank looks wrong, the decomposition is how you diagnose it — a
ligand winning on electrostatics that a desolvation penalty should have
killed, say. Note `entropy` is `None`, not `0.0`: the single-trajectory
runs here drop the configurational entropy term. That's usually fine for
ranking *similar* ligands (their entropy differences roughly cancel) and
risky across dissimilar ones.

## Where the trajectories come from

`AmberMMGBSA` is a post-processor: it needs an Amber topology (`prmtop`)
and a trajectory on disk, and it will not build them. The path of least
resistance is the AMBER MD wrapper, whose `Trajectory` already carries
its run directory — `AmberMMGBSA` finds the `prmtop` and trajectory
there automatically:

```python
from molforge.wrappers.md import AMBER

md = AMBER(water_model="tip3p")
sim = md.prepare(complex_ready, force_field="ff14SB")   # complex must be parameterized
sim = md.minimize(sim)
traj = md.run(sim, n_steps=250_000, save_every=1_000)   # 500 ps, 250 frames

result = AmberMMGBSA().run(
    traj,
    receptor={"entity_type": "protein"},
    ligand={"entity_type": "ligand"},
)
```

Parameterizing the small-molecule analog (GAFF/antechamber charges) is
part of building `complex_ready` and is the fiddly step of any MM/GBSA
campaign — it's system prep, not something the free-energy engine does.

If your trajectory came from elsewhere, point the engine at the files
explicitly:

```python
result = AmberMMGBSA().run(
    traj,
    receptor={"entity_type": "protein"},
    ligand={"entity_type": "ligand"},
    prmtop="complex.prmtop",
    trajectory_file="prod.nc",
)
```

### From GROMACS

`GromacsMMGBSA` is the same story with GROMACS inputs: it needs a
structure (`.tpr`) and trajectory (`.xtc`), and finds them automatically
in a trajectory produced by the GROMACS MD wrapper (whose run directory
holds `md.tpr`, `md.xtc`, and `topol.top`):

```python
from molforge.wrappers.md import GROMACS
from molforge.wrappers.freeenergy import GromacsMMGBSA

md = GROMACS(water_model="tip3p")
sim = md.prepare(complex_ready, force_field="amber99sb-ildn")
sim = md.minimize(sim)
traj = md.run(sim, n_steps=250_000, save_every=1_000)

result = GromacsMMGBSA().run(
    traj,
    receptor={"entity_type": "protein"},
    ligand={"entity_type": "ligand"},
)
```

The selections are resolved to GROMACS index groups internally, so you
never write an `.ndx` by hand. As with Amber, point at files explicitly
(`structure=...`, `trajectory_file=...`, optional `topology=...`) when
the trajectory came from elsewhere. Ligand parameterization is again part
of building `complex_ready`.

## GB or PB

`solvent_model="gb"` (the default) runs MM/GBSA; `"pb"` runs MM/PBSA,
which solves the Poisson–Boltzmann equation for the polar solvation term
— slower, sometimes a better polar model. The result's `method` and
component terms reflect the choice; everything downstream (ranking, ΔΔG)
is identical.

## Caching

`run()` is cached on the run's provenance (masks, solvent model, frame
range, salt, and the input files). Re-scoring the same trajectory — a
second pass, a re-run of the notebook — returns instantly without
touching the tools. Change any parameter (GB → PB, a different frame
range) and it recomputes. See [Caching results](caching-results.md).

## What MM/GBSA is and isn't for

- **Good for:** ranking a congeneric series, triaging analogs, cheap
  affinity signal from trajectories you already have.
- **Not good for:** absolute Kd/ΔG. The implicit-solvent term is
  systematically biased and entropy is usually dropped, so a single ΔG in
  isolation means little.
- **Reach for FEP/TI instead** when you need rigorous relative affinities
  (~1 kcal/mol) between two ligands and can afford the alchemical cost.
  molforge's role there is setup and analysis, not running the campaign.

Treat the output as a ranking with error bars, compare with
`delta_delta_g`, and call within-error differences ties.
