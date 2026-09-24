# Choosing a folding engine

molforge wraps five folding engines. They look the same from the
outside — `engine.predict(sequence)` returns a `Protein` —  but
underneath they trade off accuracy, speed, dependencies, and the
kinds of input they handle. This page is a decision-oriented guide
to picking one.

If you don't want to think about it: **start with ESMFold**. It's
the lowest-friction option that produces good results for most
single-chain targets.

## Side-by-side

| Engine        | Method                           | Multimer? | MSA needed?      | Typical speed (300 aa)         | When to pick it                                                                 |
| ------------- | -------------------------------- | --------- | ---------------- | ------------------------------ | -------------------------------------------------------------------------------- |
| **ESMFold**   | Language-model based, no MSA     | No        | No               | Seconds on GPU, minutes on CPU | Monomer prediction at scale, when you don't have time / infra for MSAs.        |
| **AlphaFold** | MSA-based, AF2 / AF2-multimer    | Yes       | Yes (or ColabFold) | Minutes per call, plus MSA   | Maximum monomer accuracy; multimer prediction with known interfaces.            |
| **Boltz**     | AF3-style, fully end-to-end (CLI subprocess) | Yes | Optional (server) | Minutes per call         | Multimer with ligands / cofactors; AF3-class accuracy.                          |
| **Chai-1**    | AF3-style, fully end-to-end (Python API) | Yes | Optional (server) | Minutes per call             | AF3-class accuracy from an independent re-implementation; natural cross-check for Boltz. |
| **RoseTTAFold** | RFAA (RoseTTAFold All-Atom)    | Yes       | Yes              | Minutes per call               | Atomistic prediction including nucleic acids, modified residues, cofactors.    |

The "Typical speed" numbers are order-of-magnitude on a modern GPU
(A100-class). Don't take them too literally — actual run times
depend heavily on sequence length, GPU, and (for AF / Boltz / Chai)
MSA generation overhead.

## How to choose

### Predicting a single monomer

**Use ESMFold** unless you have a specific reason not to. It's a
single forward pass through a language model — no MSA, no template
search, no multi-model ensembling. The accuracy is competitive with
AlphaFold for most well-folded soluble proteins; the speed
difference is huge.

When ESMFold won't be enough:

- **Long sequences (> ~600 residues).** ESMFold's memory grows
  quadratically with length; very long sequences need attention
  chunking (`chunk_size=64`) or just don't fit.
- **Disordered regions.** Both ESMFold and AlphaFold mark IDRs
  with low pLDDT, but AlphaFold's IDR predictions are slightly
  more reliable on average.
- **Targets far from the training distribution.** Hyperthermophile
  proteins, designed proteins, very novel folds — AlphaFold's MSA
  signal helps; ESMFold's language model can fall back to
  hallucinating.

### Predicting a multimer (complex)

**Use AlphaFold or Boltz.** ESMFold has no multimer support;
RoseTTAFold can do multimers but is more cumbersome to set up.

- **AlphaFold-multimer (AF2-multimer)** is the workhorse — well-
  validated, lots of community guidance on how to interpret
  results. Needs MSAs for each chain.
- **Boltz** is AF3-style. Co-folds protein + protein + ligand +
  nucleic acid in one shot. Newer and the dependency footprint is
  simpler (no ColabFold), but the community's published
  experience is thinner.

### Predicting with cofactors, ligands, or nucleic acids

**Use Boltz, Chai-1, or RoseTTAFold-All-Atom.** All three handle
non-protein components natively.

- **Boltz**: subprocess-based CLI wrapper. Driven by a YAML
  spec; ligands as SMILES. Solid first choice for most workflows.
- **Chai-1**: Python-API wrapper (no subprocess). Driven by typed
  FASTA (`>protein|name=...`, `>ligand|name=...`). Independent
  re-implementation of AlphaFold-3 from a different team —
  natural cross-check for Boltz on hard cases.
- **RoseTTAFold-All-Atom (RFAA)**: the most chemically explicit —
  handles modified residues, covalent ligands, and unusual
  chemistry that Boltz/Chai might mis-handle. Harder to set up
  (requires a local install of the RFAA repository).

### Cross-checking with Chai-1 and Boltz

Boltz and Chai-1 are both open-weights AlphaFold-3 re-implementations,
released within weeks of each other (October–November 2024) by
independent teams (MIT Jameel Clinic and Chai Discovery). Running
both on a hard target and comparing the two top predictions is a
robust confidence signal: when two independent AF3-class models
agree on a binding pose or interface geometry, it's much stronger
evidence than either alone.

```python
from molforge.wrappers.folding import Boltz, Chai1
from molforge.structure import rmsd

boltz_pred = Boltz(use_msa_server=True).predict(sequence)
chai_pred  = Chai1(use_msa_server=True).predict(sequence)

# Align the two predictions and compute backbone RMSD.
backbone_rmsd = rmsd(boltz_pred, chai_pred, selection="backbone")
print(f"Cross-engine RMSD: {backbone_rmsd:.2f} Å")
print(f"Boltz pTM:  {boltz_pred.metadata['ptm']:.2f}")
print(f"Chai-1 pTM: {chai_pred.metadata['ptm']:.2f}")
# Two engines agreeing on backbone (low RMSD) plus both reporting
# high pTM is the strongest single-call confidence signal molforge
# can produce.
```

### Predicting from a sequence-database search (with MSAs)

**Use AlphaFold via ColabFold.** ColabFold provides MMseqs2-based
fast MSA search, plus the AF2 forward pass — typically minutes per
prediction including MSA time, against many hours for traditional
HHblits + JackHMMER pipelines.

## Common dimensions

### Confidence metrics

Every engine reports per-residue confidence; molforge surfaces this
in a uniform shape:

| Engine        | Confidence metric          | molforge access                            |
| ------------- | -------------------------- | ------------------------------------------ |
| ESMFold       | pLDDT (0–100 per residue)  | `metadata["confidence_per_residue"]`       |
| AlphaFold     | pLDDT + PAE (matrix)       | `metadata["confidence_per_residue"]`, `metadata["pae"]` |
| Boltz         | pLDDT + pTM + iPTM         | `metadata["confidence_per_residue"]`, `metadata["ptm"]`, `metadata["iptm"]` |
| Chai-1        | pLDDT + pTM + iPTM + aggregate_score | `metadata["confidence_per_residue"]`, `metadata["ptm"]`, `metadata["iptm"]`, `metadata["aggregate_score"]` |
| RoseTTAFold   | pLDDT + PAE                | `metadata["confidence_per_residue"]`, `metadata["pae"]` |

`metadata["mean_confidence"]` is always the per-residue mean — a
single scalar you can sort by.

For multimer predictions, **iPTM** (Boltz) and **ipTM** (AF-
multimer) are the interface-quality metrics; the headline pLDDT can
be high even with badly-modelled interfaces.

### Every sample, not just the best (Chai-1)

Chai-1 always runs five diffusion samples per call. `predict()` returns
the highest-scoring one, which is usually what you want — but the other
four are fully computed structures, and the GPU time is already spent.
`predict_samples()` returns all five instead:

```python
from molforge.wrappers.folding import Chai1

samples = Chai1().predict_samples(sequence)   # 5 structures, one inference
best = samples[0]                             # == Chai1().predict(sequence)

spread = [s.metadata["aggregate_score"] for s in samples]
```

They come back ranked, so `samples[0]` is exactly what `predict()` would
have returned; each carries its own scores plus `sample_index` (Chai's
own 0–4 index) and `sample_rank` (its place in the ranking).
`predict_complex_samples(spec)` is the complex counterpart — five ligand
poses per call rather than one, which is the difference between a
3-week and a 15-week campaign when the whole thing is GPU-bound.

The two shapes share a cache, so asking for one after the other doesn't
re-run the model:

```python
engine = Chai1()
engine.predict(sequence)            # runs the model
engine.predict_samples(sequence)    # cache hit — the other four were banked
```

### Reproducible sampling (seeds)

The AF3-class engines sample structures with a diffusion model, so two
runs of the same input can differ. Both take a `seed`, on the
constructor or per call:

| Engine        | Seed control                                                                 |
| ------------- | ---------------------------------------------------------------------------- |
| Boltz         | `Boltz(seed=42)`, or per call `predict(seq, seed=42)`; passed to the CLI as `--seed`. |
| Chai-1        | `Chai1(seed=42)`; seeds torch and Chai's own sampler.                        |
| ESMFold       | Deterministic single forward pass — no sampling to seed.                     |
| AlphaFold     | Seeded by the ColabFold backend, not exposed by the wrapper yet.              |
| RoseTTAFold   | Not exposed by the wrapper yet.                                              |

```python
from molforge.wrappers.folding import Boltz

engine = Boltz(model_version="boltz2", seed=42)
first  = engine.predict(sequence)      # seeded, and cached under that seed
again  = engine.predict(sequence)      # cache hit — same structure

# A seeded ensemble: one engine, several seeds, distinct cache entries.
ensemble = [engine.predict(sequence, seed=s) for s in range(5)]
```

The seed is recorded in the structure's
[provenance](inspect-provenance.md), so it is part of the cache key —
different seeds don't collide, and a
[replayed](../guide/reproducibility.md) manifest re-runs with the seed it
was folded with.

Two caveats. Every engine rejects per-call keywords it doesn't consume,
so a typo can't quietly cost you a seeded run — `seed` is a per-call
option on Boltz alone, and the other four wrappers take no per-call
options at all, raising `TypeError` and pointing at the constructor
instead. (Which also means `cross_engine_fold(..., seed=42)` fails on
every engine but Boltz: set engine-specific options on that engine's
constructor.) And a seed pins the *sampler*, not the hardware: GPU
non-determinism means small numerical drift across machines is still
expected, so treat seeding as reproducibility of intent, not of bits.

### MSA depth

An AF3-class model leans heavily on coevolution: the deeper the
alignment, the more confidently it reproduces the fold the family
already shows. Shrinking the alignment is how you find out what the
model believes without that support — and, in practice, how you push it
off a dominant conformation towards alternatives it would otherwise
never sample.

Both AF3-class engines take `msa_depth`, a cap on how many MSA
sequences the model may use:

```python
from molforge.wrappers.folding import Boltz

ladder = {
    depth: Boltz(msa_depth=depth, use_msa_server=True).predict(sequence)
    for depth in (8, 16, 32, 64, None)   # None = the full alignment
}
```

Each rung is a separate cache entry, so the sweep re-runs only what it
must, and a repeat of the whole ladder is free.

| Engine  | `msa_depth=N` becomes                     | Where the cap bites            |
| ------- | ----------------------------------------- | ------------------------------ |
| Boltz   | `--subsample_msa --num_subsampled_msa N`   | Every trunk pass               |
| Chai-1  | `recycle_msa_subsample=N`                  | Recycling passes only; the first trunk pass still sees the full alignment |

That difference is worth knowing before you compare the two: both
genuinely shrink what the model sees, so a ladder is meaningful on
either engine, but the same `N` is not the same experiment on both.
Report which engine a ladder ran on.

`msa_depth` is a *shallow* alignment, not the absence of one. To run
with no MSA at all, use `Boltz(use_msa_server=False)` — or Chai-1's
default, which is already MSA-free. `msa_depth=0` raises and says so
rather than quietly meaning one of the two.

### Templates

A template lets the model copy a known fold instead of deriving one.
Usually helpful; occasionally the whole problem, because a model handed
a structure of the thing you're asking about will hand it back. Taking
templates away is how you check the prediction is a prediction.

`TemplatePolicy` names the mechanism, because the engines genuinely
differ:

```python
from molforge.folding import TemplatePolicy
from molforge.wrappers.folding import Boltz, Chai1

# Boltz reads template structures from its input YAML.
Boltz(templates=TemplatePolicy.from_structures("4hhb.cif", "1ubq.cif"))

# Chai-1 takes a precomputed hits table, or searches for itself.
Chai1(templates=TemplatePolicy.from_hits("hits.m8"))
Chai1(templates=TemplatePolicy.from_server())

# Both understand "no templates" — the control arm of the ablation.
Boltz(templates="none")
Chai1(templates=TemplatePolicy.none())
```

| Mode                          | Boltz | Chai-1 |
| ----------------------------- | :---: | :----: |
| `none()`                      |   ✅   |   ✅    |
| `from_structures(*paths)`     |   ✅   |   —    |
| `from_hits(path)`             |   —   |   ✅    |
| `from_server()`               |   —   |   ✅    |

An engine given a mode it cannot honour raises, naming what it does
support. That is deliberate: the alternative is folding *without* the
templates you asked for and returning a structure that looks fine, with
nothing but the provenance to suggest the ablation never happened.

molforge does not decide which templates are appropriate. Restricting a
run to, say, only unbound structures is a judgement about your targets;
you make it by choosing what to pass. What molforge guarantees is that
your choice is applied, recorded in provenance, and part of the cache
key — so an ablation and its control never share a cached result.

### Installation footprint

| Engine        | Install                                                                                  |
| ------------- | ---------------------------------------------------------------------------------------- |
| ESMFold       | `pip install "molforge[ml]"` pulls torch + transformers + esm. Weights download on first use (~3 GB). |
| AlphaFold     | Use ColabFold backend: `pip install colabfold`. Or local AF2 install (much heavier).      |
| Boltz         | `pip install boltz`. Weights download on first use.                                       |
| Chai-1        | `pip install chai_lab`. Weights (~3 GB) download on first use. Linux only; CUDA + bfloat16 GPU required. |
| RoseTTAFold   | Manual clone + install of dauparas/RoseTTAFold-All-Atom. `RFAA_HOME` env var.            |

### Licenses

| Engine        | License                                                                          |
| ------------- | -------------------------------------------------------------------------------- |
| ESMFold       | MIT (model weights and code).                                                    |
| AlphaFold     | Apache 2.0 for the code; weights have a non-commercial use clause (verify yourself). |
| Boltz         | MIT.                                                                             |
| Chai-1        | Apache 2.0 for the code; weights ship under Chai's own terms (verify yourself for commercial use). |
| RoseTTAFold   | BSD.                                                                             |

ESMFold, Boltz, and RoseTTAFold are unambiguously commercial-use-OK.
AlphaFold's and Chai-1's weight licenses have terms worth reading
if you're in a commercial context.

## Cross-engine workflows

molforge's uniform interface lets you swap engines without touching
downstream code:

```python
from molforge.wrappers.folding import ESMFold, AlphaFold, Boltz, Chai1

for engine in [ESMFold(), AlphaFold(), Boltz(), Chai1()]:
    protein = engine.predict(sequence)
    # Same Protein interface, same metadata keys — same downstream code.
```

This is the basis for *cross-engine validation* — predict with
multiple engines, look for consensus. See the
[cross-engine validation example](../examples/cross_engine_validation.ipynb).
The Boltz / Chai-1 pair is particularly powerful: same architectural
family (AF3 re-implementation), independent codebases, so agreement
is a meaningful signal.

## What molforge doesn't wrap (yet)

- **ESM-IF1** — inverse folding (sequence design from structure).
  Lives under generative engines; see
  [Choosing a generative engine](choosing-generative.md).
- **AlphaFold 3 (DeepMind release)** — not yet wrapped. The Boltz
  and Chai-1 reimplementations cover most of AF3's accuracy
  ground; the official DeepMind release adds via the
  [plugin system](../guide/plugins.md) if you need it sooner than
  the roadmap delivers.
- **Protenix** — another AF3 reimplementation; on the roadmap.
