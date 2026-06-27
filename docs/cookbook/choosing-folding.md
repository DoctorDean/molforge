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
