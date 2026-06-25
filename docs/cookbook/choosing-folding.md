# Choosing a folding engine

molforge wraps four folding engines. They look the same from the
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
| **Boltz**     | AF3-style, fully end-to-end      | Yes       | Optional (server) | Minutes per call               | Multimer with ligands / cofactors; AF3-class accuracy without ColabFold setup.  |
| **RoseTTAFold** | RFAA (RoseTTAFold All-Atom)    | Yes       | Yes              | Minutes per call               | Atomistic prediction including nucleic acids, modified residues, cofactors.    |

The "Typical speed" numbers are order-of-magnitude on a modern GPU
(A100-class). Don't take them too literally — actual run times
depend heavily on sequence length, GPU, and (for AF / Boltz) MSA
generation overhead.

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

**Use Boltz or RoseTTAFold-All-Atom.** Both handle non-protein
components natively.

- **Boltz**: easier to drive (single sequence input, ligands as
  SMILES). Good first choice.
- **RoseTTAFold-All-Atom (RFAA)**: the most chemically explicit —
  handles modified residues, covalent ligands, and unusual
  chemistry that Boltz might mis-handle. Harder to set up
  (requires a local install of the RFAA repository).

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
| RoseTTAFold   | Manual clone + install of dauparas/RoseTTAFold-All-Atom. `RFAA_HOME` env var.            |

### Licenses

| Engine        | License                                                                          |
| ------------- | -------------------------------------------------------------------------------- |
| ESMFold       | MIT (model weights and code).                                                    |
| AlphaFold     | Apache 2.0 for the code; weights have a non-commercial use clause (verify yourself). |
| Boltz         | MIT.                                                                             |
| RoseTTAFold   | BSD.                                                                             |

ESMFold and Boltz are unambiguously commercial-use-OK. AlphaFold's
weight license has been a moving target; if you're in commercial
context, read the license terms at the time of use.

## Cross-engine workflows

molforge's uniform interface lets you swap engines without touching
downstream code:

```python
from molforge.wrappers.folding import ESMFold, AlphaFold, Boltz

for engine in [ESMFold(), AlphaFold(), Boltz()]:
    protein = engine.predict(sequence)
    # Same Protein interface, same metadata keys — same downstream code.
```

This is the basis for *cross-engine validation* — predict with
multiple engines, look for consensus. See the
[cross-engine validation example](../examples/cross_engine_validation.ipynb).

## What molforge doesn't wrap (yet)

- **ESM-IF1** — inverse folding (sequence design from structure).
  Lives under generative engines; see
  [Choosing a generative engine](choosing-generative.md).
- **Chai-1, AlphaFold 3 (DeepMind release)** — not yet wrapped.
  Add via the [plugin system](../guide/plugins.md) if you need
  them sooner than the roadmap delivers.
