# Fold a sequence

You have a protein sequence (or several) and want predicted 3D
structures. This recipe uses ESMFold — single-sequence, no MSA,
fast and good for monomers up to ~600 residues.

For sequences longer than ~600 residues, multimers, or maximum
accuracy: see [Choosing a folding engine](choosing-folding.md) for
when to pick AlphaFold, Boltz, or RoseTTAFold instead. The molforge
API is the same shape; only the constructor and parameters change.

## Requirements

```bash
pip install "molforge[ml]"          # torch, transformers, esm
```

The ESMFold weights (~3 GB) download on first use. A modern GPU
helps but is not required — CPU works, slowly.

## The recipe

```python
from molforge.wrappers.folding import ESMFold

sequence = (
    "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGSEFKLFEISTDQDFEVADVPYRSKD"
    "WAKWGYGEHIVEVRGSDISGEFKKAYNALDGKVEDFRTRPSWKEDLEFFKEAKAGI"
)

engine = ESMFold()                  # download weights on first use
protein = engine.predict(sequence)

print(f"Folded {protein.n_chains} chain, {protein.n_residues} residues, "
      f"mean pLDDT = {protein.metadata['mean_confidence']:.1f}")

# Save to disk for downstream tools.
from molforge.io import save
save(protein, "folded.pdb")
```

## What you get back

A [`Protein`](../reference/core.md) with:

- The 3D coordinates in `protein.atom_array.coords` (shape
  `(n_atoms, 3)`, float32, units of Å).
- pLDDT values in three forms under `protein.metadata`:
  `confidence_per_atom`, `confidence_per_residue`, `mean_confidence`.
  All three are the same number repeated at different granularities;
  ESMFold reports a single pLDDT per residue, so per-atom values
  within a residue are uniform.
- A `Provenance` record at `protein.metadata["provenance"]` capturing
  the engine, model name, device, and the sequence you started from.
  See [Inspect provenance](inspect-provenance.md).

## Filtering by confidence

pLDDT is the headline quality metric. A typical workflow filters out
low-confidence regions before doing anything else with the structure:

```python
import numpy as np

per_residue = protein.metadata["confidence_per_residue"]
high_confidence_mask = per_residue > 70    # AlphaFold's "confident" threshold
print(f"{high_confidence_mask.sum()}/{len(per_residue)} residues above pLDDT 70")
```

For a more graceful "trim ragged ends" approach, see the
[Structures walkthrough](../walkthroughs/02_structures.ipynb) which
covers slicing a `Protein` by residue range.

## Folding many sequences

ESMFold reuses the loaded model across calls — construct one engine
and call `predict` repeatedly rather than re-constructing:

```python
from molforge.io import read_fasta

engine = ESMFold()
for record in read_fasta("targets.fasta"):
    protein = engine.predict(record.sequence)
    save(protein, f"folded/{record.name}.pdb")
```

If you're folding hundreds of sequences, consider:
- Setting `chunk_size` in the `ESMFold(...)` constructor for very
  long sequences (it trades speed for memory).
- Running on GPU: pass `device="cuda"`.

## When to pick a different engine

ESMFold is the right default for monomer prediction at scale. Switch
when:

- You need **multimer prediction** → AlphaFold (with `colabfold`
  backend) or Boltz.
- You need **maximum accuracy** on hard targets → AlphaFold or
  RoseTTAFold All-Atom, with MSAs.
- You're predicting a **protein + small molecule complex** → Boltz
  (native multimer + ligand support).

See [Choosing a folding engine](choosing-folding.md) for the full
decision matrix.
