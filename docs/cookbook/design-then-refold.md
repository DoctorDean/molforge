# Design then refold

You have a protein structure — natural, designed, or hallucinated —
and want to redesign its sequence while keeping the backbone fixed.
This is **inverse folding**, also called fixed-backbone design.
Common use cases: stabilising a fragile fold, removing surface
hydrophobic patches, designing a new sequence for an
RFdiffusion-generated backbone.

This recipe runs ProteinMPNN on a backbone, then refolds each
designed sequence with ESMFold to check the design actually folds
back into the intended shape. RMSD between the original backbone
and the refolded prediction is the key go/no-go metric.

## Requirements

```bash
pip install "molforge[ml]"               # torch, transformers, esm
# Plus ProteinMPNN itself — clone from GitHub:
git clone https://github.com/dauparas/ProteinMPNN
export PROTEINMPNN_HOME=/path/to/ProteinMPNN
```

## The recipe

```python
from molforge.io import fetch, save
from molforge.wrappers.folding import ESMFold
from molforge.wrappers.generative import ProteinMPNN
from molforge.structure import rmsd

# 1. Take a starting structure. Could be from a PDB, AlphaFold,
#    RFdiffusion — anything with a backbone.
backbone = fetch("1UBQ")                  # ubiquitin, classic small fold

# 2. Design 8 sequences for this backbone.
mpnn = ProteinMPNN(num_seqs=8, sampling_temp=0.1, seed=42)
designs = mpnn.generate(backbone)
print(f"Generated {len(designs)} designs, "
      f"score range {designs[0].score:.2f} – {designs[-1].score:.2f}")

# 3. Refold each design and compare to the original backbone.
folder = ESMFold()
for i, design in enumerate(designs):
    refolded = folder.predict(design.sequence)
    bb_rmsd = rmsd(refolded, backbone, subset="ca", align=True)
    plddt   = refolded.metadata["mean_confidence"]
    print(f"Design {i}: score={design.score:.2f}  "
          f"refold_rmsd={bb_rmsd:.2f} Å  pLDDT={plddt:.1f}")
    save(refolded, f"refolded_{i}.pdb")
```

## Reading the output

For a successful redesign you want **all three** signals together:

| Signal                       | Interpretation                                            |
| ---------------------------- | --------------------------------------------------------- |
| ProteinMPNN `score` low      | The model thinks the sequence is plausible for the fold.  |
| Refolded RMSD low (< ~2 Å)   | The new sequence actually folds back to the target.       |
| Refolded pLDDT high (> 70)   | The refold is *confident*, not just close-by-coincidence. |

A design with great MPNN score but high refold-RMSD is the classic
failure mode — MPNN thought the sequence was native-like, but ESMFold
disagrees. Trust the refold.

You'll typically see all 8 designs pass the RMSD test for a small,
well-folded target like ubiquitin. For larger or harder targets,
generate more designs and filter.

## Holding part of the sequence fixed

A common variant: redesign everything *except* an active site. Pass
the active-site positions as fixed:

```python
designs = mpnn.generate(
    backbone,
    fixed_positions={"A": [17, 35, 86]},   # 1-indexed residue positions
)
```

This is also how you redesign a single chain of a complex while
keeping the partner chain untouched:

```python
designs = mpnn.generate(
    backbone,
    chains_to_design="A",                  # only redesign chain A
    # Chain B (if present) becomes fixed context for the design.
)
```

## Picking a sampling temperature

`sampling_temp` controls how diverse the designs are:

- `0.1` (default): conservative; designs cluster around the
  ProteinMPNN-preferred residue identity at each position. Good
  for stability-oriented redesign.
- `0.3 – 0.5`: more diverse; useful when you want to explore the
  sequence space (e.g. de novo design from an RFdiffusion
  backbone).
- `1.0+`: essentially random sampling. Almost always too noisy.

A typical workflow generates a *batch* at a higher temperature, then
filters by ESMFold refold quality — see the
[End-to-end design example](../examples/end_to_end_design.ipynb).

## Pairing with RFdiffusion

For *de novo* design — generating a brand-new backbone, not just
redesigning an existing one — pair ProteinMPNN with RFdiffusion:

```python
from molforge.wrappers.generative import RFdiffusion

# 1. RFdiffusion generates novel backbones.
backbones = RFdiffusion().generate(length=80, num_designs=10)

# 2. For each backbone, design sequences with ProteinMPNN.
mpnn = ProteinMPNN(num_seqs=8)
for backbone in backbones:
    sequences = mpnn.generate(backbone)
    # 3. Pick the best by some criterion (score, refold quality, etc.)
    ...
```

See [Choosing a generative engine](choosing-generative.md) for when
each is the right tool.

## Cross-checking with ESM-IF1

molforge also wraps ESM-IF1, an inverse-folding model with a
different architecture (GVP-GNN + transformer) and different
training data (~12M AlphaFold2 predictions vs ProteinMPNN's ~20k
PDB structures). When both engines agree on a residue identity at
a position, the agreement is a strong signal — orthogonal training
data means the agreements aren't trivial.

A common workflow: design with ProteinMPNN (the field's default,
faster install), validate with ESM-IF1 on the top candidates,
filter for designs both engines like.

```python
from molforge.wrappers.generative import ProteinMPNN, ESMIF1
from molforge.wrappers.folding import ESMFold

mpnn  = ProteinMPNN(num_seqs=16, sampling_temp=0.1, seed=42)
esmif = ESMIF1(num_seqs=8, temperature=0.1, seed=42)
folder = ESMFold()

mpnn_designs = mpnn.generate(backbone)
for design in mpnn_designs[:4]:                 # top 4 from MPNN
    # Re-score via ESM-IF1 on the same backbone, count agreement.
    esmif_designs = esmif.generate(backbone)
    esmif_top = esmif_designs[0].sequence
    agreement = sum(a == b for a, b in zip(design.sequence, esmif_top)) / len(design.sequence)

    refolded = folder.predict(design.sequence)
    plddt = refolded.metadata["mean_confidence"]
    print(f"MPNN score {design.score:.2f}  ESM-IF1 agreement {agreement:.0%}  pLDDT {plddt:.1f}")
```

Designs where MPNN and ESM-IF1 strongly agree (>70% residue
identity at default temperatures) and refold cleanly are the
candidates worth carrying forward.

## Provenance

Each `DesignedSequence` carries its own `Provenance` — whether it
came from ProteinMPNN or ESM-IF1. When you refold one with ESMFold,
the resulting `Protein`'s provenance has the ESMFold call as a
top-level step but **does not** chain back to the original design
— folding takes a sequence string, not a `DesignedSequence` object,
so the chain is broken at that hand-off.

If you need a record of the design that produced a given refold,
keep a sidecar dict mapping refold filenames to
`design.metadata["provenance"].to_json()`. A future molforge
"workflow" abstraction may surface this more directly; for now, save
the provenance yourself.
