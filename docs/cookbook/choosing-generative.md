# Choosing a generative engine

molforge wraps three generative engines spanning two related
problems: backbone generation (RFdiffusion) and sequence design
from a backbone (ProteinMPNN, ESM-IF1). The sequence-design pair
solve the same problem with different architectures.

## Side-by-side

| Engine          | Designs                | Outputs                          | When to pick it                                                          |
| --------------- | ---------------------- | -------------------------------- | ------------------------------------------------------------------------ |
| **RFdiffusion** | Backbones (no sequence) | `list[Protein]`, backbone atoms only | Generating novel 3D scaffolds; sampling new folds.                       |
| **ProteinMPNN** | Sequences for a backbone | `list[DesignedSequence]`        | Inverse folding; designing sequences that adopt a given fold; the field's default. |
| **ESM-IF1**     | Sequences for a backbone | `list[DesignedSequence]`        | Inverse folding; orthogonal architecture for cross-checking ProteinMPNN; designs scored by a language-model-style log-likelihood. |

**Backbone + sequence-design together** is the standard *de novo*
design loop: RFdiffusion proposes a novel backbone, ProteinMPNN
or ESM-IF1 designs sequences that adopt that backbone, ESMFold
validates the design by refolding the sequence and checking it
lands on the target backbone. See
[Design then refold](design-then-refold.md).

## How to choose

### You want to design a sequence for an existing structure

**Use ProteinMPNN.** This is the textbook inverse-folding problem —
given a backbone (natural, designed, or hallucinated), generate a
sequence that adopts it.

Common cases:

- **Stabilising a natural protein.** Run ProteinMPNN on the wild-
  type structure; pick designs with low MPNN score and high
  refold confidence. Many will be more stable than wild-type.
- **Designing surface residues only.** Pass `fixed_positions` to
  hold the core sequence fixed; redesign just the surface for
  solubility, expression, or epitope work.
- **Cross-species transplant.** ProteinMPNN doesn't care about the
  source species — give it any backbone, it returns sequences
  drawn from the model's learned distribution.

### You're choosing between ProteinMPNN and ESM-IF1

Both solve inverse folding (backbone → sequence). They differ in
architecture, training data, and ergonomics:

|                     | ProteinMPNN                                | ESM-IF1                                                |
| ------------------- | ------------------------------------------ | ------------------------------------------------------ |
| Architecture        | Message-passing neural network (k-NN graph) | GVP-GNN + seq2seq transformer                          |
| Training data       | ~20k PDB structures (Dauparas et al. 2022) | ~12M AlphaFold2 predictions (Hsu et al. 2022)          |
| Sequence recovery   | ~52% (paper); ~50–60% in practice          | ~51% on structurally held-out backbones                |
| Install footprint   | Clone of dauparas/ProteinMPNN + `PROTEINMPNN_HOME` env var | `pip install fair-esm` + torch-geometric              |
| Multi-chain         | Native support via `chains_to_design` /     | Single-chain in v1 (multichain via library helper)     |
|                     | `fixed_positions` arguments                |                                                        |
| Sampling control    | `omit_aas` to forbid specific residues     | Plain temperature; no per-residue veto                 |

**When ProteinMPNN is the right default**: most cases. It's the
field's de-facto standard, the install is light (just a git clone),
and the API supports the most common knobs (multi-chain, fixed
positions, biased sampling).

**When to reach for ESM-IF1**: cross-checking critical designs.
ESM-IF1 is trained on a wildly different distribution (12M
AlphaFold predictions vs 20k PDB structures), so when both engines
agree on a residue identity at a position, the agreement is a
strong signal. A common pattern: design with ProteinMPNN, then run
ESM-IF1 on the top-N candidates to filter for "the two engines
agree."

The cross-engine consensus workflow:

```python
from molforge.wrappers.generative import ProteinMPNN, ESMIF1

mpnn  = ProteinMPNN(num_seqs=8, sampling_temp=0.1)
esmif = ESMIF1(num_seqs=8, temperature=0.1)

mpnn_designs  = mpnn.generate(backbone)
esmif_designs = esmif.generate(backbone)

# Look for residues both engines agree on (cheap consensus filter).
mpnn_top  = mpnn_designs[0].sequence
esmif_top = esmif_designs[0].sequence
agreement = sum(a == b for a, b in zip(mpnn_top, esmif_top)) / len(mpnn_top)
print(f"Top-design agreement: {agreement:.0%}")
```

### You want to generate a novel backbone

**Use RFdiffusion.** It generates the *coordinates* — no sequence.
The output is a poly-glycine backbone (all residue identities GLY)
with a defined 3D shape.

Common cases:

- **Unconstrained generation.** "Generate a 100-residue protein"
  → RFdiffusion picks a fold, returns coordinates.
- **Motif scaffolding.** "Generate a backbone that displays this
  6-residue motif in this geometry" → RFdiffusion fills in the
  rest of the backbone around the fixed motif.
- **Binder design.** Generate a backbone that binds a target
  surface — provide the target as input, specify a hotspot, and
  RFdiffusion builds a complementary fold.

You'll then pass the backbone to ProteinMPNN to give it a sequence.

### You want to combine them

The standard *de novo* design pipeline:

```python
from molforge.wrappers.generative import RFdiffusion, ProteinMPNN
from molforge.wrappers.folding import ESMFold
from molforge.structure import rmsd

backbones = RFdiffusion().generate(length=80, num_designs=10)

mpnn = ProteinMPNN(num_seqs=8, sampling_temp=0.1)
folder = ESMFold()

results = []
for backbone in backbones:
    for design in mpnn.generate(backbone):
        refolded = folder.predict(design.sequence)
        score = rmsd(refolded, backbone, subset="ca", align=True)
        results.append((backbone, design, refolded, score))

# Filter by refold RMSD — successful designs fold back to their target.
ok = [(d, score) for _, d, _, score in results if score < 2.0]
print(f"{len(ok)}/{len(results)} designs pass the refold test")
```

This is the workflow shown in detail in the
[de novo design example](../examples/de_novo_design.ipynb).

## Common dimensions

### Output shapes

These two engines deliberately return different types because they
*designed* different things:

```python
# RFdiffusion: a list of structures with backbone atoms only.
backbones = RFdiffusion().generate(length=80, num_designs=5)
# backbones[0]: Protein with N, CA, C, O atoms; all residues GLY.

# ProteinMPNN: a list of (sequence, score) pairs.
designs = ProteinMPNN().generate(backbone)
# designs[0]: DesignedSequence(sequence="MAVQ...", score=1.23)
```

Putting them together: feed `backbones[i]` (a Protein) into
ProteinMPNN's `generate`, get back a list of `DesignedSequence`.

### Installation footprint

| Engine        | Install                                                                            |
| ------------- | ---------------------------------------------------------------------------------- |
| RFdiffusion   | Manual clone of RosettaCommons/RFdiffusion. `RFDIFFUSION_HOME` env var. Weights download separately. |
| ProteinMPNN   | Manual clone of dauparas/ProteinMPNN. `PROTEINMPNN_HOME` env var. Weights ship in the repo. |
| ESM-IF1       | `pip install "molforge[ml]"` (pulls `fair-esm`) + `pip install torch-geometric`. Weights (~145 MB) download on first use. |

RFdiffusion and ProteinMPNN run as subprocesses against a cloned
upstream repo; ESM-IF1 runs as a Python library import — by far the
lightest install of the three.

### Confidence / quality signals

| Engine        | Signal                                                                           |
| ------------- | -------------------------------------------------------------------------------- |
| RFdiffusion   | None directly; quality is judged by *what you do with the backbone next* (refold, design, dock). |
| ProteinMPNN   | `design.score` — negative log-likelihood (lower = more native-like for the fold). |
| ESM-IF1       | `design.score` — same convention (negative log-likelihood, lower better). Plus `design.recovery` against the native sequence when applicable. |

Neither MPNN score nor ESM-IF1 score is a refold-quality predictor
on its own — always check by refolding (see
[Design then refold](design-then-refold.md)).

## What molforge doesn't wrap (yet)

- **LigandMPNN** — ProteinMPNN extension that conditions on bound
  ligands. Natural follow-up to ProteinMPNN. Roadmap.
- **Chroma, FrameDiff** — alternative diffusion backbones to
  RFdiffusion. Roadmap.
- **AbLang, IgLM** — antibody-specific language models. No plans.
- **dyMEAN, IgFold** — antibody-structure-specific design tools.
  No plans (community use lower).

For an engine that's not yet wrapped, see
[Plugin authoring](../guide/plugins.md).
