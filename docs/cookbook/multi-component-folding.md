# Multi-component cofolding

molforge wraps two AlphaFold-3-class engines that predict the
structure of biomolecular complexes — proteins, nucleic acids,
small-molecule ligands — in a single forward pass: **Boltz** and
**Chai-1**. Both engines expose a `predict_complex(spec)` method
that takes a unified input shape and returns a multi-chain `Protein`.

This recipe walks through the four most common use cases.

## The input shape

Multi-component prediction takes a `ComplexSpec` — a tuple of
typed `Entity` objects:

```python
from molforge.folding import ComplexSpec, Entity

spec = ComplexSpec(entities=(
    Entity(kind="protein", sequence="MKQHKAMIVAL..."),
    Entity(kind="ligand", smiles="CC(=O)OC1=CC=CC=C1C(=O)O"),
))
```

Entities are typed: `"protein"`, `"dna"`, `"rna"`, or `"ligand"`.
Polymers carry a one-letter `sequence`; ligands carry either a
`smiles` string or a CCD code (`ccd="ATP"`, `ccd="NAD"`,
`ccd="ZN"`). Chain IDs auto-assign A, B, C, ... in entity order;
override with `chain_id="X"` if you need a specific letter.

For the headline drug-discovery shape, use the convenience
constructor:

```python
spec = ComplexSpec.protein_ligand(
    protein_sequence="MKQHKAMIVAL...",
    ligand_smiles="CC(=O)OC1=CC=CC=C1C(=O)O",  # or ligand_ccd="ATP"
)
```

## Recipe 1: protein + small-molecule drug

Predict how aspirin binds to a target protein:

```python
from molforge.folding import ComplexSpec
from molforge.wrappers.folding import Boltz

spec = ComplexSpec.protein_ligand(
    protein_sequence=(
        "MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAAL"
        "AEAPADSGELARRLDCDARAMRVLLDALYAY"  # truncated for brevity
    ),
    ligand_smiles="CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
)
complex_struct = Boltz(use_msa_server=True).predict_complex(spec)

print(f"chains: {complex_struct.atom_array.n_chains}")
print(f"interface pTM: {complex_struct.metadata['iptm']:.3f}")
print(f"composite confidence: {complex_struct.metadata['confidence_score']:.3f}")
```

The returned `Protein` has two chains in its `atom_array`: chain
A (the protein) and chain B (the ligand atoms as hetero-atoms).
The `iptm` value is now meaningful — for single-chain inputs it
was always 0, but for a real complex it's the interface confidence
signal that should track binding-site quality.

## Recipe 2: antibody-antigen complex

Three protein chains: antibody heavy, antibody light, and antigen.
This is the standard shape for therapeutic antibody design:

```python
from molforge.folding import ComplexSpec, Entity
from molforge.wrappers.folding import Chai1

spec = ComplexSpec(entities=(
    Entity(
        kind="protein",
        sequence=heavy_chain_seq,
        chain_id="H",
        name="heavy",
    ),
    Entity(
        kind="protein",
        sequence=light_chain_seq,
        chain_id="L",
        name="light",
    ),
    Entity(
        kind="protein",
        sequence=antigen_seq,
        chain_id="A",
        name="antigen",
    ),
))
abag = Chai1(use_msa_server=True).predict_complex(spec)

# The interface pTM at the antibody-antigen contact is the
# headline antibody-design quality signal.
print(f"global iPTM: {abag.metadata['iptm']:.3f}")
# Per-chain-pair iPTM (when produced by the engine) gives the
# interface-specific confidence.
if "per_chain_pair_iptm" in abag.metadata:
    print(f"chain-pair iPTM matrix: {abag.metadata['per_chain_pair_iptm']}")
```

## Recipe 3: protein on DNA (transcription factor)

Many transcription factors bind specific DNA sequences. Predict
the protein-DNA complex with both strands of the binding site:

```python
spec = ComplexSpec(entities=(
    Entity(kind="protein", sequence=tf_sequence),
    Entity(kind="dna", sequence="ATCGTAATCG"),  # forward strand
    Entity(kind="dna", sequence="CGATTACGAT"),  # reverse complement
))
tf_complex = Boltz().predict_complex(spec)
```

For an RNA-binding protein, swap `kind="dna"` for `kind="rna"` and
use A/U/C/G alphabet.

## Recipe 4: homo-oligomers

When the same protein appears multiple times in a complex (e.g. a
homodimer enzyme), use `copies=N` on a single entity rather than
repeating the entity:

```python
spec = ComplexSpec(entities=(
    Entity(kind="protein", sequence=enzyme_seq, copies=2),  # homodimer
    Entity(kind="ligand", ccd="ATP"),                       # substrate
))
```

The output structure has chains A and B (both with the same
sequence) plus chain C (the ATP).

## Cross-checking Boltz and Chai-1

Both engines accept the same `ComplexSpec`, so cross-validating a
prediction is a one-line swap:

```python
boltz_pred = Boltz(use_msa_server=True).predict_complex(spec)
chai_pred  = Chai1(use_msa_server=True).predict_complex(spec)

# Both report iPTM on the same scale.
print(f"Boltz iPTM:  {boltz_pred.metadata['iptm']:.3f}")
print(f"Chai-1 iPTM: {chai_pred.metadata['iptm']:.3f}")
```

Two independent AlphaFold-3 reimplementations agreeing on the
binding geometry is a much stronger signal than either alone —
their training pipelines are independent, so the agreement isn't
trivially explained by shared distribution bias.

## What's in the result

The returned `Protein` carries all the usual molforge metadata
plus a few keys specific to multi-component prediction:

| Key                           | What it is                                                                                                  |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `complex_spec`                | The original `ComplexSpec` passed in.                                                                       |
| `confidence_per_residue`      | Per-residue pLDDT (uniform across all engines).                                                             |
| `mean_confidence`             | Mean pLDDT across the whole complex.                                                                        |
| `ptm`                         | Global predicted TM-score.                                                                                  |
| `iptm`                        | Interface pTM. *This is what matters for complexes* — pLDDT alone can be high with badly-modelled interfaces. |
| `per_chain_ptm`               | Per-chain pTM (when the engine produces it).                                                                |
| `per_chain_pair_iptm` (Chai)  | Pairwise interface pTM matrix (when present).                                                               |
| `pair_chains_iptm` (Boltz)    | Same idea from Boltz's confidence JSON when present.                                                        |
| `provenance`                  | A `Provenance` recording the engine, all kwargs, and a JSON-safe serialization of the spec.                 |

For complexes, the headline confidence signal is **iPTM**, not pLDDT.
A complex can have high per-residue pLDDT (the individual chains are
modelled well) but low iPTM (the engine got the chain-to-chain
geometry wrong).

## What's deliberately not in v1

- **Modified residues** (Boltz's `modifications` list, Chai's CCD
  per-residue overrides). The base `Entity` sequence is treated as
  unmodified; modified-residue support will land as an engine-
  specific kwarg in a follow-up commit.
- **Restraints** (Boltz pocket constraints, Chai covalent bonds /
  restraint files). Drop down to the engine's raw API for now.
- **Custom MSAs per entity** — use `use_msa_server=True` for v1.
- **Templates** — use the engine's underlying API for v1.

**Binding affinity (Boltz-2)** is supported: `Boltz(model_version="boltz2").predict_affinity(spec)`
folds a protein + single-ligand complex *and* predicts its affinity,
surfacing `metadata["affinity_value"]` (log-scale, lower = stronger) and
`metadata["affinity_probability"]` (0–1 binder probability) on the returned
structure.

For these advanced features, the underlying engine APIs remain
accessible — you can construct YAML / FASTA / restraints files
yourself and invoke the engine's CLI or Python entry point.
