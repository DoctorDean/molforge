# Prepare for MD

You have a PDB — from RCSB, AlphaFold, or your own folding run — and
want to feed it to an MD engine. Raw PDBs almost never simulate
directly: they have crystallographic waters and buffers that need
removing, missing residues and side-chain atoms, no terminal caps,
and no hydrogens. This recipe runs them through
`molforge.prep.prepare_for_md`.

## Requirements

```bash
pip install "molforge[prep]"      # pulls openmm + pdbfixer
```

## The recipe

```python
from molforge.io import fetch, save
from molforge.prep import prepare_for_md

# Get a structure. fetch() pulls from RCSB by PDB ID.
raw = fetch("1AKE")               # adenylate kinase, a standard MD test case
print(f"Raw: {raw.n_residues} residues, "
      f"{(raw.atom_array.entity_type == 'water').sum()} waters")

# Run the four-step preparation pipeline.
ready = prepare_for_md(raw)
print(f"Prepared: {ready.n_residues} residues "
      f"(no waters, no buffers, capped, protonated)")

save(ready, "1ake_prepared.pdb")
```

## What `prepare_for_md` actually does

It composes four steps in order. Each is also available individually
if you want finer control:

1. **`remove_heterogens(protein)`** — strips waters, ions, ligands,
   buffer molecules, and other non-protein residues. By default
   keeps standard amino acids only.
2. **`fix_missing_atoms(protein)`** — adds missing side-chain heavy
   atoms (a real PDB often has missing density for flexible
   loops). Uses PDBFixer's template library.
3. **`add_caps(protein)`** — adds ACE (N-terminal) and NME
   (C-terminal) caps to every chain. This neutralises the
   chain-terminal charges that would otherwise behave unphysically
   for short fragments. For full-length proteins this is optional;
   most workflows do it for consistency.
4. **`add_hydrogens(protein, pH=7.4)`** — adds hydrogens with
   protonation states appropriate for the given pH. Default is
   physiological 7.4.

If you want the pipeline but at a non-default pH (e.g. simulating
an endosomal pH-5 environment), call the steps explicitly:

```python
from molforge.prep import (
    remove_heterogens, fix_missing_atoms, add_caps, add_hydrogens,
)

p = remove_heterogens(raw)
p = fix_missing_atoms(p)
p = add_caps(p)
p = add_hydrogens(p, pH=5.0)        # acidic compartment
```

## Keeping co-crystallised ligands

By default `remove_heterogens` strips ligands too — fine when you
plan to dock a new ligand into the simulated apo structure, wrong
when you want to simulate the holo complex. To keep ligands:

```python
from molforge.prep import remove_heterogens
apo_no_buffers = remove_heterogens(raw, keep_ligands=True)
```

Note that MD with ligands is significantly more involved — you'll
need ligand force-field parameters (Antechamber / OpenFF), which
sit outside molforge's prep scope.

## Verifying the output

Before committing hours of GPU time to a simulation, sanity-check
the prepared structure:

```python
arr = ready.atom_array

# 1. No non-standard residues left.
from molforge.core import is_standard_amino_acid
nonstd = [r.name for r in ready.iter_residues()
          if not is_standard_amino_acid(r.name)
          and r.name not in {"ACE", "NME"}]
assert not nonstd, f"Unexpected residues: {set(nonstd)}"

# 2. Hydrogens present.
n_h = (arr.element == "H").sum()
assert n_h > 0, "No hydrogens added — add_hydrogens failed silently?"
print(f"{n_h} hydrogens added")

# 3. Caps present (one ACE and one NME per chain).
caps_per_chain = {}
for residue in ready.iter_residues():
    if residue.name in {"ACE", "NME"}:
        caps_per_chain.setdefault(residue.chain_id, []).append(residue.name)
print(f"Caps per chain: {caps_per_chain}")
```

## Provenance

The prepared structure carries a 4-deep `Provenance` chain
documenting every step. If the input already had a provenance (e.g.
it came from `ESMFold().predict(...)`), the chain extends back
through it. See [Inspect provenance](inspect-provenance.md).

## What this recipe doesn't do

- **Solvation and ions.** Adding TIP3P water and neutralising Na⁺/Cl⁻
  is the next step — done by the MD engine wrapper itself (OpenMM's
  `prepare()` solvates by default). See [MD and RMSD](md-and-rmsd.md).
- **Loop modelling.** If the input has long missing loops (>5
  residues), PDBFixer's templates fall over. For aggressive loop
  modelling, run the input through a folding engine first
  (AlphaFold rebuilds loops well) and feed *that* to
  `prepare_for_md`.
- **Disulfide bond detection.** Cysteine-cysteine bonds need explicit
  declaration in the force field. PDBFixer handles common cases via
  template matching; for unusual geometries, post-process by hand.
