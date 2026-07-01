# Check a structure for steric clashes

Folding and docking engines can hand back structures with atoms jammed
into each other — bad rotamers, backbone overlaps, a ligand pose that
bumps the receptor. A quick **clash check** is a cheap gate to run
before you spend compute on anything downstream.

molforge scores clashes from geometry alone (coordinates + elements),
so it works on any `Protein` without a separate topology file.

## The one-liner gate

```python
from molforge.structure import clash_score, has_clashes

model = esmfold.predict("MKQHKAMIVAL...")

if has_clashes(model):
    print("rejecting model — it has steric clashes")

print("clashscore:", clash_score(model))  # clashes per 1000 atoms
```

`clash_score` returns clashes per 1000 atoms (the MolProbity
convention), so it is comparable across structures of different sizes.
A well-refined structure scores near zero; a handful of clashes on a
few-hundred-atom model pushes it into the tens.

## Looking at the actual clashes

`find_clashes` returns the offending pairs, worst overlap first:

```python
from molforge.structure import find_clashes

for c in find_clashes(model)[:5]:
    print(
        f"{c.element_i}{c.residue_i[1]} – {c.element_j}{c.residue_j[1]}"
        f"  d={c.distance:.2f} Å  overlap={c.overlap:.2f} Å"
    )
```

Each `Clash` carries the two global atom indices (`atom_i`, `atom_j`),
their elements, the `distance`, the van der Waals `vdw_sum`, the
`overlap` (`vdw_sum - distance`), and a
`(chain, residue_id, residue_name)` label for each atom's residue.

## What counts as a clash

Two atoms clash when their van der Waals shells overlap by at least
`tolerance` Å:

```
overlap = (vdw_radius_i + vdw_radius_j) - distance   # clash if >= tolerance
```

The default `tolerance=0.4` matches the MolProbity all-atom-contact
threshold. Lower it to surface softer contacts:

```python
soft = find_clashes(model, tolerance=0.2)
```

Covalently bonded atoms naturally sit well inside each other's vdW sum.
Since molforge has no bond graph, bonds are **inferred from geometry**
and any pair within `bonded_separation` bonds (default 3 — i.e. 1-2,
1-3 and 1-4 neighbours) is excluded. This handles intra-residue bonds,
the peptide bond, disulfides, and bonds inside a ligand without any
special-casing. Set `bonded_separation=0` to see every raw overlap.

Hydrogens are ignored by default (folding output usually lacks them);
pass `include_hydrogens=True` if your model has them and you want
hydrogen contacts counted too.

## Water and ligands

Clash detection looks at every atom you hand it. A crystallographic
water making a short (2.2–2.5 Å) contact, or a bound ligand, will show
up. Filter first if you only care about the polymer:

```python
protein_clashes = find_clashes(model.remove_water())   # drop solvent
polymer_only = find_clashes(model.protein_only())       # drop ligands + solvent too
```

## Gating a design pipeline

Clash detection composes with the rest of the library — for example,
reject designs whose re-folded structure clashes before you bother
scoring them:

```python
from molforge.structure import has_clashes

keep = []
for design in designs:
    refolded = esmfold.predict(design.sequence)
    if not has_clashes(refolded):
        keep.append((design, refolded))
```

## Limitations

Because bonds are inferred by distance, two genuinely non-bonded atoms
closer than normal bonding distance (roughly < 2 Å for heavy atoms)
are mistaken for a bond and not reported — a pathology better caught as
a duplicate-atom check. Clashes in the range that actually matters
(≈ 2.0–3.2 Å heavy-atom overlaps) are detected reliably. For
topology-aware validation with explicit hydrogens, reach for a
dedicated tool such as MolProbity.
