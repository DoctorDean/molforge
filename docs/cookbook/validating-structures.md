# Validate structure quality

Folding and docking engines can hand back structures with atoms jammed
into each other, backbone conformations that shouldn't exist, or worse.
molforge's `structure` module has a set of cheap, geometry-only quality
gates — steric clashes and Ramachandran outliers today — that run on any
`Protein` without a separate topology file, so you can reject bad models
before spending compute on anything downstream.

This recipe covers each check in turn.

# Steric clashes

Bad rotamers, backbone overlaps, a ligand pose that bumps the receptor —
all show up as pairs of non-bonded atoms whose van der Waals shells
overlap.

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

## Clash detection limitations

Because bonds are inferred by distance, two genuinely non-bonded atoms
closer than normal bonding distance (roughly < 2 Å for heavy atoms)
are mistaken for a bond and not reported — a pathology better caught as
a duplicate-atom check. Clashes in the range that actually matters
(≈ 2.0–3.2 Å heavy-atom overlaps) are detected reliably. For
topology-aware validation with explicit hydrogens, reach for a
dedicated tool such as MolProbity.

# Ramachandran outliers

Every residue's backbone conformation is summarised by two dihedral
angles, φ and ψ. Real proteins only populate a few regions of that
plane; a residue sitting far outside them is a red flag for a modelling
error.

## Favored fraction and outliers

```python
from molforge.structure import (
    ramachandran_favored_fraction,
    ramachandran_outliers,
)

print("favored:", ramachandran_favored_fraction(model))  # 1.0 is perfect

for r in ramachandran_outliers(model):
    chain, resid, resname = r.residue
    print(f"{resname}{resid} ({chain})  φ={r.phi:.0f}  ψ={r.psi:.0f}")
```

`ramachandran_favored_fraction` is a single-number quality signal in the
spirit of MolProbity's "Ramachandran favored %": a well-refined
structure scores near 1.0, and a model full of impossible backbone
angles scores near 0.

## Per-residue classification

`classify_ramachandran` returns one record per residue that has a
defined (φ, ψ) — chain termini and residues across a chain break are
skipped:

```python
from molforge.structure import classify_ramachandran

for r in classify_ramachandran(model):
    print(r.residue, r.category, r.classification)
```

Each `RamachandranResult` carries the `(chain, residue_id, residue_name)`
label, the `phi`/`psi` angles, the `category` used to judge it, and the
`classification` (`Favored` / `Allowed` / `Outlier`). You can also
classify a bare angle pair:

```python
from molforge.structure import ramachandran_type

ramachandran_type(-63, -43)                      # "Favored"  (α-helix)
ramachandran_type(60, 45)                        # "Allowed"  (left-handed α)
ramachandran_type(60, 45, category="Glycine")    # "Favored"  (fine for Gly)
ramachandran_type(60, 45, category="Proline")    # "Outlier"  (ring forbids +φ)
```

The classifier splits residues into three region sets — **General**,
**Glycine** (whose plot is point-symmetric, since glycine is achiral),
and **Proline** (whose φ is pinned near −63° by its ring). Pre-proline
residues use the general regions.

## Ramachandran limitations

This is a *simplified* region model — unions of rectangles tuned to the
standard basins — not the 2D probability contours that MolProbity
estimates from tens of thousands of reference residues. It reliably
flags gross outliers and gives a useful favored fraction, but the exact
favored/allowed boundary is approximate. For publication-grade
percentiles, use a tool that ships the reference distributions.

