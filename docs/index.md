---
hide:
  - navigation
  - toc
---

<div class="mf-hero" markdown>

![molforge](assets/emblem.png){ .mf-hero-logo alt="molforge" }

# molforge { .mf-hero-title }

<p class="mf-hero-tagline">
The connective tissue for computational biology — fold, dock, simulate,
and design across sixteen engines with one data model and no
format-conversion tax.
</p>

[Get started](getting-started/installation.md){ .md-button .md-button--primary }
[Quickstart](getting-started/quickstart.md){ .md-button }
[GitHub :fontawesome-brands-github:](https://github.com/DoctorDean/molforge){ .md-button }

</div>

<div class="mf-install" markdown>

```bash
pip install molforge
```

</div>

It is a *library*, not a framework: no orchestrator, no DAG runtime, no
decorators to import. Bring your structures and sequences in, plug in your
engines of choice, and walk out with a coherent pipeline instead of five
incompatible Python environments and a graveyard of conversion scripts.

## What it does

<div class="grid cards" markdown>

-   :material-dna:{ .lg .middle } &nbsp; **Folding**

    ---

    ESMFold, AlphaFold/ColabFold, Boltz, Chai-1, RoseTTAFold — with
    multi-component cofolding on the AF3-style engines.

-   :material-target:{ .lg .middle } &nbsp; **Docking**

    ---

    AutoDock Vina, Gnina (CNN rescoring), DiffDock — with automatic
    meeko/RDKit ligand prep.

-   :material-atom:{ .lg .middle } &nbsp; **Molecular dynamics**

    ---

    OpenMM, GROMACS, AMBER behind one `prepare → minimize → run` interface,
    plus trajectory I/O and analysis.

-   :material-auto-fix:{ .lg .middle } &nbsp; **Generative design**

    ---

    RFdiffusion for backbones, ProteinMPNN and ESM-IF1 for sequence design —
    the full *de novo* loop in one library.

-   :material-scale-balance:{ .lg .middle } &nbsp; **Binding free energy**

    ---

    MM-PB(GB)SA via AmberTools and gmx_MMPBSA, FEP/TI ingestion through
    alchemlyb and cinnabar, plus Boltz-2 binding-affinity prediction.

-   :material-magnify-scan:{ .lg .middle } &nbsp; **Pocket detection**

    ---

    fpocket (geometric) and P2Rank (ML) surface-pocket detection, feeding
    straight into the docking workflow.

</div>

## The glue — an engine-agnostic layer

<div class="grid cards" markdown>

-   :material-layers-triple:{ .lg .middle } &nbsp; **Cross-engine ensembles**

    ---

    `cross_engine_fold` folds one sequence with several engines and returns
    the pairwise TM/RMSD spread, a consensus, and a per-residue map of where
    they disagree — trust the regions your methods agree on.

-   :material-sync:{ .lg .middle } &nbsp; **Design loop**

    ---

    `DesignLoop` runs generate → fold → dock → score → iterate with a ranked
    design table — the protein-engineering loop in one object.

-   :material-medal:{ .lg .middle } &nbsp; **Unified scoring**

    ---

    `molforge.scoring` gives every score an explicit direction, so docking
    affinity, folding confidence, and learned scores rank and compare
    uniformly — and plug into the design loop.

-   :material-file-document-check:{ .lg .middle } &nbsp; **Reproducibility**

    ---

    Every output carries a `Provenance` chain; emit it as a citable
    `pipeline.yaml`, then `replay()` re-runs the whole workflow.

</div>

## Why molforge

<div class="grid cards" markdown>

-   **Workflows over silos**

    ---

    Every design decision is judged by one question: *does this make it
    easier to chain N tools together?*

-   **Wrappers, not reimplementations**

    ---

    We don't rebuild OpenMM or AutoDock. We give them a shared vocabulary —
    one `Protein`, one `Provenance`, one cache.

-   **One data model, two views**

    ---

    Hierarchical (`protein.chains["A"].residues[42]`) for biology, linear
    (`protein.atom_array.coords`) for ML — same data, no conversion.

-   **Typed, tested, documented**

    ---

    Strict mypy, ruff-clean, a large test suite on a 3-OS × 3-Python matrix,
    every public symbol with a Google-style docstring.

</div>

## Where to go next

- **New here?** [Installation](getting-started/installation.md) and the
  [Quickstart](getting-started/quickstart.md).
- **Trying to do something specific?** The [Cookbook](cookbook/index.md) has
  task-oriented recipes and decision tables for choosing engines.
- **Want the design rationale?** The
  [Architecture overview](architecture/overview.md) and
  [Roadmap](architecture/roadmap.md).
- **Looking for a symbol?** Browse the [API reference](reference/core.md) or
  hit the search box.

---

`molforge` is MIT-licensed. Issues and pull requests are welcome at
[github.com/DoctorDean/molforge](https://github.com/DoctorDean/molforge).
