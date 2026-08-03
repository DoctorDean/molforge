# Longevity & maintenance

A wrapper library ages differently from a normal one. molforge's value is
that it speaks for sixteen external engines at once — but those engines are
**not** dependencies it pins and controls. They are third-party tools that
change output formats, rename CLI flags, and move on their own timelines,
mostly behind GPUs that CI can't run. Left alone, a wrapper library rots
*silently*: a new release changes one CSV column and the parser returns
wrong numbers with a green test suite.

So longevity here is not a feature to ship once; it is a standing posture.
This page describes the near-term work that keeps molforge **installable,
runnable, and relevant** over time. It runs *alongside* the
[roadmap](roadmap.md) — the roadmap adds capability, this plan defends the
capability that already exists — and it extends the Trust chapter
(roadmap §1) from "verify the numbers" to "keep them verified as the world
moves."

## The threat model — rot, not missing features

There are two distinct failure modes, and they need different defences:

| Failure mode | What it looks like | Who guards it |
| --- | --- | --- |
| **Python dependency drift** | a `numpy` / `torch` / tooling bump breaks install or behaviour | mostly automated — `dependabot`, the 3-OS × 3-Python CI matrix, pinned tooling |
| **Wrapped-engine drift** | an external engine changes its output format, CLI, or version, and a wrapper silently misparses | *this plan* — `dependabot` can't see it, and CI can't install the GPU-only engines to catch it |

The first is largely solved and self-running. The second is the real
long-term risk, and the near-term plan below is aimed squarely at it.

## The core insight

The wrappers are a **depreciating asset** — every engine will eventually
change or be replaced (AlphaFold2 → 3, ESMFold → ESM3, Boltz-1 → 2, all
within this project's lifetime). The **appreciating asset** is everything
engine-agnostic: the canonical `Protein` / `AtomArray` data model, the
reference-validated analysis stack (numpy-only, with no upstream that can
break it), and the provenance / reproducibility layer. That core is what
installs and runs forever and stays identical across engine generations.

The strategy follows directly: **invest in the durable core, contract-test
the perishable wrappers, and offload the long tail to plugins.** molforge's
relevance is "the stable interface that survives engine churn," not any one
engine.

## What already protects longevity

The foundation is in place (see roadmap §1 and §2):

- **A numpy-only core with opt-in extras** — a broken engine dependency can
  never break the base install or the validated metrics.
- **Engine version recording + drift warnings** (`molforge.wrappers._versions`)
  — wrappers stamp the installed engine version into `Provenance` and warn
  non-fatally when it drifts outside the tested range.
- **Nightly real-engine smoke tests** (`.github/workflows/nightly.yml`) —
  exercise the CPU-installable engines against their *real* implementations,
  not mocks.
- **Dependabot + the 3-OS × 3-Python matrix** — weekly pip / actions bumps,
  validated on every PR.
- **A plugin registry** (`molforge.plugins`, entry points) — third parties
  can own the long-tail wrappers without forking.
- **Reproducibility** — `Provenance` and `pipeline.yaml` carry engine
  versions, so a run stays diagnosable and replayable even as engines move.

## The near-term plan

Five items, ordered quick-win → highest-leverage → the pieces that build on
them. Each is a self-contained change; none is a large undertaking.

### 1. Extras-install smoke test — *planned (quick win)*

**Why.** Catches dependency-*resolution* rot — the class of failure where an
extra stops installing cleanly (as when meeko silently needed `scipy`) —
which ordinary unit tests never exercise.

**The work.** A CI job, matrix over each extra (`docking`, `md`, `ml`,
`chem`, `prep`, `repro`, `docs`): `pip install -e ".[<extra>]"`, then import
a representative submodule. Trigger weekly and on any change to
`pyproject.toml`.

**Done when.** Every extra installs and imports in CI; a broken extra fails
loudly and early.

### 2. Golden-output fixtures — *planned (highest leverage)*

**Why.** The single most effective guard against silent wrapper rot: parse a
**real** engine output in CI with no engine, no GPU, and no subprocess, so an
upstream format change turns a test red immediately.

**The work.** Add `tests/fixtures/engine_outputs/<engine>/` holding one real
captured artifact per parser (P2Rank `predictions.csv`, Boltz
`confidence_*.json` + affinity JSON, Vina `out.pdbqt`, Gnina / DiffDock SDF,
fpocket output, MMPBSA `FINAL_RESULTS_MMPBSA.dat`, an alchemlyb dataframe,
…). For each parser, a test that loads the fixture and asserts the parsed
result. Record the exact engine version each fixture came from (feeds item
3).

**Done when.** Every result parser has at least one real-data golden test in
the *base* suite (no extras required), each tagged with its source version.
The delta over today's parser tests is *real captured outputs as committed
fixtures* rather than hand-built strings.

### 3. A single-source-of-truth version matrix — *planned*

**Why.** One place should define the tested version range per engine, so the
runtime drift warning, the docs, and the fixtures can never disagree.

**The work.** A registry in `_versions.py` (or a new `_support.py`):
`{engine: (distribution, min_tested, max_tested)}`, which the existing
`check_engine_version` drift check reads. Generate a
`docs/architecture/supported-engines.md` table from it, with a test that
fails if docs and registry diverge.

**Done when.** Bumping a tested range in one place updates both the runtime
warning and the published table; divergence fails CI.

### 4. An unpinned "canary" job — *planned (early warning)*

**Why.** Find out about a breaking upstream release *before* a user files a
bug. Nightly tests the known-good pinned range; nothing currently watches
the bleeding edge, and dependabot can't see non-Python tools (the Java
P2Rank binary, compiled docking engines).

**The work.** A scheduled workflow that installs the **latest unpinned**
version of each CPU-installable engine and tool, runs the real-engine subset
plus the item-2 fixtures, is **non-blocking**, and opens/annotates an issue
on failure.

**Done when.** A breaking upstream release turns the canary red and files an
issue within a week, while pinned nightly and release stay green.

### 5. Engine lifecycle tags + pruning — *ongoing process*

**Why.** Relevance decays from *accumulation* too — a handful of dead
wrappers makes an otherwise healthy project read as abandoned.

**The work.** A `status` per engine (`maintained` / `experimental` /
`deprecated`) surfaced in the docs, with a runtime warning for deprecated
ones; a quarterly pass to deprecate dead upstreams on a stated timeline,
announced in `CHANGELOG.md`.

## Maintenance cadence

The architecture is deliberately built so that one maintainer does **not**
have to track sixteen upstreams in real time. The recurring rhythm:

- **Weekly** — merge the green dependabot PRs; glance at nightly and the
  canary.
- **On a red canary** — triage the flagged engine; pin the last-good version
  and open a wrapper-fix issue.
- **Per release (roughly every 1–2 months)** — cut a version, update the
  tested-version matrix, note deprecations.
- **Quarterly** — review each engine's upstream health (item 5).
- **Continuously** — the one-week response target in
  [CONTRIBUTING](https://github.com/DoctorDean/molforge/blob/master/CONTRIBUTING.md)
  and the plugin API turn users and contributors into the mechanism that
  carries engines the core team can't babysit alone.

## Related

- [Roadmap](roadmap.md) — capability direction; this plan defends what the
  roadmap builds.
- [API stability](api-stability.md) — the versioning and deprecation
  contract this plan operates within.
