# Reproducibility

Most papers in structural biology don't ship reproducible code. molforge
already records *what produced an output* — every engine wrapper attaches a
[`Provenance`](../reference/core.md) (engine, version, parameters, inputs,
and a pointer to the step it consumed) to `result.metadata["provenance"]`.
[`molforge.reproducibility`](../reference/reproducibility.md) turns that
chain into a single, human-readable **`pipeline.yaml`** — the artifact a
methods section can point at.

```python
from molforge.reproducibility import emit_pipeline

folded = esmfold.predict(sequence)
docked = vina.dock(folded, ligand)

emit_pipeline(docked, "pipeline.yaml")
```

## The artifact

The manifest linearizes the provenance chain into ordered steps and adds a
consolidated environment block:

```yaml
molforge_pipeline: 1
generated: "2026-07-15T12:00:00+00:00"
environment:
  molforge_version: "0.8.0"
  python_version: "3.12.13"
  platform: "macOS-14.3-arm64"
  engines: {ESMFold: "1.0.3", Vina: "1.2.5"}
steps:
  - step: 1
    engine: ESMFold
    engine_version: "1.0.3"
    inputs: {sequence: "MKT..."}
    parameters: {recycles: 4}
  - step: 2
    engine: Vina
    engine_version: "1.2.5"
    inputs: {ligand: "lig.sdf"}
    parameters: {exhaustiveness: 8}
output: {type: DockingResult}
```

## Inspecting a manifest

You don't have to write a file to use it. Build the manifest in memory and
inspect it:

```python
from molforge.reproducibility import pipeline_manifest

m = pipeline_manifest(docked)
print(m.describe())
# pipeline (2 steps) — molforge 0.8.0
#   1. ESMFold v1.0.3
#   2. Vina v1.2.5

m.steps                 # list[PipelineStep]
m.environment           # the environment block
m.to_dict()             # plain dict — for logging, comparison, a DataFrame
```

`pipeline_manifest` (and `emit_pipeline`) accept any molforge output that
carries provenance — a `Protein`, `DockingResult`, `Pose`,
`DesignedSequence`, ... — or a `Provenance` instance directly.

## Engine and backend versions

A step records the version its wrapper could read at the time. For a
pip-installed engine that is exact. For a wrapper that shells out to a native
binary — fpocket, P2Rank, GROMACS, `sander`, gnina — there is nothing to read,
so those steps carry no version and the manifest has a blank exactly where a
reader wants a number.

`engine_versions` is the registry that answers the question directly:

```python
from molforge import engine_versions

for backend in engine_versions().values():
    if backend.available:
        print(f"{backend.name:14s} {backend.version or '(no version flag)'}")
```

Every backend molforge can drive is reported, installed or not — an absent
engine is a fact about the environment worth recording. Each
`BackendVersion` carries its `category` (`folding`, `docking`, `pockets`,
`md`, `freeenergy`, `generative`, `runtime`), how it was found (`kind`:
`python`, `executable`, or `repo`), whether it is `available`, the resolved
`location` for a binary, and a `detail` explaining any empty version. Those
are three different reasons a version can be missing and the registry keeps
them apart:

| `kind` | Found by | Version from |
| --- | --- | --- |
| `python` | `importlib.metadata` | the distribution — exact, free |
| `executable` | `shutil.which` | running its version flag, where it has one |
| `repo` | nothing — the caller passes `repo_dir` | undetectable; said so explicitly |

Nothing here raises, on any machine, with none of it installed.

The manifest uses the registry to fill its own blanks, under a separate
`engines_detected` key:

```yaml
environment:
  engines: {ESMFold: "1.0.3"}            # what ran, recorded at the time
  engines_detected: {fpocket: "4.1"}     # what is installed now
```

They are kept apart because `engines_detected` is the weaker claim: it
describes the machine when the manifest was written, which is not necessarily
what produced the output. If the run is long and the record matters, capture
`engine_versions()` up front rather than relying on the backfill.

Probing a native binary means running it, so the sweep spawns subprocesses —
for installed tools only, with a timeout, and memoized. Pass
`probe_executables=False` to skip every subprocess and still get availability
from `$PATH`.

## Formats and the `repro` extra

The in-memory manifest and its `to_dict()` / `to_json()` forms need **no
third-party dependency** — they're part of molforge's numpy-only core:

```python
emit_pipeline(docked, "pipeline.json", fmt="json")   # no extra needed
```

Reading and writing the **`.yaml`** form needs PyYAML, an opt-in extra so
the core stays light:

```bash
pip install "molforge[repro]"
```

Without it, `to_yaml()` / a `.yaml` `load_pipeline` raise a clear
`ImportError` with that install hint; JSON keeps working regardless. Load a
manifest back with [`load_pipeline`](../reference/reproducibility.md), which
picks the format by file suffix.

## Replaying a pipeline

`replay()` re-executes a manifest's chain, threading each step's output into
the next:

```python
from molforge.reproducibility import load_pipeline, replay

manifest = load_pipeline("pipeline.yaml")
result = replay(manifest, context={"ligand": "aspirin.sdf"})
```

It resolves each step's engine from the registry (molforge's own wrappers,
plus anything registered under [`molforge.plugins`](plugins.md)),
reconstructs the call from the recorded parameters, and runs it. Each
*operation* (`predict`, `dock`, …) has a **replay handler** that owns its
reconstruction, so the "which recorded input is the previous step's output
vs. a literal" wiring is handled per operation rather than guessed — a
docking step's receptor is threaded from the fold that preceded it, its
ligand comes from `context` or the recorded value.

Register a handler for a custom operation with `register_replay_handler`:

```python
from molforge.reproducibility import register_replay_handler

@register_replay_handler("my_op")
def _replay_my_op(engine_factory, step, upstream_output, context):
    ...
```

Replay is inherently partial: the engines must be installed, GPU steps need
the hardware (replay orchestrates, it doesn't provide compute), and inputs
that aren't literals must be supplied via `context`. A missing engine, an
operation with no handler, or an unresolvable input raises a clear
`ReplayError`. molforge ships `predict` and `dock` handlers in v1.

## Many outputs at once

A `PipelineManifest` describes one output. Plenty of results aren't one
output: a consensus structure stands on several folds, a ranking on a
screen of thousands. Emitting one manifest per contributing prediction
answers the question in a form nobody can read — and restates the shared
upstream work, the target prep and the MSA, once per prediction.

`aggregate_manifest` folds them into one:

```python
from molforge.reproducibility import aggregate_manifest

manifest = aggregate_manifest(ensemble.members)
print(manifest.describe())
# aggregate (5 outputs, 6 unique steps from 10, 4 deduplicated) — molforge 0.8.0
#   shared:
#     MMseqs2 -> 5 outputs  [6623ea6703b3]
```

Every distinct computation appears once. A thousand predictions off one
MSA give you one MSA step marked `contributes_to: 1000`, and a manifest
that is a page rather than a directory.

### How steps are recognised as the same

By `Provenance.content_id()` — a digest of the engine, version,
operation, parameters, inputs, *and the whole parent chain*, with
timestamps stripped. Equal ids mean the same work reached the same way,
so two chains that diverge at step 3 share steps 1 and 2 and split from 3
onwards.

That the ancestry participates is the part that matters: the same docking
run under two different folds is two steps, not one, because it isn't the
same work — it only looks like it from the last line.

This is the same notion `molforge.cache` uses to decide a recomputation
is redundant, shared deliberately so the cache's idea of identical work
and a manifest's idea of a shared ancestor can't drift apart. One
difference: `content_id` ignores the molforge version, so an aggregate
spanning two releases still sees what they had in common, whereas the
cache must treat a version bump as invalidating.

### Aggregating manifests off disk

By the time someone asks how a screen was produced, the outputs are
usually long gone and what survives is a directory of `pipeline.yaml`
files. `PipelineManifest.aggregate` starts from those instead:

```python
from pathlib import Path
from molforge.reproducibility import PipelineManifest, load_pipeline

runs = [load_pipeline(p) for p in Path("runs").glob("*.yaml")]
combined = PipelineManifest.aggregate(runs)
combined.to_yaml()
```

A manifest holds no parent pointers, but its steps are ordered
oldest-first, which says the same thing. The chain is rebuilt from that
ordering, so both routes produce **identical step ids** — a manifest read
off disk deduplicates against a live object on exactly the same terms.

### Reading one

| Accessor | What it gives you |
| --- | --- |
| `len(manifest)` | distinct steps |
| `manifest.n_outputs` | how many outputs are described |
| `manifest.total_steps` | chain positions *before* dedup |
| `manifest.shared_steps` | steps more than one output needs, most-shared first |
| `manifest.roots()` | originating steps |
| `manifest.step_by_id(id)` | one step |
| `manifest.summary()` | the four headline counts |

`shared_steps` is usually the one worth reading: a step that many outputs
depend on is the step whose settings matter most, and the one to check
first when a whole batch looks wrong.

The `summary` block is written to the file but never read back — it's
derived from the steps, so a hand-edited copy can't report counts its own
steps contradict.

`emit_aggregate` / `load_aggregate` mirror `emit_pipeline` /
`load_pipeline`, YAML or JSON, same `repro` extra for YAML.

## What v1 doesn't do

- **Provenance stays linear.** A single output's chain has one parent
  pointer, so `chain()` keeps meaning what it says. The many-to-one
  structure lives in the aggregate instead — which is where it belongs,
  being a property of the question you're asking of a *set* of outputs
  rather than of any one of them. An engine that genuinely fuses several
  inputs into one output still records the input it consumed as its
  parent; describe the rest by aggregating.
