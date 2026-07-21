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
  molforge_version: "0.6.0"
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
# pipeline (2 steps) — molforge 0.6.0
#   1. ESMFold v1.0.3
#   2. Vina v1.2.5

m.steps                 # list[PipelineStep]
m.environment           # the environment block
m.to_dict()             # plain dict — for logging, comparison, a DataFrame
```

`pipeline_manifest` (and `emit_pipeline`) accept any molforge output that
carries provenance — a `Protein`, `DockingResult`, `Pose`,
`DesignedSequence`, ... — or a `Provenance` instance directly.

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

## What v1 doesn't do

- **Single output, linear chain.** A manifest describes one output's
  provenance chain (provenance has a single parent pointer). Merging
  several outputs' chains — e.g. an entire `DesignTable` — into one manifest
  with shared-ancestry deduplication is a future extension.
