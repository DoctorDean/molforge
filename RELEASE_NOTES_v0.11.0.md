# molforge v0.11.0

A molforge pipeline is a graph — fold a sequence, search its homologues, dock a ligand against the result. Written as a script that structure is implicit. This release makes it a thing you can draw.

```sh
pip install "molforge[gui]"
molforge-gui
```

Blocks for the engines, arrows for what feeds what, and a 3D view of whatever each step produced.

As a minor release, everything is additive and backwards compatible with 0.10.0. The editor is an opt-in extra, and the pipelines it reads and writes are the `pipeline.yaml` 0.10.0 already used.

---

## The editor adds no capability

That is the point of it. Every button maps to something the library already did:

| In the editor | Underneath |
| --- | --- |
| The block palette | the engine registry, read with `inspect` |
| Dragging a wire | the argument one engine's output becomes |
| **Save** | `pipeline.yaml` |
| **Open** | `load_pipeline()` |
| **Run** | `replay()` |

So a graph drawn on a canvas is an ordinary manifest, and someone who has never opened the editor can run it:

```python
from molforge.reproducibility import load_pipeline, replay

replay(load_pipeline("pipeline.yaml"), context={"ligand": "aspirin.sdf"})
```

The editor saving the library's existing format rather than one of its own is the constraint the whole thing is built around. There is no pipeline you can build here that only this tool understands, and nothing to migrate if you stop using it.

## Nothing hard-codes which engines exist

The palette, the ports and every setting are derived from the same engine registry `replay()` resolves against, plus the wrapper signatures themselves. A new wrapper that registers itself and records a known operation becomes a block with no change to the editor — and a wrapper whose signature changes cannot leave a stale knob behind in the UI.

Settings come from the annotations. A parameter typed `Literal["boltz1", "boltz2"]` becomes a dropdown holding exactly those options; a `bool` becomes a checkbox; a docking box becomes three numeric fields rather than a comma-separated string, because that argument decides whether a run docks to the pocket or to empty space and a typo in free text is invisible until the results make no sense.

Annotations are read as *text*, never resolved — the catalog finds out that ESMFold takes a `device` without importing torch, which is what keeps the palette cheap enough to rebuild on every page load.

## Ports are typed

A docking block takes a *structure* and a *ligand*; a fold produces a *structure*. Wires connect only when the types match, so feeding a docking result into a folding block is not a mistake you can make — it is a wire that will not draw.

Port types are semantic rather than Python types. Docking's receptor is annotated `Protein | str | PathLike`, which describes how forgiving the wrapper is, not what a graph should hand it.

## Running, a block at a time

Each block shows queued, running, done or failed, with how long it took and a one-line description of what it produced.

- **A failure keeps the work before it.** A pipeline that dies in step three still built the structures from steps one and two. `replay()` returned only the terminal output and discarded the rest; it is now built on a new public `replay_steps()` generator that yields each step as it lands, so progress, per-step timing and partial results are available to anyone driving a long pipeline, not just the editor. `replay()` itself is unchanged.
- **Cancel takes effect between steps.** It cannot interrupt one — a step is a subprocess or a GPU call, and killing it mid-write is how a half-written file gets mistaken for a finished one.
- **One run at a time**, enforced server-side, because the client is a browser tab someone can double-click and the thing on the other end is a GPU.

## Results, in 3D

Clicking a finished block shows what it produced:

| Step produced | You see |
| --- | --- |
| A structure | the fold, cartoon-coloured along the chain |
| An ensemble | each model, one per row |
| A docking result | the receptor plus every pose — select a row to show that pose and dim the others |
| Pockets | the protein they were found in, with each centre marked |
| Designed sequences | sequences with scores and recovery |
| An alignment | its depth, width, and the hits it found |

Outputs are written to `editor-runs/` under molforge's cache directory rather than held in memory, so a long-lived editor doesn't pin every intermediate of every run — and the structure you are looking at is a file you can keep. Old runs are pruned as new ones start.

3Dmol is bundled rather than loaded from a CDN. The editor is a local tool, and no network is exactly the situation someone is in while waiting on a long local run.

## Drawing for a machine that isn't this one

A pipeline built on a laptop for a GPU cluster is a valid thing to build, so **Check** answers two questions separately: whether the wiring is coherent, and whether every engine is also installed *here*. A graph can be the first without the second, and the editor will still save it. Copy the `pipeline.yaml` to the cluster and replay it there.

Engines that aren't installed stay in the palette, dimmed, with the reason. Knowing molforge can drive Gnina is worth more than a palette that pretends it can't.

## Pipelines are chains, not trees

A manifest is a linear sequence of steps, so one fold feeding three different docking blocks is an ordinary thing to want and not something the format can express. The editor refuses it at the branching block rather than silently saving one arbitrary branch.

Lifting that means a manifest whose steps form a DAG — a `pipeline.yaml` schema change affecting every reader of it. It is now the roadmap's **Next**.

---

## Fixed

The editor drives `replay()`, which turned out to be the most thorough test it had ever had. Four defects, all of them silent:

- **`replay()` could not re-run a real docking step.** The handler called `dock(receptor, ligand)` and dropped every recorded call argument. Vina and Gnina both require `center`, so every docking step failed with a `TypeError`. The search box and the rest are now threaded through, back as the tuples the engines declare.
- **`replay()` could not resolve the generative engines.** A `generate` handler existed, but ProteinMPNN, ESM-IF1 and RFdiffusion were never registered, so a recorded design step failed with "no engine registered".
- **`engine_versions()` missed Chai-1 and MMseqs2.** The backend registry keyed Chai-1 as `Chai1` while every Chai-1 run records `Chai-1`, and MMseqs2 had no row at all. Both fail quietly: the manifest simply omits the version of the engine that ran. MMseqs2 now sits under a new `msa` category.
- **The sdist shipped a developer's virtualenv** — 4,400 files, 16.5 MB, now 1.1 MB. The root `.gitignore` matches `.venv/` but not `.venv-arm64/`, which git ignores only through the nested file `uv` writes inside it, and hatchling reads the root file alone. CI has no virtualenv in its tree, which is why it was never caught there.

Each of these had a guard test that should have caught it and didn't. The replay tests used fake engines that accept any arguments; the version test asserted against a hand-written list that restated the bug. Both now derive their expectations from the engine registry and the shipped wrapper classes, so a wrapper that drifts from what replay can resolve fails the suite.

## Tested by looking at it

The editor's two worst bugs — a 3D canvas that rendered nothing, and a view that opened every graph at 4× zoom — were invisible to both pytest and the type checker, and were found by opening a browser.

So CI now does that too: a job drives Chromium against a real `molforge-gui` and checks that a mismatched wire is refused, that **Check** blames the right block, that a run reports on the block it came from, that the 3D canvas actually draws, and that a saved pipeline reopens as the same graph. The regression test for the blank canvas was itself checked against the bug — removing the fix makes it fail.

---

## Installing

```sh
pip install "molforge[gui]"     # the editor
pip install "molforge[all]"     # everything, editor included
```

The wheel ships the editor already built; there is no Node toolchain to install. A source checkout builds it with `cd gui && npm install && npm run build`.

**The editor is a local tool.** It runs engines and reads files at paths the page asks for, with no authentication, so it binds to loopback. `--host` exists; exposing it on a shared network makes it a remote code execution service for anyone who can reach the port.
