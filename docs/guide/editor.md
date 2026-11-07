# The pipeline editor

A visual editor for molforge pipelines: blocks for the engines, arrows for
what feeds what.

```sh
pip install "molforge[gui]"
molforge-gui
```

That serves the editor at <http://localhost:8765> and opens it.

!!! warning "It is a local tool"

    The editor runs engines and reads files at paths the page asks for,
    with no authentication. It binds to loopback for that reason. `--host`
    exists, and exposing it on a shared network makes it a remote code
    execution service for anyone who can reach the port.

## What it is for

molforge pipelines are graphs — fold a sequence, search its homologues,
dock a ligand against the result. Written as a script that structure is
implicit; drawn as blocks and arrows it *is* the thing, which makes the
shape of an experiment visible and a mis-wiring obvious.

The editor adds no capability. Everything it does, the library already
did:

| In the editor | Underneath |
|---|---|
| The block palette | the engine registry, read with `inspect` |
| Dragging a wire | the argument one engine's output becomes |
| **Save** | [`pipeline.yaml`](reproducibility.md) |
| **Open** | `load_pipeline()` |
| **Run** | [`replay()`](reproducibility.md#replaying-a-pipeline) |

A graph drawn here is an ordinary manifest. Someone who has never opened
the editor can run it:

```python
from molforge.reproducibility import load_pipeline, replay

replay(load_pipeline("pipeline.yaml"), context={"ligand": "aspirin.sdf"})
```

That is the point of the editor saving the library's existing format
rather than one of its own — there is no pipeline you can build here that
only this tool understands.

## Blocks

Every engine molforge can drive appears as a block, derived from the same
registry `replay()` resolves against. A wrapper that registers itself and
records a known operation becomes a block with no change to the editor.

A block shows what it takes, what it produces, and — after a run — how it
went:

- **Ports** are typed. A docking block takes a *structure* and a *ligand*;
  a fold produces a *structure*. Wires only connect matching types, so you
  cannot feed a docking result into a folding block.
- **Settings** come from the wrapper's own signature. A parameter
  annotated `Literal["boltz1", "boltz2"]` becomes a dropdown holding
  exactly those options; a `bool` becomes a checkbox; the docking box
  becomes three numeric fields.
- **Engines that aren't installed** are listed and dimmed, with the reason.

### Three states for a setting

A setting that is **unset** is not the same as one sitting on its
**default**. `None` on an optional parameter means *let the engine decide*,
which for several wrappers is different from any value you could type. The
editor keeps them distinct, and so does the manifest it writes.

### Required settings

A parameter with no default has to be set before the graph will run.
Docking's `center` is the one that matters: it decides whether a run docks
to the pocket or to empty space, and the editor refuses to save a graph
without it rather than letting the engine fail later.

## Running

**Run** executes the chain a block at a time. Each block shows queued,
running, done or failed, with how long it took and a one-line description
of what it produced.

Three things worth knowing:

- **A failure keeps the work before it.** A pipeline that dies in step
  three still built the structures from steps one and two, and the editor
  says which step failed and what the engine said. The partial results are
  reported rather than discarded.
- **Cancel takes effect between steps.** It cannot interrupt one — a step
  is a subprocess or a GPU call, and killing it mid-write is how a
  half-written file gets mistaken for a finished one. A long fold runs to
  completion, then the pipeline stops.
- **One run at a time.** These are GPU calls and subprocesses pinned to
  every core; the server refuses a second rather than letting a
  double-clicked button start one.

## Drawing for another machine

A pipeline built on a laptop for a GPU cluster is a valid thing to build.
**Check** answers two questions separately:

- **Valid** — the wiring is coherent and it will serialize.
- **Runnable here** — every engine is also installed on this machine.

A graph can be the first without the second, and the editor will still
save it. Copy the `pipeline.yaml` to the cluster and `replay()` it there.

## Limits

**Pipelines are chains, not trees.** A manifest is a linear sequence of
steps, so one fold feeding three different docking blocks is a reasonable
thing to want and not something a manifest can express. The editor refuses
it at the branching block rather than silently saving one arbitrary
branch. Run them as separate pipelines for now.

**Unwired inputs share a namespace.** `replay()`'s `context` is keyed by
input name across the whole pipeline, so two blocks that each need their
own typed-in `ligand` cannot be told apart. The editor reports that rather
than letting one quietly win.

## Building from source

A wheel from PyPI ships the editor already built. A source checkout does
not — the assets are generated, so they are not committed — and the server
serves a page telling you so:

```sh
cd gui
npm install
npm run build
```

`npm run dev` runs the frontend with hot reload against a `molforge-gui`
server on port 8765, which is the way to work on the editor itself.
