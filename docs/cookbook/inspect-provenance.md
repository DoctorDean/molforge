# Inspect provenance

Every output from a molforge engine wrapper carries a `Provenance`
record describing what produced it — the engine name and version,
the parameters, the inputs, and a `parent` pointer to the upstream
step. Walking this chain reconstructs the full workflow that led
to any given output.

This recipe shows how to inspect provenance: print a chain, query
specific steps, serialise to JSON, and reconstruct.

## Where provenance lives

Any molforge object with a `metadata` dict carries provenance under
the documented key:

```python
from molforge.core import metadata_keys as mk

prov = result.metadata[mk.PROVENANCE]
```

The objects with provenance attached after wrapper adoption:

| Object              | Where provenance is set                            |
| ------------------- | -------------------------------------------------- |
| `Protein`           | Folded outputs (ESMFold, AlphaFold, Boltz, RoseTTAFold), `load_alphafold`, RFdiffusion designs, every `molforge.prep` function. |
| `DockingResult`     | Vina and DiffDock outputs.                         |
| `DesignedSequence`  | ProteinMPNN outputs (all designs from one call share the same Provenance). |
| `Simulation`        | OpenMM / GROMACS `prepare` and `minimize` outputs. |
| `Trajectory`        | OpenMM / GROMACS `run` outputs.                    |

## Printing the chain

The most common operation: "what produced this?"

```python
from molforge.io import fetch
from molforge.prep import prepare_for_md
from molforge.wrappers.md import OpenMM
from molforge.core import metadata_keys as mk

protein = fetch("1AKE")
ready = prepare_for_md(protein)
sim = OpenMM(platform="CPU").prepare(ready, force_field="amber14-all")
sim = OpenMM(platform="CPU").minimize(sim, max_iterations=100)
trajectory = OpenMM(platform="CPU").run(sim, n_steps=1000, save_every=100)

prov = trajectory.metadata[mk.PROVENANCE]
for step in prov.chain():               # oldest first
    print(f"{step.engine:35}  {step.timestamp}")
```

Output (timestamps redacted):

```
molforge.prep.remove_heterogens     2026-06-25T...
molforge.prep.fix_missing_atoms     2026-06-25T...
molforge.prep.add_caps              2026-06-25T...
molforge.prep.add_hydrogens         2026-06-25T...
OpenMM.prepare                       2026-06-25T...
OpenMM.minimize                      2026-06-25T...
OpenMM.run                           2026-06-25T...
```

Seven steps, every one with its own engine name, parameters, and
timestamp. `chain()` returns oldest-first (the originating step is
first, this step is last). `walk()` is the same but newest-first
— same data, different iteration order.

## Inspecting individual steps

Each `Provenance` is a frozen dataclass with named fields:

```python
top_step = prov                          # the newest step (the one attached)
print(top_step.engine)                   # "OpenMM.run"
print(top_step.engine_version)           # "" (engine doesn't expose one)
print(top_step.molforge_version)         # "0.4.0"
print(top_step.timestamp)                # "2026-06-25T..."
print(top_step.parameters)
# {'n_steps': 1000, 'save_every': 100, 'timestep_ps': 0.002, ...}
print(top_step.inputs)
# {} for run; intermediate; the previous step is the parent.
print(top_step.parent.engine)            # "OpenMM.minimize"
```

`prov.depth` is the number of steps in the chain (`7` in this
example).

## Filtering by engine

A common analysis: "find the folding step in this workflow":

```python
fold_step = next(
    (step for step in prov.chain()
     if step.engine in {"ESMFold", "AlphaFold", "Boltz", "RoseTTAFold"}),
    None,
)
if fold_step:
    print(f"Folded with {fold_step.engine}, "
          f"model={fold_step.parameters.get('model_name')}")
```

Or filter by category:

```python
md_steps = [s for s in prov.chain() if s.engine.startswith("OpenMM")]
print(f"{len(md_steps)} MD steps")
```

## Saving and loading

Provenance has a stable JSON round-trip shape — useful when you
want to record what produced an output alongside the output file:

```python
import json

# Save alongside the structure.
from molforge.io import save
save(ready, "ready.pdb")
with open("ready.provenance.json", "w") as f:
    f.write(ready.metadata[mk.PROVENANCE].to_json())

# Reload later.
from molforge.core import Provenance
with open("ready.provenance.json") as f:
    prov = Provenance.from_json(f.read())
print(prov.chain()[-1].engine)
```

**The PDB and mmCIF writers do not carry provenance** — they preserve
only the six documented IO header keys. Serialise to a sidecar JSON
explicitly if you want to keep it around. A future "molforge bundle"
format may unify structure + provenance + extras, but for now the
sidecar is the supported pattern.

## Comparing two outputs

A reproducibility check: did two runs produce outputs with the same
parameters?

```python
prov_a = result_a.metadata[mk.PROVENANCE]
prov_b = result_b.metadata[mk.PROVENANCE]

# Compare the parameter dicts, ignoring timestamp.
def normalise(p):
    d = p.to_dict()
    d.pop("timestamp", None)
    if d.get("parent"):
        d["parent"] = normalise(p.parent)
    return d

assert normalise(prov_a) == normalise(prov_b), "runs differ!"
```

Two runs of the same workflow on the same inputs will produce the
same chain modulo `timestamp` and (sometimes) `molforge_version` —
that's the contract `to_dict` is designed to support. Anything else
is either a difference in inputs / parameters or a non-determinism
bug in the engine.

## Building a cache key

Provenance's `to_dict` shape is content-addressable: hash it and you
have a stable identifier for "this exact computation on these exact
inputs." A future caching layer in molforge will use exactly this
approach. You can already build one yourself:

```python
import hashlib, json

def cache_key(prov):
    d = prov.to_dict()
    d.pop("timestamp", None)
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()

print(cache_key(prov))
```

This is the kind of thing the caching layer will absorb when it
ships; in the meantime it's a useful one-liner.

## What provenance doesn't capture

- **Random seeds for steps that don't expose them.** Some engines
  (especially deep-learning ones) make non-determinism hard to
  pin down. If a workflow says `seed=42` but produces different
  outputs across runs, the engine itself is the source of the
  variance, not provenance.
- **The actual input bytes.** Provenance records *identifiers*
  (paths, sequences, IDs) but not the file contents at those
  paths. A receptor file that's mutated under the same path
  produces different outputs with identical provenance — the
  identifier didn't change.
- **External state.** Things like `$CUDA_VISIBLE_DEVICES`, the
  installed version of an external binary (Vina, gmx), or the
  GPU driver version aren't captured. For full reproducibility,
  pair provenance with a frozen environment (Docker, conda lock
  file).

Provenance is *workflow* documentation, not *reproducibility-
under-arbitrary-environments* documentation. It tells you what
molforge did; it relies on you to track the surrounding context.
