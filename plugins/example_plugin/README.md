# molforge-example-plugin

A minimal but **working** reference plugin showing how to integrate with
`molforge` via the `molforge.plugins` entry-point group. It registers
`ExtendedChainFolder`, a real
[`FoldingEngine`](https://doctordean.github.io/molforge/reference/wrappers/folding/)
that lays a sequence out as an extended Cα chain — a faithful template you
replace with your own engine.

## Install

```bash
pip install -e .
```

## Use

```python
from molforge.plugins import discover, get

discover()                             # walks entry points, runs register()
engine_cls = get("engine", "example")  # -> ExtendedChainFolder
engine = engine_cls()
protein = engine.predict("MKTAYIAKQR")  # a real molforge Protein
print(protein.metadata["mean_confidence"])
```

## Anatomy

| File | Role |
| ---- | ---- |
| `pyproject.toml` | declares the `molforge.plugins` entry point → `example_plugin:register` |
| `src/example_plugin/__init__.py` | the engine (`ExtendedChainFolder`) and the `register()` callable |

To build your own: copy this directory, rename the package, and swap the
engine body. That's the whole contract.
