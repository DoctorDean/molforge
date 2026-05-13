# molforge-example-plugin

A minimal reference plugin showing how to integrate with `molforge` via
the `molforge.plugins` entry-point group.

## Install

```bash
pip install -e .
```

## Use

```python
from molforge.plugins import discover, get

discover()                            # walks entry points
engine_cls = get("engine", "example") # the engine registered below
engine = engine_cls()
```
