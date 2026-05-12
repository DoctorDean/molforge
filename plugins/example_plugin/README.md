# biocore-example-plugin

A minimal reference plugin showing how to integrate with `biocore` via
the `biocore.plugins` entry-point group.

## Install

```bash
pip install -e .
```

## Use

```python
from biocore.plugins import discover, get

discover()                            # walks entry points
engine_cls = get("engine", "example") # the engine registered below
engine = engine_cls()
```
