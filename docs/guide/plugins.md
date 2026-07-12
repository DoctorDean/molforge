# Plugins

`molforge` ships with a small plugin registry so third-party packages
can add engines, parsers, and scorers without forking the library.

A plugin is a Python package that:

1. Declares an entry point under the `molforge.plugins` group, and
2. Exposes a callable that registers one or more items via
   `register_engine`, `register_parser`, or `register_scorer`.

That's it. Users opt in by calling `molforge.plugins.discover()`.

## Writing a plugin

Suppose you've built a new docking engine and want it usable from
`molforge`. In your package's `pyproject.toml`:

```toml
[project.entry-points."molforge.plugins"]
my_docker = "my_pkg.molforge_integration:register"
```

Then in `my_pkg/molforge_integration.py`:

```python
from molforge.plugins import register_engine
from my_pkg import MyDocker

def register() -> None:
    register_engine("my_docker", MyDocker)
```

## Using a plugin

```python
from molforge.plugins import discover, get, available

loaded = discover()           # walk entry points, register everything
print(loaded)                 # ["my_docker"]
print(available("engine"))    # ["my_docker"]

engine_cls = get("engine", "my_docker")
engine = engine_cls()
```

## Discovery semantics

`discover()` deliberately swallows exceptions from individual
plugins: one broken third-party plugin can't break every downstream
user of `molforge`. The return value tells you which plugins
*actually* loaded — anything missing failed quietly.

For a full worked example (custom engine, custom parser, custom
scorer, all wired through one `register()` entry point), see the
[plugin authoring walkthrough](https://github.com/DoctorDean/molforge/blob/master/notebooks/walkthroughs/06_plugin_authoring.ipynb).

## Reference

- [`molforge.plugins`](../reference/plugins.md) — full API.
