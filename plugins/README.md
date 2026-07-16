# Reference plugins

This directory contains **reference / example plugins** that demonstrate
how to extend `molforge` from an external package via the
`molforge.plugins` entry-point group.

These are intentionally separate from the main `molforge` package — they
are *not* installed by `pip install molforge`. To use them, install them
explicitly:

```bash
pip install -e plugins/example_plugin
```

## Using this as a template

`example_plugin/` is a complete, working plugin — the fastest way to start
your own is to copy it into a new repository:

```bash
cp -r plugins/example_plugin /path/to/my-molforge-plugin
```

Then rename the package (`src/example_plugin/` and the `pyproject.toml`
`name` / entry-point target), replace `ExtendedChainFolder` with your real
engine, and publish. `molforge.plugins.discover()` will pick it up wherever
it's installed.
