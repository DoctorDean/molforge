# Reference plugins

This directory contains **reference / example plugins** that demonstrate
how to extend `molforge` from an external package via the
`molforge.plugins` entry-point group.

These are intentionally separate from the main `molforge` package — they
are *not* installed by `pip install biocore`. To use them, install them
explicitly:

```bash
pip install -e plugins/example_plugin
```

The example below shows the minimum viable plugin layout.
