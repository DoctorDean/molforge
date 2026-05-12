# Reference plugins

This directory contains **reference / example plugins** that demonstrate
how to extend `biocore` from an external package via the
`biocore.plugins` entry-point group.

These are intentionally separate from the main `biocore` package — they
are *not* installed by `pip install biocore`. To use them, install them
explicitly:

```bash
pip install -e plugins/example_plugin
```

The example below shows the minimum viable plugin layout.
