# Notebooks

This directory hosts runnable walkthroughs and examples.

- `walkthroughs/` — short, didactic notebooks that go end-to-end for a single capability area.
- `examples/`     — longer real-world examples that combine multiple subpackages.

All notebooks should:

- Run top-to-bottom on a fresh kernel.
- Pin any model weights / data assets they depend on.
- Print version info at the top (`import biocore; print(biocore.__version__)`).
- Be small enough to render on GitHub (clear outputs before committing if heavy).
