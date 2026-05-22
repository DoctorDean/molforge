"""MkDocs build hook: stage Jupyter notebooks into the docs tree.

The canonical home for notebooks is the top-level ``notebooks/``
directory — that's where they're authored, executed, and where CI
runs them. The documentation site needs to *render* them, which
means the files have to physically exist somewhere under
``docs_dir`` for mkdocs to discover them and for the mkdocs-jupyter
plugin to convert them to HTML pages.

We used to bridge that gap with symlinks (``docs/walkthroughs/*.ipynb
-> ../../notebooks/...``). Symlinks turned out to be a bad fit:

- They don't survive extraction of a release tarball on Windows
  without the ``SeCreateSymbolicLinkPrivilege`` ("a required
  privilege is not held by the client").
- ``actions/checkout`` and other tooling don't reliably preserve
  them, so the docs build on CI saw dangling links and failed in
  strict mode.

This hook replaces the symlinks with a real copy. The notebooks live
in exactly one place in version control (``notebooks/``); the copies
under ``docs/`` are produced fresh on every build and are
git-ignored, so there's no duplication in the repo and no chance of
the two drifting apart.

Why ``on_config`` and not ``on_files``:

mkdocs-jupyter converts ``.ipynb`` files to pages during its own
``on_files`` event. Hooks run *after* plugins for the same event, so
copying the notebooks in a hook's ``on_files`` is too late —
mkdocs-jupyter has already scanned and would never see them. By
copying the files to disk in ``on_config`` (which runs before
mkdocs's own file discovery, which in turn runs before any plugin's
``on_files``), the notebooks are present on disk in time for the
normal pipeline to pick them up: mkdocs discovers them as files, and
mkdocs-jupyter renders them like any other notebook in ``docs_dir``.

See https://www.mkdocs.org/user-guide/configuration/#hooks and the
event ordering at
https://www.mkdocs.org/dev-guide/plugins/#events.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig

# (source dir under repo root, destination dir under docs/)
_NOTEBOOK_DIRS: tuple[tuple[str, str], ...] = (
    ("notebooks/walkthroughs", "walkthroughs"),
    ("notebooks/examples", "examples"),
)

# This hook lives at <repo>/docs/_hooks/copy_notebooks.py, so the repo
# root is two directory levels up from the hook file itself.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def on_config(config: MkDocsConfig) -> MkDocsConfig:
    """Copy notebooks into ``docs_dir`` before mkdocs scans for files.

    Runs once per build, and again on every live-reload rebuild under
    ``mkdocs serve``. Existing copies are overwritten so edits to the
    canonical notebooks always propagate to the rendered site.
    """
    docs_dir = Path(config["docs_dir"])

    for src_rel, dst_rel in _NOTEBOOK_DIRS:
        src_dir = _REPO_ROOT / src_rel
        dst_dir = docs_dir / dst_rel
        if not src_dir.is_dir():
            # Notebooks directory missing — skip rather than crash the
            # build here. The nav references will surface as
            # strict-mode warnings, which is the correct loud failure.
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)

        for notebook in sorted(src_dir.glob("*.ipynb")):
            shutil.copyfile(notebook, dst_dir / notebook.name)

    return config
