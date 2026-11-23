"""A visual editor for molforge pipelines.

The library's engines compose into pipelines, and a pipeline is a graph —
fold a sequence, search its homologues, dock a ligand against the result.
Written as a script that structure is implicit; drawn as blocks and arrows
it is the thing itself, which is the case for a canvas rather than another
API surface.

The GUI adds no capability. It builds the same calls the wrappers already
expose, saves graphs as the ``pipeline.yaml`` of
:mod:`molforge.reproducibility`, and runs them through
:func:`~molforge.reproducibility.replay` — so a graph drawn here is a file
anyone can re-run without it, which is the only version of a research GUI
worth shipping.

Its dependencies (FastAPI, uvicorn) are an opt-in extra::

    pip install "molforge[gui]"

:mod:`molforge.gui.catalog` needs none of them: the block catalog is
derived from the engine registry with :mod:`inspect` alone, and is
importable anywhere molforge is.
"""

from __future__ import annotations

from molforge.gui.catalog import (
    BlockParameter,
    BlockPort,
    BlockSpec,
    build_catalog,
    catalog_to_dict,
)

__all__ = [
    "BlockParameter",
    "BlockPort",
    "BlockSpec",
    "build_catalog",
    "catalog_to_dict",
]
