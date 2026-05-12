"""Thin wrappers around external engines.

Each wrapper exposes a stable, typed interface to a third-party tool
(folding, docking, MD) so that engines are swappable from user code.
Heavy dependencies are imported lazily inside method bodies to keep
``import biocore`` cheap.
"""

from __future__ import annotations

__all__: list[str] = []
