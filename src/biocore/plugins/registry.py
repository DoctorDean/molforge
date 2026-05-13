"""Plugin registry implementation.

The registry is a simple in-memory mapping keyed by ``(kind, name)`` pairs.
Discovery walks Python entry points under the ``molforge.plugins`` group.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_REGISTRY: dict[tuple[str, str], Any] = {}


def register_engine(name: str, factory: Callable[..., Any]) -> None:
    """Register a docking / folding / MD engine factory by name."""
    _REGISTRY[("engine", name)] = factory


def register_parser(name: str, factory: Callable[..., Any]) -> None:
    """Register a file-format parser by name (typically the extension)."""
    _REGISTRY[("parser", name)] = factory


def register_scorer(name: str, factory: Callable[..., Any]) -> None:
    """Register a scoring function by name."""
    _REGISTRY[("scorer", name)] = factory


def get(kind: str, name: str) -> Any:
    """Return a registered factory, raising ``KeyError`` if absent."""
    return _REGISTRY[(kind, name)]


def available(kind: str | None = None) -> list[str]:
    """List registered plugin names, optionally filtered by ``kind``."""
    return [n for (k, n) in _REGISTRY if kind is None or k == kind]


def discover() -> None:
    """Discover and load plugins exposed via the ``molforge.plugins`` entry-point group.

    Each entry point should be a callable taking no arguments; it is
    expected to perform its own registration via the ``register_*``
    functions.
    """
    # TODO: use importlib.metadata.entry_points(group="molforge.plugins")
    raise NotImplementedError
