"""Plugin registry implementation.

The registry is a simple in-memory mapping keyed by ``(kind, name)`` pairs.
Discovery walks Python entry points under the ``molforge.plugins`` group.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import metadata
from typing import Any, cast

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


def discover() -> list[str]:
    """Discover and load plugins exposed via the ``molforge.plugins`` entry-point group.

    Each entry point should be a callable taking no arguments; it is
    expected to perform its own registration via the ``register_*``
    functions exported from this module.

    Returns:
        A list of entry-point names that were successfully loaded. If a
        plugin fails to import or its registration function raises, the
        exception is suppressed and the name is omitted from the
        returned list. (We deliberately don't propagate exceptions
        because one broken plugin shouldn't be able to break every
        downstream user of molforge.)

    Example
    -------
    A third-party plugin declared in its own ``pyproject.toml``::

        [project.entry-points."molforge.plugins"]
        my_docker = "my_pkg.molforge_integration:register"

    where ``my_pkg.molforge_integration:register`` is a callable that
    calls :func:`register_engine` (or :func:`register_parser` /
    :func:`register_scorer`) one or more times.

    From a user's code::

        from molforge.plugins import discover, get
        loaded = discover()
        # loaded == ["my_docker"]
        engine = get("engine", "my_docker_engine")()
    """
    loaded: list[str] = []
    # importlib.metadata.entry_points() returns an EntryPoints object on
    # 3.10+. The `group` filter is the modern API; older Pythons can
    # still call .select(group=...).
    try:
        eps = metadata.entry_points(group="molforge.plugins")
    except TypeError:
        # Older importlib.metadata didn't accept group=; fall back.
        # (Unreachable on Python >= 3.10, the supported floor, but kept
        # defensively.) The legacy .entry_points() returns a dict; cast
        # the result so its static type matches the try branch above.
        eps = cast("Any", metadata.entry_points()).get("molforge.plugins", [])

    for ep in eps:
        try:
            register_fn = ep.load()
            register_fn()
        except Exception:
            continue
        loaded.append(ep.name)
    return loaded


def clear() -> None:
    """Empty the registry. Primarily useful for tests."""
    _REGISTRY.clear()
