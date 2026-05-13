"""Plugin registry and discovery.

Third-party packages can register additional engines, parsers, or scoring
functions by exposing entry points under the ``molforge.plugins`` group:

.. code-block:: toml

    [project.entry-points."molforge.plugins"]
    my_docker = "my_pkg:register"

The function pointed to by the entry point should call
:func:`register_engine`, :func:`register_parser`, or :func:`register_scorer`
to make its capability available.
"""

from __future__ import annotations

from molforge.plugins.registry import (
    available,
    discover,
    get,
    register_engine,
    register_parser,
    register_scorer,
)

__all__ = [
    "available",
    "discover",
    "get",
    "register_engine",
    "register_parser",
    "register_scorer",
]
