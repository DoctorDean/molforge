"""Reference biocore plugin — registers a no-op example engine."""

from __future__ import annotations

from biocore.plugins import register_engine


class ExampleEngine:
    """A trivial engine that does nothing — exists to demonstrate registration."""

    def run(self) -> str:
        return "hello from example plugin"


def register() -> None:
    """Entry-point callable. Called by `biocore.plugins.discover()`."""
    register_engine("example", ExampleEngine)
