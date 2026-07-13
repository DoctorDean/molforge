"""Guard against docstrings that break the strict docs build.

The docs site renders the API with mkdocstrings, which parses every
public docstring with griffe's Google-style parser and runs in
``--strict`` mode, so a single parser warning aborts the build. The
usual culprits are grouping several parameters on one ``Args:`` line
(``a, b: ...`` — griffe reads the whole ``"a, b"`` as one parameter name)
and mis-indented continuation lines.

griffe is a docs-only dependency, so this test skips when it isn't
installed and runs wherever it is (the docs job, local doc builds).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

griffe = pytest.importorskip("griffe")

SRC = Path(__file__).resolve().parents[2] / "src"


def _collect_docstring_warnings() -> list[str]:
    from griffe import Parser

    messages: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    logger = logging.getLogger("griffe")
    handler = _Capture()
    logger.addHandler(handler)
    previous_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        package = griffe.load("molforge", search_paths=[str(SRC)], submodules=True)

        def walk(obj: object) -> None:
            docstring = getattr(obj, "docstring", None)
            if docstring is not None:
                docstring.parse(Parser.google)
            for member in getattr(obj, "members", {}).values():
                if getattr(member, "is_alias", False):
                    continue
                walk(member)

        walk(package)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
    return messages


def test_docstrings_parse_clean_under_griffe() -> None:
    warnings = _collect_docstring_warnings()
    assert warnings == [], (
        "griffe docstring warnings (would abort the strict docs build):\n" + "\n".join(warnings)
    )
