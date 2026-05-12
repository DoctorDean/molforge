"""Bump the project version in src/biocore/__init__.py.

Usage:
    python scripts/bump_version.py 0.0.2
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

INIT = Path(__file__).resolve().parent.parent / "src" / "biocore" / "__init__.py"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    new = sys.argv[1]
    text = INIT.read_text()
    new_text = re.sub(r'__version__ = "[^"]+"', f'__version__ = "{new}"', text)
    if new_text == text:
        print("No version line found.")
        return 2
    INIT.write_text(new_text)
    print(f"Bumped to {new}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
