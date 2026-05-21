"""Execute molforge's runnable notebooks as a CI smoke test.

Why this exists
---------------

Notebooks are easy to drift out of sync with their library. Outputs
get baked in once and rot quietly when the API changes underneath
them. This script catches that drift by executing each notebook
top-to-bottom against the currently installed molforge.

What gets executed
------------------

Notebooks that don't require heavy external dependencies (RFdiffusion,
ProteinMPNN, OpenMM, ColabFold, torch, vina) execute end-to-end.
Notebooks that do need those deps are still validated as parseable
nbformat JSON but not executed — we just inspect them.

The split is determined by an allowlist below. To add a notebook,
make sure its code cells either:

  (a) only use fixtures shipped in ``tests/fixtures/``, or
  (b) are explicitly marked ``# 🐢 SLOW`` (which we then skip).

Usage
-----

    python scripts/execute_notebooks.py
    python scripts/execute_notebooks.py --check-only   # don't execute, just parse-validate

Exit code is non-zero if any notebook fails to execute.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

# Notebooks that should execute end-to-end without external engines.
# Tested against the bundled fixtures plus inline-registered engines.
EXECUTABLE_NOTEBOOKS = [
    "notebooks/walkthroughs/01_sequences.ipynb",
    "notebooks/walkthroughs/02_structures.ipynb",
    "notebooks/walkthroughs/05_ml_featurization.ipynb",
    "notebooks/walkthroughs/06_plugin_authoring.ipynb",
    "notebooks/examples/cross_engine_validation.ipynb",
]

# Notebooks that require external engines — we just parse-validate them.
# They use ``# 🐢 SLOW`` comments on heavy cells but the wrapper imports
# themselves can fail without the engines installed.
PARSE_ONLY_NOTEBOOKS = [
    "notebooks/walkthroughs/03_md_simulations.ipynb",
    "notebooks/walkthroughs/04_docking.ipynb",
    "notebooks/examples/end_to_end_design.ipynb",
    "notebooks/examples/de_novo_design.ipynb",
]


def parse_validate(path: Path) -> tuple[bool, str]:
    """Read the notebook, run nbformat.validate. Returns (ok, message)."""
    try:
        nb = nbformat.read(str(path), as_version=4)
        nbformat.validate(nb)
        return True, f"  parse-ok  {path}  ({len(nb.cells)} cells)"
    except Exception as e:
        return False, f"  PARSE-FAIL {path}\n    {type(e).__name__}: {e}"


def execute(path: Path, *, timeout: int = 300) -> tuple[bool, str]:
    """Read, execute, and validate the notebook. Returns (ok, message)."""
    try:
        nb = nbformat.read(str(path), as_version=4)
    except Exception as e:
        return False, f"  PARSE-FAIL {path}\n    {type(e).__name__}: {e}"

    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    try:
        client.execute()
    except CellExecutionError as e:
        return (
            False,
            f"  EXEC-FAIL {path}\n"
            f"    Cell raised: {type(e).__name__}\n"
            f"    {str(e).splitlines()[-1] if str(e) else '(no message)'}",
        )
    except Exception as e:
        return (
            False,
            f"  EXEC-FAIL {path}\n    {type(e).__name__}: {e}\n    {traceback.format_exc(limit=2)}",
        )
    return True, f"  exec-ok   {path}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Skip notebook execution; only parse-validate every notebook.",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-cell execution timeout in seconds (default: 300).",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    failures = 0

    print("Parse-validating all notebooks:")
    all_notebooks = EXECUTABLE_NOTEBOOKS + PARSE_ONLY_NOTEBOOKS
    for relpath in sorted(all_notebooks):
        path = root / relpath
        if not path.exists():
            print(f"  MISSING   {relpath}")
            failures += 1
            continue
        ok, msg = parse_validate(path)
        print(msg)
        if not ok:
            failures += 1

    if args.check_only:
        print(f"\n{failures} failure(s)")
        return 1 if failures else 0

    print("\nExecuting runnable notebooks:")
    for relpath in EXECUTABLE_NOTEBOOKS:
        path = root / relpath
        if not path.exists():
            continue  # already counted above
        ok, msg = execute(path, timeout=args.timeout)
        print(msg)
        if not ok:
            failures += 1

    print(f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
