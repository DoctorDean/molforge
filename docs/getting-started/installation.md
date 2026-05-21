# Installation

`molforge` supports Python 3.10, 3.11, and 3.12 on Linux, macOS, and
Windows. Install from PyPI:

```bash
pip install molforge
```

That gives you the core data model
([`Protein`](../reference/core.md), [`AtomArray`](../reference/core.md))
and the minimal dependency set (`numpy`, `typing-extensions`). Most
real workflows want at least one of the extras below.

## Extras

`molforge` ships heavy scientific dependencies as opt-in extras so the
base install stays small and import-fast.

| Extra        | What it pulls in                            | When you need it                                          |
| ------------ | ------------------------------------------- | --------------------------------------------------------- |
| `structure`  | biopython, biotite, scipy                   | RMSD, SASA, DSSP, contacts, superposition.                |
| `sequence`   | biopython                                   | Alignments using BLOSUM/PAM matrices.                     |
| `io`         | biopython, biotite                          | PDB / mmCIF parsing beyond the built-in fallback.         |
| `md`         | mdtraj, openmm (non-Windows)                | Molecular dynamics via the OpenMM wrapper.                |
| `docking`    | rdkit                                       | Small-molecule prep for AutoDock Vina.                    |
| `ml`         | torch, transformers                         | ESM-2 embeddings, GNN featurization.                      |
| `all`        | everything above                            | One-line install when you don't care about footprint.     |

Install combinations with bracket syntax:

```bash
pip install "molforge[structure]"
pip install "molforge[structure,ml]"
pip install "molforge[all]"
```

## Development install

If you're working on `molforge` itself, clone the repository and
install in editable mode with dev tooling:

```bash
git clone https://github.com/DoctorDean/molforge.git
cd molforge
pip install -e ".[dev,all]"
pre-commit install
```

The `dev` extra includes pytest, ruff, mypy, pre-commit, and the
build/publish tooling. With `[dev,all]` installed, run the test suite:

```bash
pytest                          # full suite (~664 tests)
pytest -m "not slow"            # skip slow tests
pytest -n auto                  # parallel (uses pytest-xdist)
```

## Optional engine binaries

Some wrappers shell out to external binaries that aren't on PyPI. You
only need these if you're using the corresponding wrapper:

- **AutoDock Vina** — install via conda (`conda install -c bioconda vina`)
  or [from source](https://vina.scripps.edu/downloads/).
- **ESMFold / AlphaFold / RFdiffusion / ProteinMPNN** — see each
  wrapper's docstring for setup instructions. GPU recommended.
- **OpenMM** — the `md` extra installs the Python bindings; no extra
  binary needed.

## Verify your install

```python
import molforge
print(molforge.__version__)

from molforge.core import Protein, AtomArray
from molforge.io import load
print("ok")
```

If that prints a version and `ok`, you're set. Head to the
[Quickstart](quickstart.md).
