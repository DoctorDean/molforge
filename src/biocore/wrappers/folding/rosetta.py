"""PyRosetta wrapper (sequence-design / structure-prediction utilities).

Note: PyRosetta has its own license and is not pip-installable from PyPI.
This wrapper is provided as an optional integration; users are responsible
for installing PyRosetta separately.
"""

from __future__ import annotations

from biocore.wrappers.folding._base import FoldingEngine


class Rosetta(FoldingEngine):
    """Wrapper around PyRosetta. TODO: implement."""

    def __init__(self, **kwargs: object) -> None: ...

    def predict(self, sequence: str, **kwargs: object) -> object:
        raise NotImplementedError
