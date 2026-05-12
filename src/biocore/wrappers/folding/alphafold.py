"""AlphaFold / ColabFold wrapper."""

from __future__ import annotations

from biocore.wrappers.folding._base import FoldingEngine


class AlphaFold(FoldingEngine):
    """Wrapper around AlphaFold / ColabFold.

    TODO: implement model loading, prediction, and conversion of the engine's
    native output into a :class:`biocore.core.Protein`.
    """

    def __init__(self, **kwargs: object) -> None:
        # TODO: handle model weights / device selection / config.
        ...

    def predict(self, sequence: str, **kwargs: object) -> object:
        raise NotImplementedError
