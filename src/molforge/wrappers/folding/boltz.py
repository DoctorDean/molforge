"""Boltz wrapper."""

from __future__ import annotations

from molforge.wrappers.folding._base import FoldingEngine


class Boltz(FoldingEngine):
    """Wrapper around Boltz.

    TODO: implement model loading, prediction, and conversion of the engine's
    native output into a :class:`molforge.core.Protein`.
    """

    def __init__(self, **kwargs: object) -> None:
        # TODO: handle model weights / device selection / config.
        ...

    def predict(self, sequence: str, **kwargs: object) -> object:
        raise NotImplementedError
