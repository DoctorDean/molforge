"""GROMACS wrapper."""

from __future__ import annotations

from molforge.wrappers.md._base import MDEngine


class GROMACS(MDEngine):
    """Wrapper around GROMACS.

    TODO: implement system preparation, force-field selection, and
    integration with :class:`molforge.md.Trajectory`.
    """

    def __init__(self, **kwargs: object) -> None: ...

    def simulate(self, protein: object, *, steps: int, **kwargs: object) -> object:
        raise NotImplementedError
