"""DiffDock wrapper."""

from __future__ import annotations

from biocore.docking import DockingEngine, DockingResult


class DiffDock(DockingEngine):
    """Wrapper around DiffDock.

    TODO: implement receptor/ligand prep, invocation, and pose parsing.
    """

    def __init__(self, **kwargs: object) -> None: ...

    def dock(self, receptor: object, ligand: object) -> DockingResult:
        raise NotImplementedError
