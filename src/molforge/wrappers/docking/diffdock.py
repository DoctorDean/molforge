"""DiffDock molecular-docking wrapper.

**Status: stub.** The class is part of the committed public API
surface — :class:`DiffDock` is exported from
:mod:`molforge.wrappers.docking` so the import path is stable — but
the engine itself is not yet implemented. :meth:`dock` raises
:class:`NotImplementedError` with a pointer to the tracking issue.

For working docking today, use :class:`molforge.wrappers.docking.Vina`.

When implemented, DiffDock will follow the same :class:`DockingEngine`
contract as Vina: :meth:`dock` takes a receptor and ligand and returns
a :class:`DockingResult` with poses sorted best-first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from molforge.docking import DockingEngine

if TYPE_CHECKING:
    from molforge.core import Protein
    from molforge.docking import DockingResult


_NOT_IMPLEMENTED_HINT = (
    "The DiffDock engine is not yet implemented. Its API surface is "
    "committed (the import path molforge.wrappers.docking.DiffDock is "
    "stable) but the implementation is planned. For working docking "
    "today, use molforge.wrappers.docking.Vina. Track progress at "
    "https://github.com/DoctorDean/molforge/issues."
)


class DiffDock(DockingEngine):
    """Wrapper around DiffDock (stub — not yet implemented).

    :meth:`dock` raises :class:`NotImplementedError`. The class exists
    so the public API surface (import path, class name, method
    signature) is committed and stable; the implementation will
    follow. Use :class:`Vina` for working docking in the meantime.
    """

    name = "DiffDock"

    def dock(
        self,
        receptor: Protein,
        ligand: object,
        **kwargs: object,
    ) -> DockingResult:
        """Not implemented — see :data:`_NOT_IMPLEMENTED_HINT`."""
        raise NotImplementedError(_NOT_IMPLEMENTED_HINT)
