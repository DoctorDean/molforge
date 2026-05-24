"""GROMACS MD-engine wrapper.

**Status: stub.** The class is part of the committed public API
surface — :class:`GROMACS` is exported from :mod:`molforge.wrappers.md`
so the import path is stable — but the engine itself is not yet
implemented. Every method raises :class:`NotImplementedError` with a
pointer to the tracking issue.

For working MD today, use :class:`molforge.wrappers.md.OpenMM`.

When implemented, GROMACS will follow the same :class:`MDEngine`
contract as OpenMM: :meth:`prepare` builds a :class:`Simulation`,
:meth:`minimize` energy-minimizes it, and :meth:`run` integrates and
returns a :class:`Trajectory`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from molforge.wrappers.md._base import MDEngine

if TYPE_CHECKING:
    from molforge.core import Protein
    from molforge.md import Simulation, Trajectory


_NOT_IMPLEMENTED_HINT = (
    "The GROMACS engine is not yet implemented. Its API surface is "
    "committed (the import path molforge.wrappers.md.GROMACS is stable) "
    "but the implementation is planned. For working MD today, use "
    "molforge.wrappers.md.OpenMM. Track progress at "
    "https://github.com/DoctorDean/molforge/issues."
)


class GROMACS(MDEngine):
    """Wrapper around GROMACS (stub — not yet implemented).

    All methods raise :class:`NotImplementedError`. The class exists so
    the public API surface (import path, class name, method names) is
    committed and stable; the implementation will follow. Use
    :class:`OpenMM` for working MD in the meantime.
    """

    name = "GROMACS"

    def prepare(
        self,
        protein: Protein,
        *,
        force_field: str,
        **kwargs: object,
    ) -> Simulation:
        """Not implemented — see :data:`_NOT_IMPLEMENTED_HINT`."""
        raise NotImplementedError(_NOT_IMPLEMENTED_HINT)

    def minimize(
        self,
        simulation: Simulation,
        *,
        max_iterations: int = 1000,
        tolerance: float = 10.0,
        **kwargs: object,
    ) -> Simulation:
        """Not implemented — see :data:`_NOT_IMPLEMENTED_HINT`."""
        raise NotImplementedError(_NOT_IMPLEMENTED_HINT)

    def run(
        self,
        simulation: Simulation,
        *,
        n_steps: int,
        save_every: int = 1,
        **kwargs: object,
    ) -> Trajectory:
        """Not implemented — see :data:`_NOT_IMPLEMENTED_HINT`."""
        raise NotImplementedError(_NOT_IMPLEMENTED_HINT)
