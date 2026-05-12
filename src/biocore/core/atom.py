"""Atom — the leaf node of the structural hierarchy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from biocore.core.residue import Residue


@dataclass
class Atom:
    """A single atom within a residue.

    Attributes:
        name: PDB atom name (e.g. ``"CA"``, ``"N"``, ``"OD1"``).
        element: Element symbol (e.g. ``"C"``, ``"N"``, ``"O"``).
        coord: 3-D Cartesian coordinate in angstroms, shape ``(3,)``.
        serial: Atom serial number as it appears in the source file.
        b_factor: Temperature factor (B-factor), if present.
        occupancy: Occupancy, if present.
        charge: Formal charge, if known.
        parent: Backreference to the containing :class:`Residue`.
    """

    name: str
    element: str
    coord: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    serial: int = 0
    b_factor: float = 0.0
    occupancy: float = 1.0
    charge: float = 0.0
    parent: Residue | None = None

    def __post_init__(self) -> None:
        # TODO: validate element symbol against a periodic-table table.
        # TODO: enforce coord shape == (3,) and dtype float32.
        ...

    @property
    def is_backbone(self) -> bool:
        """Return ``True`` if this atom is a standard backbone atom."""
        raise NotImplementedError
