"""Structural analysis: superposition, RMSD, contacts, geometry, DSSP.

Workhorses for analyzing the geometric properties of protein
structures and comparing them.

Common entry points:
    - :func:`rmsd` — RMSD between two structures (with optional
      superposition).
    - :func:`superpose` — Kabsch / Umeyama optimal rigid-body alignment.
    - :func:`contact_map` / :func:`distance_map` — residue-residue
      contact and distance matrices.
    - :func:`residue_contacts` — all-atom contacts as a sorted list.
    - :func:`radius_of_gyration`, :func:`centroid`, :func:`center_of_mass` —
      bulk geometric properties.
    - :func:`translate`, :func:`rotate`, :func:`center_at_origin` —
      in-place coordinate transforms.
    - :func:`dssp` / :func:`dssp_3state` — Kabsch-Sander secondary-
      structure assignment (8-state and 3-state).

Not yet implemented (stubs raising NotImplementedError):
    - SASA (solvent-accessible surface area) — coming next
    - Backbone dihedrals (phi, psi, omega) — coming next
"""

from __future__ import annotations

from molforge.structure.contacts import (
    contact_map,
    distance_map,
    residue_contacts,
)
from molforge.structure.dssp import dssp, dssp_3state
from molforge.structure.geometry import (
    bounding_box,
    center_at_origin,
    center_of_mass,
    centroid,
    radius_of_gyration,
    rotate,
    translate,
)
from molforge.structure.rmsd import (
    rmsd,
    rmsd_per_residue,
    rmsd_raw,
)
from molforge.structure.superposition import (
    SuperpositionResult,
    kabsch_rmsd,
    superpose,
)

__all__ = [  # noqa: RUF022 — grouped by concern
    # Superposition / RMSD
    "superpose",
    "kabsch_rmsd",
    "SuperpositionResult",
    "rmsd",
    "rmsd_raw",
    "rmsd_per_residue",
    # Contacts / distance
    "contact_map",
    "distance_map",
    "residue_contacts",
    # Geometry
    "centroid",
    "center_of_mass",
    "radius_of_gyration",
    "bounding_box",
    "translate",
    "rotate",
    "center_at_origin",
    # Secondary structure
    "dssp",
    "dssp_3state",
    # Not yet implemented
    "sasa",
]


def sasa(protein: object) -> object:
    """Solvent-accessible surface area. TODO: implement (Shrake-Rupley)."""
    raise NotImplementedError(
        "SASA computation is planned; track at https://github.com/DoctorDean/molforge/issues."
    )
