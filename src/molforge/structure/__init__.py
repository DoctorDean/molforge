"""Structural analysis: superposition, RMSD, contacts, geometry, DSSP, SASA, dihedrals, clashes.

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
    - :func:`sasa` / :func:`sasa_per_residue` / :func:`total_sasa` —
      solvent-accessible surface area (Shrake-Rupley).
    - :func:`phi` / :func:`psi` / :func:`omega` / :func:`phi_psi_omega` /
      :func:`ramachandran` / :func:`dihedral` — backbone dihedral angles.
"""

from __future__ import annotations

from molforge.structure.bond_geometry import (
    DEFAULT_MAX_Z,
    IDEAL_BOND_LENGTHS,
    BondLengthOutlier,
    bond_length_rmsd,
    check_bond_lengths,
    has_bond_length_outliers,
)
from molforge.structure.chirality import (
    DEFAULT_PLANAR_TOLERANCE,
    ChiralityConfig,
    ChiralityResult,
    ca_chirality,
    chirality_outliers,
    classify_chirality,
    has_chirality_outliers,
)
from molforge.structure.clashes import (
    DEFAULT_TOLERANCE,
    DEFAULT_VDW_RADIUS,
    VDW_RADII,
    Clash,
    clash_score,
    find_clashes,
    has_clashes,
)
from molforge.structure.contacts import (
    contact_map,
    distance_map,
    residue_contacts,
)
from molforge.structure.dihedrals import (
    RamachandranCategory,
    RamachandranClass,
    RamachandranResult,
    classify_ramachandran,
    dihedral,
    dihedrals_batch,
    omega,
    phi,
    phi_psi_omega,
    psi,
    ramachandran,
    ramachandran_favored_fraction,
    ramachandran_outliers,
    ramachandran_type,
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
from molforge.structure.sasa import (
    sasa,
    sasa_per_residue,
    total_sasa,
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
    # clashes
    "find_clashes",
    "clash_score",
    "has_clashes",
    "Clash",
    "VDW_RADII",
    "DEFAULT_VDW_RADIUS",
    "DEFAULT_TOLERANCE",
    # bond geometry
    "check_bond_lengths",
    "bond_length_rmsd",
    "has_bond_length_outliers",
    "BondLengthOutlier",
    "IDEAL_BOND_LENGTHS",
    "DEFAULT_MAX_Z",
    # chirality
    "ca_chirality",
    "classify_chirality",
    "chirality_outliers",
    "has_chirality_outliers",
    "ChiralityResult",
    "ChiralityConfig",
    "DEFAULT_PLANAR_TOLERANCE",
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
    # SASA
    "sasa",
    "sasa_per_residue",
    "total_sasa",
    # Dihedrals
    "dihedral",
    "dihedrals_batch",
    "phi",
    "psi",
    "omega",
    "phi_psi_omega",
    "ramachandran",
    # Ramachandran classification
    "ramachandran_type",
    "classify_ramachandran",
    "ramachandran_outliers",
    "ramachandran_favored_fraction",
    "RamachandranResult",
    "RamachandranClass",
    "RamachandranCategory",
]
