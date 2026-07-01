"""Backbone bond-length validation.

Refinement pins covalent bond lengths to tight, well-known ideals (the
Engh & Huber standard-geometry values). A bond that strays far from its
ideal is a sign of a distorted or badly-built model — a cheap,
geometry-only quality gate that complements clash and Ramachandran
checks.

This module checks the mainchain bonds every residue has —
``N-CA``, ``CA-C``, ``C-O`` and the inter-residue peptide ``C-N`` — plus
``CA-CB`` where a Cβ is present. Each length is compared to its ideal;
the deviation is reported in Å and as a z-score (deviations in units of
the ideal's standard deviation), and anything beyond ``max_z`` (default
4σ, the usual outlier line) is flagged.

The peptide ``C-N`` bond is only checked between residues that are in the
same chain and numbered consecutively, so ordinary chain breaks and gaps
are not mistaken for broken bonds. A genuine break that *is* modelled
with consecutive numbering will (correctly) surface as a very long bond.

Example::

    from molforge.structure import check_bond_lengths, bond_length_rmsd

    for o in check_bond_lengths(model):
        print(f"{o.bond} at {o.residue_i[2]}{o.residue_i[1]}: "
              f"{o.length:.3f} Å ({o.z_score:+.1f}σ)")
    print("bond-length RMSD:", bond_length_rmsd(model))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from molforge.core import Protein

# Engh & Huber (1991) standard-geometry bond lengths: (ideal Å, sigma Å).
# These are the target values protein refinement restrains towards, so a
# well-built model sits within a couple of sigma of them.
IDEAL_BOND_LENGTHS: dict[str, tuple[float, float]] = {
    "N-CA": (1.458, 0.019),
    "CA-C": (1.525, 0.021),
    "C-O": (1.229, 0.019),
    "CA-CB": (1.530, 0.020),
    "C-N": (1.329, 0.014),  # inter-residue peptide bond
}

#: Intra-residue bonds checked, as ``(atom_i, atom_j, bond_name)``.
_INTRA_BONDS: tuple[tuple[str, str, str], ...] = (
    ("N", "CA", "N-CA"),
    ("CA", "C", "CA-C"),
    ("C", "O", "C-O"),
)

#: Default outlier threshold, in standard deviations.
DEFAULT_MAX_Z: float = 4.0


@dataclass(frozen=True)
class BondLengthOutlier:
    """A bond whose length deviates from ideal by more than ``max_z``.

    Attributes:
        atom_i, atom_j: Global atom indices into ``protein.atom_array``.
        name_i, name_j: Atom names (e.g. ``"N"``, ``"CA"``).
        residue_i, residue_j: ``(chain, residue_id, residue_name)`` for
            each atom (identical for intra-residue bonds).
        bond: Bond key from :data:`IDEAL_BOND_LENGTHS` (e.g. ``"N-CA"``).
        length: Measured length in Å.
        ideal: Engh & Huber ideal length in Å.
        sigma: Ideal standard deviation in Å.
        deviation: ``length - ideal`` in Å (signed).
        z_score: ``deviation / sigma`` (signed).
    """

    atom_i: int
    atom_j: int
    name_i: str
    name_j: str
    residue_i: tuple[str, int, str]
    residue_j: tuple[str, int, str]
    bond: str
    length: float
    ideal: float
    sigma: float
    deviation: float
    z_score: float


def _residue_atoms(
    protein: Protein,
) -> list[tuple[tuple[str, int, str], dict[str, tuple[int, NDArray[np.float64]]]]]:
    """Per residue: a ``(chain, resid, resname)`` label and a map from
    atom name to ``(global_index, coord)``.

    Only protein residues are included; ligands / water are skipped so
    they never masquerade as backbone.
    """
    arr = protein.atom_array
    out: list[tuple[tuple[str, int, str], dict[str, tuple[int, NDArray[np.float64]]]]] = []
    for sl in arr.iter_residue_slices():
        if str(arr.entity_type[sl.start]) != "protein":
            continue
        label = (
            str(arr.chain_id[sl.start]),
            int(arr.residue_id[sl.start]),
            str(arr.residue_name[sl.start]),
        )
        names = arr.atom_name[sl]
        coords = arr.coords[sl]
        atoms: dict[str, tuple[int, NDArray[np.float64]]] = {}
        for name in ("N", "CA", "C", "O", "CB"):
            idx = np.where(names == name)[0]
            if idx.size:
                local = int(idx[0])
                atoms[name] = (sl.start + local, coords[local].astype(np.float64))
        out.append((label, atoms))
    return out


def _iter_bonds(
    protein: Protein,
    *,
    include_cb: bool,
) -> list[tuple[str, int, str, int, str, tuple[str, int, str], tuple[str, int, str], float]]:
    """Yield every checkable bond as
    ``(bond, gi, name_i, gj, name_j, res_i, res_j, length)``.
    """
    residues = _residue_atoms(protein)
    bonds: list[
        tuple[str, int, str, int, str, tuple[str, int, str], tuple[str, int, str], float]
    ] = []
    intra = _INTRA_BONDS + (("CA", "CB", "CA-CB"),) if include_cb else _INTRA_BONDS

    prev_c: tuple[int, NDArray[np.float64]] | None = None
    prev_label: tuple[str, int, str] | None = None
    for label, atoms in residues:
        for a_name, b_name, bond in intra:
            if a_name in atoms and b_name in atoms:
                gi, ci = atoms[a_name]
                gj, cj = atoms[b_name]
                bonds.append(
                    (bond, gi, a_name, gj, b_name, label, label, float(np.linalg.norm(ci - cj)))
                )
        # Peptide bond C(prev) - N(this), only within a chain and between
        # consecutively-numbered residues (so gaps aren't false bonds).
        if prev_c is not None and prev_label is not None and "N" in atoms:
            same_chain = prev_label[0] == label[0]
            consecutive = label[1] == prev_label[1] + 1
            if same_chain and consecutive:
                gi, ci = prev_c
                gj, cj = atoms["N"]
                bonds.append(
                    ("C-N", gi, "C", gj, "N", prev_label, label, float(np.linalg.norm(ci - cj)))
                )
        prev_c = atoms.get("C")
        prev_label = label
    return bonds


def check_bond_lengths(
    protein: Protein,
    *,
    max_z: float = DEFAULT_MAX_Z,
    include_cb: bool = True,
) -> list[BondLengthOutlier]:
    """Backbone bonds whose length is more than ``max_z`` σ from ideal.

    Args:
        protein: Structure to check.
        max_z: Outlier threshold in standard deviations (default 4).
        include_cb: Also check the ``CA-CB`` bond where a Cβ is present.

    Returns:
        Outliers sorted by absolute z-score, worst first.
    """
    outliers: list[BondLengthOutlier] = []
    for bond, gi, name_i, gj, name_j, res_i, res_j, length in _iter_bonds(
        protein, include_cb=include_cb
    ):
        ideal, sigma = IDEAL_BOND_LENGTHS[bond]
        deviation = length - ideal
        z = deviation / sigma
        if abs(z) > max_z:
            outliers.append(
                BondLengthOutlier(
                    atom_i=gi,
                    atom_j=gj,
                    name_i=name_i,
                    name_j=name_j,
                    residue_i=res_i,
                    residue_j=res_j,
                    bond=bond,
                    length=length,
                    ideal=ideal,
                    sigma=sigma,
                    deviation=deviation,
                    z_score=z,
                )
            )
    outliers.sort(key=lambda o: abs(o.z_score), reverse=True)
    return outliers


def bond_length_rmsd(protein: Protein, *, include_cb: bool = True) -> float:
    """RMS deviation of all checked bonds from their ideal lengths (Å).

    A standard-geometry quality metric: near zero for a well-refined
    model. Returns 0.0 when there are no checkable bonds.
    """
    devs = [
        length - IDEAL_BOND_LENGTHS[bond][0]
        for bond, _gi, _ni, _gj, _nj, _ri, _rj, length in _iter_bonds(
            protein, include_cb=include_cb
        )
    ]
    if not devs:
        return 0.0
    return float(np.sqrt(np.mean(np.square(devs))))


def has_bond_length_outliers(
    protein: Protein,
    *,
    max_z: float = DEFAULT_MAX_Z,
    include_cb: bool = True,
) -> bool:
    """Whether ``protein`` has any backbone bond-length outlier."""
    return bool(check_bond_lengths(protein, max_z=max_z, include_cb=include_cb))
