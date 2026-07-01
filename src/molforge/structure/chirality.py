"""Cα chirality checks (L vs D amino acids).

Every amino acid except glycine is chiral at its Cα. Ribosomal proteins
are built exclusively from **L**-amino acids, so a **D** centre is nearly
always a modelling error (a flipped residue, a mirror-image build) —
unless you are deliberately working with D-peptides.

Chirality is read from the four Cα substituents. Hydrogens are usually
absent from models, so the handedness is taken from the heavy atoms N,
C (carbonyl) and Cβ via the signed volume::

    V = (N - CA) · ((C - CA) × (CB - CA))

By the CIP priorities at Cα (N > carbonyl-C > Cβ > H), an L-amino acid
is the S enantiomer, which corresponds to ``V > 0``; a D-amino acid has
``V < 0``. (Cysteine is labelled R rather than S because its sulfur
outranks the carbonyl, but its *geometry* is the same as the other
L-residues, so the same ``V > 0`` test applies — this is a handedness
check, not a CIP R/S assignment.) A near-zero volume means the four
atoms are almost coplanar — a degenerate, unphysical Cα — and is
reported as ``Planar``.

Glycine (no Cβ) and any residue missing N, CA, C or CB is skipped.

Example::

    from molforge.structure import chirality_outliers

    for r in chirality_outliers(model):
        print(f"{r.residue[2]}{r.residue[1]} is {r.configuration}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from molforge.core import Protein

ChiralityConfig = Literal["L", "D", "Planar"]

#: |V| (Å³) at or below which a Cα is treated as degenerate (``Planar``).
#: Standard tetrahedral centres give |V| ≈ 2.3 Å³, so this only catches
#: genuinely near-coplanar atoms.
DEFAULT_PLANAR_TOLERANCE: float = 0.1


@dataclass(frozen=True)
class ChiralityResult:
    """Cα handedness of one residue.

    Attributes:
        residue: ``(chain_id, residue_id, residue_name)``.
        ca_index: Global atom index of the Cα.
        configuration: ``L`` (natural), ``D`` (inverted), or ``Planar``
            (degenerate / near-coplanar Cα).
        volume: Signed tetrahedral volume in Å³
            (``(N-CA)·((C-CA)×(CB-CA))``); positive for L.
    """

    residue: tuple[str, int, str]
    ca_index: int
    configuration: ChiralityConfig
    volume: float


def ca_chirality(
    n: NDArray[np.float64],
    ca: NDArray[np.float64],
    c: NDArray[np.float64],
    cb: NDArray[np.float64],
    *,
    planar_tolerance: float = DEFAULT_PLANAR_TOLERANCE,
) -> ChiralityConfig:
    """Classify a Cα centre as ``L`` / ``D`` / ``Planar``.

    Args:
        n, ca, c, cb: Cartesian coordinates of the backbone N, Cα,
            carbonyl C and Cβ.
        planar_tolerance: |signed volume| (Å³) at or below which the
            centre is reported as ``Planar``.

    Returns:
        The chirality label.
    """
    volume = float(np.dot(n - ca, np.cross(c - ca, cb - ca)))
    if abs(volume) <= planar_tolerance:
        return "Planar"
    return "L" if volume > 0 else "D"


def _ca_chirality_volume(
    n: NDArray[np.float64],
    ca: NDArray[np.float64],
    c: NDArray[np.float64],
    cb: NDArray[np.float64],
) -> float:
    return float(np.dot(n - ca, np.cross(c - ca, cb - ca)))


def classify_chirality(
    protein: Protein,
    *,
    planar_tolerance: float = DEFAULT_PLANAR_TOLERANCE,
) -> list[ChiralityResult]:
    """Classify the Cα chirality of every eligible residue.

    Residues without a full N / CA / C / CB set — glycine, or any
    residue missing one of those atoms — are skipped (glycine is
    achiral). Only protein residues are considered.

    Args:
        protein: Structure to analyze.
        planar_tolerance: See :func:`ca_chirality`.

    Returns:
        One :class:`ChiralityResult` per eligible residue, in
        chain/sequence order.
    """
    arr = protein.atom_array
    results: list[ChiralityResult] = []
    for sl in arr.iter_residue_slices():
        if str(arr.entity_type[sl.start]) != "protein":
            continue
        names = arr.atom_name[sl]
        coords = arr.coords[sl]
        picked: dict[str, tuple[int, NDArray[np.float64]]] = {}
        for name in ("N", "CA", "C", "CB"):
            idx = np.where(names == name)[0]
            if idx.size:
                local = int(idx[0])
                picked[name] = (local, coords[local].astype(np.float64))
        if len(picked) < 4:
            continue
        n_c = picked["N"][1]
        ca_local, ca_c = picked["CA"]
        c_c = picked["C"][1]
        cb_c = picked["CB"][1]
        volume = _ca_chirality_volume(n_c, ca_c, c_c, cb_c)
        if abs(volume) <= planar_tolerance:
            config: ChiralityConfig = "Planar"
        else:
            config = "L" if volume > 0 else "D"
        results.append(
            ChiralityResult(
                residue=(
                    str(arr.chain_id[sl.start]),
                    int(arr.residue_id[sl.start]),
                    str(arr.residue_name[sl.start]),
                ),
                ca_index=sl.start + ca_local,
                configuration=config,
                volume=volume,
            )
        )
    return results


def chirality_outliers(
    protein: Protein,
    *,
    planar_tolerance: float = DEFAULT_PLANAR_TOLERANCE,
) -> list[ChiralityResult]:
    """Residues whose Cα is not the natural ``L`` configuration.

    Returns the ``D`` and ``Planar`` residues (everything that is not a
    clean L centre).
    """
    return [
        r
        for r in classify_chirality(protein, planar_tolerance=planar_tolerance)
        if r.configuration != "L"
    ]


def has_chirality_outliers(
    protein: Protein,
    *,
    planar_tolerance: float = DEFAULT_PLANAR_TOLERANCE,
) -> bool:
    """Whether ``protein`` has any non-``L`` Cα centre."""
    return bool(chirality_outliers(protein, planar_tolerance=planar_tolerance))
