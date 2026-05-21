"""Root-mean-square-deviation metrics for protein structures.

Two flavors:
  - :func:`rmsd_raw` — coordinates only, no alignment. Use when you've
    already aligned the structures and just want the distance.
  - :func:`rmsd` — accepts :class:`Protein` objects, picks compatible
    atoms automatically (default: alpha-carbons), and superposes before
    measuring. This is what most users want.

For chain-pair or per-residue RMSD see :func:`rmsd_per_residue`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from molforge.structure.superposition import superpose

if TYPE_CHECKING:
    from molforge.core import Protein


def rmsd_raw(
    a: NDArray[np.floating],
    b: NDArray[np.floating],
) -> float:
    """RMSD between two equal-length coordinate sets, no alignment.

    Args:
        a: First ``(n, 3)`` coordinate array.
        b: Second ``(n, 3)`` coordinate array, same shape as ``a``.

    Returns:
        Root-mean-square deviation in the input units (Å for biology).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    diff = a - b
    return float(np.sqrt((diff * diff).sum() / a.shape[0]))


# Atom selectors for the common subsets used in RMSD measurements.
_BACKBONE_NAMES = ("N", "CA", "C")
_BACKBONE_PLUS_O_NAMES = ("N", "CA", "C", "O")

AtomSubset = Literal["ca", "backbone", "backbone_o", "all_heavy", "all"]


def _select_atoms(protein: Protein, subset: AtomSubset) -> NDArray[np.int_]:
    """Return atom indices for the requested subset."""
    arr = protein.atom_array
    if subset == "ca":
        return np.where(arr.atom_name == "CA")[0]
    if subset == "backbone":
        return np.where(np.isin(arr.atom_name, _BACKBONE_NAMES))[0]
    if subset == "backbone_o":
        return np.where(np.isin(arr.atom_name, _BACKBONE_PLUS_O_NAMES))[0]
    if subset == "all_heavy":
        return np.where(arr.element != "H")[0]
    if subset == "all":
        return np.arange(len(arr))
    raise ValueError(
        f"unknown atom subset {subset!r}; "
        "expected 'ca' | 'backbone' | 'backbone_o' | 'all_heavy' | 'all'"
    )


def rmsd(
    mobile: Protein,
    reference: Protein,
    *,
    subset: AtomSubset = "ca",
    align: bool = True,
) -> float:
    """RMSD between two structures, optionally with optimal superposition.

    Args:
        mobile, reference: The two structures. They must contain the same
            number of atoms in the requested subset, in matching order
            (typically same chain, same residue range, both with their
            CAs in residue-sequence order).
        subset: Which atoms to compare:

            - ``"ca"`` (default): alpha-carbons only. Standard for
              evaluating folding models.
            - ``"backbone"``: N, CA, C.
            - ``"backbone_o"``: N, CA, C, O (= mainchain).
            - ``"all_heavy"``: every non-hydrogen atom.
            - ``"all"``: every atom.
        align: If True (default), superpose ``mobile`` onto ``reference``
            and return the post-superposition RMSD. If False, compute
            the raw RMSD on the input coordinates.

    Returns:
        RMSD in angstroms.

    Raises:
        ValueError: If the atom counts of the two subsets don't match.
    """
    mi = _select_atoms(mobile, subset)
    ri = _select_atoms(reference, subset)
    if mi.shape[0] != ri.shape[0]:
        raise ValueError(
            f"atom subset {subset!r} has {mi.shape[0]} atoms in mobile but "
            f"{ri.shape[0]} in reference; the structures aren't comparable"
        )
    if mi.shape[0] == 0:
        raise ValueError(f"atom subset {subset!r} selected no atoms")

    m_coords = mobile.atom_array.coords[mi]
    r_coords = reference.atom_array.coords[ri]
    if align:
        return superpose(m_coords, r_coords).rmsd
    return rmsd_raw(m_coords, r_coords)


def rmsd_per_residue(
    mobile: Protein,
    reference: Protein,
    *,
    subset: AtomSubset = "ca",
    align: bool = True,
) -> NDArray[np.float32]:
    """Per-residue RMSD after (optionally) aligning the structures globally.

    Useful for spotting which loops moved between two conformations or
    where a folding model disagrees with experiment.

    Args:
        mobile: First structure (the one that is moved to align with reference).
        reference: Second structure; must have the same residue count as ``mobile``.
        subset: Atom selector for both the global alignment and the
            per-residue comparison.
        align: Whether to superpose first.

    Returns:
        ``(n_residues,)`` float32 array of per-residue RMSDs.
    """
    mi = _select_atoms(mobile, subset)
    ri = _select_atoms(reference, subset)
    if mi.shape[0] != ri.shape[0]:
        raise ValueError(f"atom counts differ between structures: {mi.shape[0]} vs {ri.shape[0]}")
    m_coords = mobile.atom_array.coords[mi]
    r_coords = reference.atom_array.coords[ri]
    if align:
        m_coords = superpose(m_coords, r_coords).mobile_aligned
    # Group by residue_id and chain_id of the mobile structure.
    m_arr = mobile.atom_array
    keys = np.array(
        list(
            zip(
                m_arr.chain_id[mi].tolist(),
                m_arr.residue_id[mi].tolist(),
                m_arr.insertion_code[mi].tolist(),
                strict=True,
            )
        )
    )
    # Unique residues in encounter order.
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    out = np.zeros(unique_keys.shape[0], dtype=np.float32)
    counts = np.zeros(unique_keys.shape[0], dtype=np.int32)
    diff = m_coords - r_coords
    sq = (diff * diff).sum(axis=1)
    for i, inv in enumerate(inverse):
        out[inv] += sq[i]
        counts[inv] += 1
    return np.sqrt(out / counts).astype(np.float32)
