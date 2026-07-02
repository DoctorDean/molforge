"""DockQ score for protein-protein complex prediction quality.

Reference: Basu, S. & Wallner, B. (2016) "DockQ: A Quality Measure for
Protein-Protein Docking Models." *PLoS ONE* 11: e0161879.

DockQ is the standard quality metric used in CAPRI (the docking
benchmark series). It collapses three classic CAPRI quality measures
into a single 0-1 score:

  - **Fnat** — fraction of native interface residue contacts that are
    recovered in the model.
  - **iRMS** — RMSD of the interface backbone (interface residues
    only, superposed against the native interface).
  - **LRMS** — ligand RMSD (smaller chain backbone, superposed via
    the larger chain).

The DockQ formula combines these with a sigmoid weighting (see paper).
Interpretation (CAPRI scale):

  - **< 0.23** — incorrect
  - **0.23 - 0.49** — acceptable
  - **0.49 - 0.80** — medium
  - **> 0.80** — high

This implementation works for 2-chain complexes. For multi-chain
complexes, compute DockQ for each interface separately.

For external validation, the reference Python implementation lives at
https://github.com/bjornwallner/DockQ. molforge's version is
NumPy-only and produces comparable values for standard cases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from molforge.structure.superposition import superpose

if TYPE_CHECKING:
    from molforge.core import Protein


# Fnat default cutoff (Å). Standard CAPRI definition: a "native contact"
# is any pair of heavy atoms in different chains within this distance.
_FNAT_CUTOFF = 5.0
# Interface-residue cutoff used to define the interface for iRMS.
_INTERFACE_CUTOFF = 10.0
# Sigmoid parameters from the DockQ paper.
_LRMS_SCALE = 8.5
_IRMS_SCALE = 1.5


def _heavy_atom_coords_per_chain(
    protein: Protein,
) -> dict[str, np.ndarray]:
    """Group heavy-atom coordinates by chain ID."""
    arr = protein.atom_array
    out: dict[str, list[np.ndarray]] = {}
    for i in range(len(arr)):
        if str(arr.element[i]).upper() == "H":
            continue
        if str(arr.entity_type[i]) != "protein":
            continue
        chain = str(arr.chain_id[i])
        out.setdefault(chain, []).append(arr.coords[i])
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def _backbone_atom_coords_per_chain(
    protein: Protein,
    backbone_names: tuple[str, ...] = ("N", "CA", "C", "O"),
) -> dict[str, np.ndarray]:
    """Group backbone-atom coordinates by chain ID, preserving order."""
    arr = protein.atom_array
    out: dict[str, list[np.ndarray]] = {}
    for i in range(len(arr)):
        if str(arr.entity_type[i]) != "protein":
            continue
        if str(arr.atom_name[i]) not in backbone_names:
            continue
        chain = str(arr.chain_id[i])
        out.setdefault(chain, []).append(arr.coords[i])
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def _native_contacts(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    cutoff: float = _FNAT_CUTOFF,
) -> set[tuple[int, int]]:
    """Pairs ``(i, j)`` of inter-chain heavy atoms within ``cutoff``."""
    # Vectorized pairwise distance, returning the set of close pairs.
    diff = coords_a[:, None, :] - coords_b[None, :, :]
    d = np.linalg.norm(diff, axis=-1)
    sources, targets = np.where(d < cutoff)
    return set(zip(sources.tolist(), targets.tolist(), strict=True))


def fnat(
    model: Protein,
    reference: Protein,
    *,
    chain_a: str | None = None,
    chain_b: str | None = None,
    cutoff: float = _FNAT_CUTOFF,
) -> float:
    """Fraction of native interface contacts recovered in the model.

    Args:
        model: Predicted complex.
        reference: Native complex.
        chain_a: Which chain to compare on the receptor side. If
            ``None``, uses the first protein chain shared between
            ``model`` and ``reference``. Chain IDs must match
            between model and reference.
        chain_b: Which chain to compare on the partner side. See
            ``chain_a`` for default behavior.
        cutoff: Heavy-atom distance defining a contact (default 5 Å).

    Returns:
        ``fnat`` in ``[0, 1]``. 1.0 = every native contact recovered.
    """
    m_chains = _heavy_atom_coords_per_chain(model)
    r_chains = _heavy_atom_coords_per_chain(reference)
    if chain_a is None or chain_b is None:
        common = sorted(set(m_chains) & set(r_chains))
        if len(common) < 2:
            raise ValueError(f"need at least 2 protein chains in both structures; got {common}")
        chain_a, chain_b = common[0], common[1]

    if chain_a not in r_chains or chain_b not in r_chains:
        raise ValueError(f"reference is missing chain {chain_a!r} or {chain_b!r}")
    if chain_a not in m_chains or chain_b not in m_chains:
        raise ValueError(f"model is missing chain {chain_a!r} or {chain_b!r}")

    native = _native_contacts(r_chains[chain_a], r_chains[chain_b], cutoff)
    if not native:
        return 0.0
    predicted = _native_contacts(m_chains[chain_a], m_chains[chain_b], cutoff)
    return len(native & predicted) / len(native)


def _interface_residues(
    protein: Protein,
    chain_a: str,
    chain_b: str,
    cutoff: float = _INTERFACE_CUTOFF,
) -> tuple[list[int], list[int]]:
    """Identify interface residues in each chain.

    Returns ``(indices_in_a, indices_in_b)`` — residue indices (in CA-
    order within each chain) of residues that have any heavy atom within
    ``cutoff`` Å of any heavy atom in the other chain.
    """
    arr = protein.atom_array
    # Group atoms by (chain, residue-position-in-chain). Use CA-order
    # within each chain as the per-chain residue index.
    chain_residues: dict[str, list[tuple[int, int]]] = {}
    for slot, sl in enumerate(arr.iter_residue_slices()):
        chain = str(arr.chain_id[sl.start])
        if str(arr.entity_type[sl.start]) != "protein":
            continue
        chain_residues.setdefault(chain, []).append((slot, len(chain_residues.get(chain, []))))

    # Heavy-atom positions per (chain, residue-position-in-chain).
    per_residue_coords: dict[tuple[str, int], np.ndarray] = {}
    chain_position: dict[str, int] = {}
    for sl in arr.iter_residue_slices():
        chain = str(arr.chain_id[sl.start])
        if str(arr.entity_type[sl.start]) != "protein":
            continue
        pos = chain_position.get(chain, 0)
        chain_position[chain] = pos + 1
        heavy = [
            arr.coords[i] for i in range(sl.start, sl.stop) if str(arr.element[i]).upper() != "H"
        ]
        if heavy:
            per_residue_coords[(chain, pos)] = np.asarray(heavy, dtype=np.float64)

    a_positions = [p for (c, p) in per_residue_coords if c == chain_a]
    b_positions = [p for (c, p) in per_residue_coords if c == chain_b]

    interface_a, interface_b = set(), set()
    for pa in a_positions:
        coords_a = per_residue_coords[(chain_a, pa)]
        for pb in b_positions:
            coords_b = per_residue_coords[(chain_b, pb)]
            diff = coords_a[:, None, :] - coords_b[None, :, :]
            d = np.linalg.norm(diff, axis=-1)
            if (d < cutoff).any():
                interface_a.add(pa)
                interface_b.add(pb)
    return sorted(interface_a), sorted(interface_b)


def irms(
    model: Protein,
    reference: Protein,
    *,
    chain_a: str | None = None,
    chain_b: str | None = None,
) -> float:
    """Interface RMSD — backbone RMSD over the interface residues only."""
    m_bb = _backbone_atom_coords_per_chain(model)
    r_bb = _backbone_atom_coords_per_chain(reference)
    if chain_a is None or chain_b is None:
        common = sorted(set(m_bb) & set(r_bb))
        if len(common) < 2:
            raise ValueError("need at least 2 protein chains")
        chain_a, chain_b = common[0], common[1]
    iface_a, iface_b = _interface_residues(reference, chain_a, chain_b)
    if not iface_a or not iface_b:
        return 0.0

    # Take the 4 backbone atoms per interface residue. We assume equal
    # backbone-atom counts per residue between model and reference (true
    # if both have full N/CA/C/O sets); otherwise this fails loudly.
    # Build indexing: each interface residue contributes its 4 backbone atoms.
    def _interface_bb(chain_bb: np.ndarray, positions: list[int]) -> np.ndarray:
        # backbone-atoms-per-residue: 4 (N, CA, C, O).
        sel: list[int] = []
        for p in positions:
            sel.extend(range(p * 4, p * 4 + 4))
        return chain_bb[sel]

    m_interface = np.concatenate(
        [_interface_bb(m_bb[chain_a], iface_a), _interface_bb(m_bb[chain_b], iface_b)]
    )
    r_interface = np.concatenate(
        [_interface_bb(r_bb[chain_a], iface_a), _interface_bb(r_bb[chain_b], iface_b)]
    )
    if m_interface.shape != r_interface.shape:
        raise ValueError(
            f"interface backbone shape mismatch: {m_interface.shape} vs {r_interface.shape}"
        )
    return superpose(m_interface, r_interface).rmsd


def lrms(
    model: Protein,
    reference: Protein,
    *,
    chain_a: str | None = None,
    chain_b: str | None = None,
) -> float:
    """Ligand RMSD — superpose the *receptor* (larger chain) and measure
    the *ligand* (smaller chain) RMSD."""
    m_bb = _backbone_atom_coords_per_chain(model)
    r_bb = _backbone_atom_coords_per_chain(reference)
    if chain_a is None or chain_b is None:
        common = sorted(set(m_bb) & set(r_bb))
        if len(common) < 2:
            raise ValueError("need at least 2 protein chains")
        chain_a, chain_b = common[0], common[1]
    # Receptor = larger chain by atom count
    if r_bb[chain_a].shape[0] >= r_bb[chain_b].shape[0]:
        receptor, ligand = chain_a, chain_b
    else:
        receptor, ligand = chain_b, chain_a
    # Superpose on receptor backbone
    sup = superpose(m_bb[receptor], r_bb[receptor])
    # Apply that transform to the ligand
    aligned_ligand = (sup.rotation @ m_bb[ligand].T).T + sup.translation
    diff = aligned_ligand - r_bb[ligand]
    return float(np.sqrt((diff * diff).sum(axis=1).mean()))


def dockq(
    model: Protein,
    reference: Protein,
    *,
    chain_a: str | None = None,
    chain_b: str | None = None,
) -> dict[str, float]:
    """DockQ — single-number quality for a docked complex.

    Args:
        model: predicted complex.
        reference: native complex (same chain IDs).
        chain_a: first chain to compare. Defaults to the first common
            chain.
        chain_b: second chain to compare. Defaults to the second common
            chain.

    Returns:
        Dict with keys ``"DockQ"`` (the combined 0-1 score), ``"fnat"``,
        ``"iRMS"``, ``"LRMS"``.
    """
    fnat_score = fnat(model, reference, chain_a=chain_a, chain_b=chain_b)
    irms_score = irms(model, reference, chain_a=chain_a, chain_b=chain_b)
    lrms_score = lrms(model, reference, chain_a=chain_a, chain_b=chain_b)

    # DockQ paper's combining formula:
    #   DockQ = (Fnat + 1/(1 + (LRMS/8.5)^2) + 1/(1 + (iRMS/1.5)^2)) / 3
    score = (
        fnat_score
        + 1.0 / (1.0 + (lrms_score / _LRMS_SCALE) ** 2)
        + 1.0 / (1.0 + (irms_score / _IRMS_SCALE) ** 2)
    ) / 3.0
    return {
        "DockQ": float(score),
        "fnat": float(fnat_score),
        "iRMS": float(irms_score),
        "LRMS": float(lrms_score),
    }
