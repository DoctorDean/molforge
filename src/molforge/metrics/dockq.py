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
# is a pair of residues (one in each chain) with any heavy atoms within
# this distance of each other.
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


_BACKBONE_ATOMS = ("N", "CA", "C", "O")


def _backbone_by_residue(protein: Protein, chain_id: str) -> list[np.ndarray | None]:
    """N, CA, C, O coordinates per protein residue of ``chain_id``.

    Returns one ``(4, 3)`` array per residue (atoms in N, CA, C, O order),
    in the chain's residue order — so the list index is the residue's
    position within the chain. A residue missing any backbone atom yields
    ``None`` rather than shifting every later residue's index, which flat
    position-based ``p * 4`` slicing would do — silently pairing the wrong
    atoms (or indexing out of bounds) in the interface RMSD.
    """
    arr = protein.atom_array
    out: list[np.ndarray | None] = []
    for sl in arr.iter_residue_slices():
        if str(arr.entity_type[sl.start]) != "protein":
            continue
        if str(arr.chain_id[sl.start]) != chain_id:
            continue
        names = arr.atom_name[sl]
        coords = arr.coords[sl]
        bb: list[np.ndarray] = []
        for name in _BACKBONE_ATOMS:
            idx = np.where(names == name)[0]
            if idx.size == 0:
                bb = []
                break
            bb.append(coords[idx[0]])
        out.append(np.asarray(bb, dtype=np.float64) if bb else None)
    return out


def _heavy_atoms_by_residue(protein: Protein, chain_id: str) -> list[np.ndarray]:
    """Heavy-atom coordinates for each protein residue of ``chain_id``.

    Returns one ``(n_heavy, 3)`` array per residue, in the chain's residue
    order — so the list index is the residue's position within the chain.
    Contacts are keyed by residue position rather than atom index, which
    keeps the contact set comparable between model and reference even when
    their per-residue atom counts or ordering differ (a missing side-chain
    atom, different atom naming, etc.).
    """
    arr = protein.atom_array
    residues: list[np.ndarray] = []
    for sl in arr.iter_residue_slices():
        if str(arr.entity_type[sl.start]) != "protein":
            continue
        if str(arr.chain_id[sl.start]) != chain_id:
            continue
        heavy = [
            arr.coords[i]
            for i in range(sl.start, sl.stop)
            if str(arr.element[i]).upper() != "H"
        ]
        residues.append(
            np.asarray(heavy, dtype=np.float64) if heavy else np.empty((0, 3), dtype=np.float64)
        )
    return residues


def _residue_contacts(
    residues_a: list[np.ndarray],
    residues_b: list[np.ndarray],
    cutoff: float = _FNAT_CUTOFF,
) -> set[tuple[int, int]]:
    """Residue-pair contacts ``(pos_a, pos_b)``.

    A pair is a contact — the CAPRI definition — when the two residues
    have any heavy atoms within ``cutoff`` of each other.
    """
    contacts: set[tuple[int, int]] = set()
    for pa, coords_a in enumerate(residues_a):
        if coords_a.shape[0] == 0:
            continue
        for pb, coords_b in enumerate(residues_b):
            if coords_b.shape[0] == 0:
                continue
            diff = coords_a[:, None, :] - coords_b[None, :, :]
            if bool((np.linalg.norm(diff, axis=-1) < cutoff).any()):
                contacts.add((pa, pb))
    return contacts


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

    native = _residue_contacts(
        _heavy_atoms_by_residue(reference, chain_a),
        _heavy_atoms_by_residue(reference, chain_b),
        cutoff,
    )
    if not native:
        return 0.0
    predicted = _residue_contacts(
        _heavy_atoms_by_residue(model, chain_a),
        _heavy_atoms_by_residue(model, chain_b),
        cutoff,
    )
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
    if chain_a is None or chain_b is None:
        common = sorted(
            set(_backbone_atom_coords_per_chain(model))
            & set(_backbone_atom_coords_per_chain(reference))
        )
        if len(common) < 2:
            raise ValueError("need at least 2 protein chains")
        chain_a, chain_b = common[0], common[1]
    iface_a, iface_b = _interface_residues(reference, chain_a, chain_b)
    if not iface_a or not iface_b:
        return 0.0

    # Select each interface residue's N/CA/C/O by residue position, so a
    # residue missing a backbone atom can't shift the indexing (flat
    # position*4 slicing silently paired the wrong atoms, or indexed out of
    # bounds). Interface residues without a complete backbone in *both*
    # structures are skipped.
    def _matched(chain_id: str, positions: list[int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        m_res = _backbone_by_residue(model, chain_id)
        r_res = _backbone_by_residue(reference, chain_id)
        m_sel: list[np.ndarray] = []
        r_sel: list[np.ndarray] = []
        for p in positions:
            if p < len(m_res) and p < len(r_res):
                m_bb, r_bb = m_res[p], r_res[p]
                if m_bb is not None and r_bb is not None:
                    m_sel.append(m_bb)
                    r_sel.append(r_bb)
        return m_sel, r_sel

    m_a, r_a = _matched(chain_a, iface_a)
    m_b, r_b = _matched(chain_b, iface_b)
    m_blocks = m_a + m_b
    r_blocks = r_a + r_b
    if not m_blocks:
        return 0.0
    m_interface = np.concatenate(m_blocks)
    r_interface = np.concatenate(r_blocks)
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
