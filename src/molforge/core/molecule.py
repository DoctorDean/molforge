"""A first-class small-molecule type, backed by RDKit."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from molforge.core import _rdkit
from molforge.core.atom_array import AtomArray

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["Molecule"]


def _element_atom_names(elements: list[str]) -> Any:
    """Per-element atom names — ``C1``, ``C2``, ``N1``, ... (PDB ligand style)."""
    counts: dict[str, int] = {}
    names: list[str] = []
    for element in elements:
        counts[element] = counts.get(element, 0) + 1
        names.append(f"{element}{counts[element]}")
    return np.array(names, dtype="U4")


class Molecule:
    """A small molecule — a ligand, cofactor, or any organic compound.

    Where :class:`~molforge.core.Protein` (and the flat
    :class:`~molforge.core.AtomArray`) captures coordinates and atom names,
    a molecule's *value* is its chemistry: bonds and their orders, formal
    charges, aromaticity, and stereochemistry. Those are exactly what the
    coordinate-only path drops, so ``Molecule`` wraps an RDKit ``Mol`` and
    lets molforge reason about ligands as chemistry rather than as bond-less
    point clouds.

    RDKit is a lazy dependency: constructing a molecule that needs it (e.g.
    :meth:`from_smiles`) or reading a chemistry property raises
    :class:`~molforge.core._rdkit.RDKitNotInstalledError` if RDKit is
    absent, but importing :mod:`molforge.core` never pulls it in.

    The wrapped ``Mol`` is shared, not copied — :meth:`to_rdkit` hands back
    the same object, so RDKit's full API is one call away and mutations are
    visible both ways.

    Attributes:
        name: A human-readable label (e.g. an SDF title); may be empty.
        metadata: Free-form provenance/annotation, e.g. source file or ID.
    """

    __slots__ = ("_mol", "metadata", "name")

    def __init__(
        self,
        mol: Any,
        *,
        name: str = "",
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Wrap an RDKit ``Mol``.

        Args:
            mol: An RDKit ``Mol`` (or anything exposing the same surface).
            name: Optional label.
            metadata: Optional annotation dict.

        Raises:
            ValueError: If ``mol`` is ``None`` (e.g. a failed parse).
        """
        if mol is None:
            raise ValueError("Molecule requires an RDKit Mol, got None")
        self._mol = mol
        self.name = name
        self.metadata: dict[str, object] = dict(metadata or {})

    # -- constructors ---------------------------------------------------

    @classmethod
    def from_rdkit(
        cls,
        mol: Any,
        *,
        name: str = "",
        metadata: Mapping[str, object] | None = None,
    ) -> Molecule:
        """Wrap an existing RDKit ``Mol`` (shared, not copied)."""
        return cls(mol, name=name, metadata=metadata)

    @classmethod
    def from_smiles(
        cls,
        smiles: str,
        *,
        name: str = "",
        sanitize: bool = True,
        metadata: Mapping[str, object] | None = None,
    ) -> Molecule:
        """Build a molecule from a SMILES string.

        Args:
            smiles: The SMILES to parse.
            name: Optional label.
            sanitize: Run RDKit sanitization (valence, aromaticity). Turn
                off only if you intend to sanitize yourself.
            metadata: Optional annotation dict.

        Raises:
            RDKitNotInstalledError: If RDKit isn't installed.
            ValueError: If RDKit can't parse ``smiles``.
        """
        mol = _rdkit.mol_from_smiles(smiles, sanitize=sanitize)
        return cls(mol, name=name, metadata=metadata)

    @classmethod
    def from_atom_array(
        cls,
        atom_array: AtomArray,
        *,
        charge: int = 0,
        perceive_bond_orders: bool = True,
        name: str = "",
        metadata: Mapping[str, object] | None = None,
    ) -> Molecule:
        """Build a molecule from an :class:`~molforge.core.AtomArray`, perceiving
        bonds from geometry.

        The reverse of :meth:`to_atom_array`: it takes the array's element
        symbols and 3D coordinates and infers connectivity — and, by default,
        bond orders — with RDKit's geometry-based perception, recovering the
        chemistry the flat coordinate representation doesn't carry.

        Perception is designed for *small molecules*: pass a ligand you've
        sliced out of a structure, not a whole protein (perceiving bonds over
        thousands of atoms is slow and unreliable). Formal charges aren't
        stored on the array, so give the net ``charge`` if the molecule isn't
        neutral — perception needs it to assign bond orders correctly.

        Args:
            atom_array: The atoms to build from; its ``element`` and ``coords``
                are used.
            charge: Net formal charge of the molecule (for bond-order
                perception).
            perceive_bond_orders: Infer bond orders (single/double/aromatic);
                when False, only connectivity is perceived (all bonds single).
            name: Optional label.
            metadata: Optional annotation dict.

        Returns:
            A new :class:`Molecule` with perceived bonds.

        Raises:
            RDKitNotInstalledError: If RDKit isn't installed.
            ValueError: If RDKit can't perceive bonds from the geometry.
        """
        elements = [str(element) for element in atom_array.element]
        mol = _rdkit.mol_from_atoms(
            elements,
            atom_array.coords,
            charge=charge,
            perceive_bond_orders=perceive_bond_orders,
        )
        return cls(mol, name=name, metadata=metadata)

    # -- conversion -----------------------------------------------------

    def to_rdkit(self) -> Any:
        """The underlying RDKit ``Mol`` (the same object, not a copy)."""
        return self._mol

    def to_atom_array(
        self,
        *,
        embed: bool = False,
        add_hydrogens: bool = False,
        seed: int = 0xF00D,
    ) -> AtomArray:
        """Flatten to an :class:`~molforge.core.AtomArray` of 3D coordinates.

        This is the bridge from chemistry (bonds, charges) to the flat,
        coordinate-first world of :class:`~molforge.core.AtomArray` and the
        structure/ML tooling built on it.

        If the molecule already carries a conformer, its coordinates are used
        as-is. If it has none — as a molecule parsed from SMILES does —
        ``embed=True`` generates one on demand with RDKit's ETKDG, while
        ``embed=False`` (the default) raises rather than inventing geometry.

        The atoms come back as a single ``HETATM`` / ``ligand`` residue, with
        per-element atom names (``C1``, ``C2``, ``N1``, ...) and formal
        charges carried across.

        Args:
            embed: Generate a 3D conformer when the molecule has none. Has no
                effect when the molecule already carries coordinates.
            add_hydrogens: When embedding, add explicit hydrogens first for
                more realistic geometry (they appear in the output).
            seed: Random seed for embedding, so the geometry is reproducible.

        Returns:
            An :class:`~molforge.core.AtomArray` with one atom per mol atom.

        Raises:
            ValueError: If the molecule has no conformer and ``embed`` is
                False, or if RDKit can't embed one.
            RDKitNotInstalledError: If RDKit isn't installed.
        """
        if _rdkit.has_conformer(self._mol):
            mol = self._mol
        elif embed:
            mol = _rdkit.embed_conformer(self._mol, seed=seed, add_hs=add_hydrogens)
        else:
            raise ValueError(
                "Molecule has no 3D coordinates; pass embed=True to generate a "
                "conformer, or convert one that already carries them (e.g. read "
                "from an SDF with 3D structures)."
            )
        elements, charges, coords = _rdkit.conformer_atoms(mol)
        n = len(elements)
        return AtomArray.from_dict(
            {
                "coords": np.asarray(coords, dtype=np.float32).reshape(n, 3),
                "element": np.array(elements, dtype="U2"),
                "atom_name": _element_atom_names(elements),
                "charge": np.array(charges, dtype=np.float32),
                "serial": np.arange(1, n + 1, dtype=np.int32),
                "record_type": np.full(n, "HETATM", dtype="U6"),
                "entity_type": np.full(n, "ligand", dtype="U8"),
                "residue_name": np.full(n, "LIG", dtype="U3"),
                "residue_id": np.ones(n, dtype=np.int32),
            }
        )

    # -- identity / properties -----------------------------------------

    @property
    def smiles(self) -> str:
        """Canonical isomeric SMILES."""
        return _rdkit.to_smiles(self._mol)

    @property
    def inchi(self) -> str:
        """Standard InChI."""
        return _rdkit.to_inchi(self._mol)

    @property
    def inchikey(self) -> str:
        """Standard InChIKey — a stable structural identifier, handy for
        deduplicating a set of molecules."""
        return _rdkit.to_inchikey(self._mol)

    @property
    def formula(self) -> str:
        """Hill-system molecular formula, e.g. ``"C2H6O"``."""
        return _rdkit.formula(self._mol)

    @property
    def molecular_weight(self) -> float:
        """Average molecular weight in g/mol."""
        return _rdkit.molecular_weight(self._mol)

    @property
    def formal_charge(self) -> int:
        """Net formal charge."""
        return _rdkit.formal_charge(self._mol)

    @property
    def n_atoms(self) -> int:
        """Number of atoms (including explicit hydrogens)."""
        return int(self._mol.GetNumAtoms())

    @property
    def n_heavy_atoms(self) -> int:
        """Number of non-hydrogen atoms."""
        return int(self._mol.GetNumHeavyAtoms())

    def __repr__(self) -> str:
        label = f" name={self.name!r}" if self.name else ""
        return f"Molecule(n_atoms={self.n_atoms}{label})"
