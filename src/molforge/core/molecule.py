"""A first-class small-molecule type, backed by RDKit."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from molforge.core import _rdkit

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["Molecule"]


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

    __slots__ = ("_mol", "name", "metadata")

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

    # -- conversion -----------------------------------------------------

    def to_rdkit(self) -> Any:
        """The underlying RDKit ``Mol`` (the same object, not a copy)."""
        return self._mol

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
