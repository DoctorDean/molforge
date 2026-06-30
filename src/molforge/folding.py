"""Data types for multi-component structure prediction.

molforge's single-sequence folding interface
(:meth:`FoldingEngine.predict`) covers the bread-and-butter case of
"predict the structure of one protein chain." For the AlphaFold-3-
class engines :class:`Boltz` and :class:`Chai1`, that's a small
fraction of what the underlying models can do — both engines fold
*complexes* of multiple proteins, nucleic acids, and small-molecule
ligands in a single forward pass. Modelling those complexes is the
headline drug-discovery and structural-biology use case.

This module provides the shared input shape for multi-component
prediction:

- :class:`Entity` — one component of a complex (a protein chain, a
  ligand, a DNA strand, etc.).
- :class:`ComplexSpec` — an ordered list of entities defining the
  complete system to fold.

Engines that support multi-component prediction expose a
``predict_complex(spec)`` method (see :class:`Boltz.predict_complex`
and :class:`Chai1.predict_complex`). The method returns a multi-chain
:class:`Protein` whose ``atom_array`` has one logical chain per
entity (or per copy, for homo-oligomers).

Engines that don't support complexes (:class:`ESMFold`,
:class:`AlphaFold` in single-chain mode) simply don't expose
``predict_complex``. Check with ``hasattr(engine, "predict_complex")``
or by importing the engine and reading its documented capabilities.

Design notes
------------

The :class:`Entity` shape is a *common denominator* of Boltz YAML
and Chai-1 FASTA. Both engines support:

- Polymer entities (protein / DNA / RNA) specified by one-letter
  sequence.
- Ligand entities specified by SMILES *or* CCD code (mutually
  exclusive).
- Multiple copies of an entity (homo-oligomers).
- Per-entity chain IDs (auto-assigned A, B, C, ... if omitted).

What this v1 deliberately doesn't model:

- Modified residues (Boltz uses ``modifications`` lists; Chai
  needs a separate mechanism). Carry the unmodified sequence here;
  add engine-specific modification kwargs as a follow-up.
- Custom MSAs per entity (use ``use_msa_server`` for v1).
- Templates (use the engine's underlying API for v1).
- Restraints (covalent bonds, pocket constraints). Boltz supports
  these via top-level YAML keys; deferred.
- Ions specified as a separate entity type — Boltz models them as
  CCD ligand entries with codes like ``ZN`` / ``MG``; Chai accepts
  them as ``ligand|name=`` SMILES like ``[Zn+2]``. We treat them
  as ligands.

For these advanced features, drop down to the engine's raw input
format using the engine instance's documented advanced API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# The set of entity kinds supported across Boltz and Chai-1. Both
# engines understand all four; downstream serializers map to engine-
# specific names where they differ (e.g. Boltz uses "rna" and Chai-1
# uses "rna" — happily aligned).
EntityKind = Literal["protein", "dna", "rna", "ligand"]


# The standard 20 amino acids plus the gap-style placeholder. Used
# to validate protein sequences upfront so a typo doesn't reach
# the engine and surface as an opaque parse error.
_PROTEIN_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")

# DNA / RNA alphabets. Lower-case is accepted on input but
# normalized to upper-case before serialization.
_DNA_ALPHABET = set("ACGT")
_RNA_ALPHABET = set("ACGU")


@dataclass(frozen=True)
class Entity:
    """One component of a biomolecular complex.

    An Entity describes a single chain (for polymers) or a single
    small molecule (for ligands). It does not specify a 3D structure;
    that's what the folding engine produces.

    For polymer entities (protein, dna, rna), supply ``sequence`` as
    a one-letter string. For ligand entities, supply exactly one of
    ``smiles`` (a SMILES string) or ``ccd`` (a 3-letter PDB Chemical
    Component Dictionary code, e.g. ``"ATP"``, ``"NAD"``, ``"ZN"``).

    Args:
        kind: One of ``"protein"``, ``"dna"``, ``"rna"``, ``"ligand"``.
        sequence: One-letter sequence string. Required for polymer
            kinds; must be ``None`` for ligand.
        smiles: SMILES string. Only valid for ``kind="ligand"``.
            Mutually exclusive with ``ccd``.
        ccd: 3-letter CCD code. Only valid for ``kind="ligand"``.
            Mutually exclusive with ``smiles``.
        chain_id: Logical chain identifier (single letter A-Z, or
            two-letter for large complexes). ``None`` means
            "auto-assign at serialization time" in the order
            entities appear in the :class:`ComplexSpec`.
        copies: Number of identical copies for homo-oligomers.
            ``copies=2`` declares a homodimer of this entity; the
            engine produces two chains with the same sequence but
            distinct chain IDs.
        name: Human-readable identifier for logs and reports.
            ``None`` defaults to ``"{kind}_{chain_id}"`` at
            serialization time.

    Examples:
        Single protein chain::

            Entity(kind="protein", sequence="MKQHKAMIVAL...")

        Aspirin (small molecule via SMILES)::

            Entity(kind="ligand", smiles="CC(=O)OC1=CC=CC=C1C(=O)O")

        ATP cofactor (via CCD code)::

            Entity(kind="ligand", ccd="ATP")

        Homodimer (two identical protein chains)::

            Entity(kind="protein", sequence="MKQH...", copies=2)

        DNA double helix (two complementary strands)::

            Entity(kind="dna", sequence="ATCGATCG", chain_id="A")
            Entity(kind="dna", sequence="CGATCGAT", chain_id="B")
    """

    kind: EntityKind
    sequence: str | None = None
    smiles: str | None = None
    ccd: str | None = None
    chain_id: str | None = None
    copies: int = 1
    name: str | None = None

    def __post_init__(self) -> None:
        # copies validation, applies to every kind.
        if not isinstance(self.copies, int) or self.copies < 1:
            raise ValueError(f"Entity.copies must be a positive int, got {self.copies!r}")

        if self.kind == "ligand":
            self._validate_ligand()
        elif self.kind in ("protein", "dna", "rna"):
            self._validate_polymer()
        else:
            # Literal-typed but defensive against future expansion.
            raise ValueError(
                f"Entity.kind must be one of 'protein', 'dna', 'rna', 'ligand'; got {self.kind!r}"
            )

        if self.chain_id is not None:
            self._validate_chain_id()

    def _validate_ligand(self) -> None:
        if self.sequence is not None:
            raise ValueError("Entity.kind='ligand' takes 'smiles' or 'ccd', not 'sequence'.")
        if (self.smiles is None) == (self.ccd is None):
            # Either both set or both None -> error.
            raise ValueError(
                "Entity.kind='ligand' requires exactly one of 'smiles' or 'ccd' "
                "(currently {})".format(
                    "neither was set" if self.smiles is None else "both were set"
                )
            )
        if self.smiles is not None and not self.smiles.strip():
            raise ValueError("Entity.smiles must be non-empty")
        if self.ccd is not None:
            if not self.ccd.strip():
                raise ValueError("Entity.ccd must be non-empty")
            # CCD codes are usually 1-3 uppercase letters but length
            # 5 codes exist for newer entries; accept up to 5.
            if not (1 <= len(self.ccd) <= 5):
                raise ValueError(f"Entity.ccd should be 1–5 characters, got {self.ccd!r}")

    def _validate_polymer(self) -> None:
        if self.smiles is not None or self.ccd is not None:
            raise ValueError(f"Entity.kind={self.kind!r} takes 'sequence', not 'smiles' or 'ccd'.")
        if self.sequence is None or not self.sequence.strip():
            raise ValueError(f"Entity.kind={self.kind!r} requires a non-empty 'sequence'.")
        # Validate the alphabet — catches typos before they hit the
        # engine. Normalize to upper-case for the check; the
        # serializer applies the same normalization.
        normalized = self.sequence.replace(" ", "").replace("\n", "").upper()
        if self.kind == "protein":
            alphabet = _PROTEIN_ALPHABET
        elif self.kind == "dna":
            alphabet = _DNA_ALPHABET
        else:  # rna
            alphabet = _RNA_ALPHABET
        invalid = set(normalized) - alphabet
        if invalid:
            raise ValueError(
                f"Entity.sequence for kind={self.kind!r} contains invalid "
                f"characters {sorted(invalid)!r}; expected only "
                f"{sorted(alphabet)!r}"
            )

    def _validate_chain_id(self) -> None:
        cid = self.chain_id
        if not cid or not cid.strip():
            raise ValueError("Entity.chain_id must be a non-empty string")
        # Chain IDs in mmCIF can be longer, but for cofolding both
        # Boltz and Chai work best with 1-2 character chain IDs.
        if len(cid) > 4:
            raise ValueError(
                f"Entity.chain_id {cid!r} is longer than 4 characters; "
                "Boltz and Chai-1 both produce poor output with long IDs."
            )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def is_polymer(self) -> bool:
        """True for protein / DNA / RNA entities (anything with a
        ``sequence``)."""
        return self.kind in ("protein", "dna", "rna")

    @property
    def is_ligand(self) -> bool:
        """True for ligand entities (small molecules, ions, cofactors)."""
        return self.kind == "ligand"

    def normalized_sequence(self) -> str:
        """The polymer sequence with whitespace stripped and uppercased.

        Raises:
            ValueError: If called on a non-polymer entity.
        """
        if not self.is_polymer:
            raise ValueError(
                f"normalized_sequence() called on kind={self.kind!r}; "
                "only valid for polymer entities"
            )
        assert self.sequence is not None  # for type checker
        return self.sequence.replace(" ", "").replace("\n", "").upper()


@dataclass(frozen=True)
class ComplexSpec:
    """An ordered specification of a biomolecular complex to fold.

    A :class:`ComplexSpec` is the input to multi-component prediction
    via :meth:`Boltz.predict_complex` and :meth:`Chai1.predict_complex`.
    It's the engine-agnostic shape; serializers in each wrapper convert
    it to the engine's native format (Boltz YAML or Chai typed FASTA).

    The spec is *ordered*: the order of entities maps deterministically
    to chain IDs A, B, C, ... when none are explicitly assigned. For
    multi-copy entities (``Entity.copies > 1``), each copy gets its
    own chain ID in sequence.

    Args:
        entities: Tuple of :class:`Entity` defining the complex.
            Must be non-empty.

    Validation:
        - At least one entity required.
        - All explicitly-set chain_ids must be unique across the spec.
        - At most one ligand entity may carry an ``affinity_binder``
          marker (deferred; not exposed in v1).

    Examples:
        Single-protein "complex" (degenerate; usually go through
        :meth:`FoldingEngine.predict` instead)::

            spec = ComplexSpec.from_protein("MKQH...")

        Protein + small-molecule drug::

            spec = ComplexSpec.protein_ligand(
                protein_sequence="MKQHKAMIVAL...",
                ligand_smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
            )

        Antibody-antigen (3-chain protein complex: heavy, light,
        antigen)::

            spec = ComplexSpec(entities=(
                Entity(kind="protein", sequence=heavy_chain_seq,
                       chain_id="H", name="heavy"),
                Entity(kind="protein", sequence=light_chain_seq,
                       chain_id="L", name="light"),
                Entity(kind="protein", sequence=antigen_seq,
                       chain_id="A", name="antigen"),
            ))

        Transcription factor on DNA::

            spec = ComplexSpec(entities=(
                Entity(kind="protein", sequence=tf_seq, copies=1),
                Entity(kind="dna", sequence=forward_strand),
                Entity(kind="dna", sequence=reverse_strand),
            ))
    """

    entities: tuple[Entity, ...]

    def __post_init__(self) -> None:
        if not self.entities:
            raise ValueError("ComplexSpec requires at least one entity")
        # Check chain-ID uniqueness across explicitly-set IDs.
        explicit_ids = [e.chain_id for e in self.entities if e.chain_id is not None]
        if len(explicit_ids) != len(set(explicit_ids)):
            seen: set[str] = set()
            duplicates = []
            for cid in explicit_ids:
                if cid in seen:
                    duplicates.append(cid)
                else:
                    seen.add(cid)
            raise ValueError(f"ComplexSpec has duplicate chain_id values: {duplicates!r}")

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_protein(cls, sequence: str, *, chain_id: str = "A") -> ComplexSpec:
        """Single-protein convenience constructor.

        Equivalent to ``ComplexSpec(entities=(Entity(kind="protein",
        sequence=sequence, chain_id=chain_id),))``. Used internally
        so :meth:`Boltz.predict(sequence)` can delegate to the same
        code path as :meth:`Boltz.predict_complex(...)`.
        """
        return cls(entities=(Entity(kind="protein", sequence=sequence, chain_id=chain_id),))

    @classmethod
    def protein_ligand(
        cls,
        *,
        protein_sequence: str,
        ligand_smiles: str | None = None,
        ligand_ccd: str | None = None,
        protein_chain_id: str = "A",
        ligand_chain_id: str = "B",
    ) -> ComplexSpec:
        """Headline protein + small-molecule constructor.

        The standard drug-discovery shape: one protein chain plus one
        small molecule. Supply the ligand as ``ligand_smiles`` or
        ``ligand_ccd`` (mutually exclusive).

        Example::

            spec = ComplexSpec.protein_ligand(
                protein_sequence="MVTPEG...",
                ligand_smiles="N[C@@H](Cc1ccc(O)cc1)C(=O)O",
            )
        """
        return cls(
            entities=(
                Entity(
                    kind="protein",
                    sequence=protein_sequence,
                    chain_id=protein_chain_id,
                ),
                Entity(
                    kind="ligand",
                    smiles=ligand_smiles,
                    ccd=ligand_ccd,
                    chain_id=ligand_chain_id,
                ),
            )
        )

    # ------------------------------------------------------------------
    # Iteration with chain-ID assignment
    # ------------------------------------------------------------------
    def assigned_chain_ids(self) -> list[list[str]]:
        """Return chain IDs per entity after auto-assignment.

        Returns a list of length ``len(self.entities)``. Each element
        is itself a list of length ``entity.copies`` containing the
        chain IDs that will appear in the final structure for that
        entity's copies.

        Auto-assignment uses A, B, C, ..., Z, AA, AB, ... in spec
        order, skipping any IDs already claimed by explicit
        ``Entity.chain_id`` values.

        Example::

            spec = ComplexSpec(entities=(
                Entity(kind="protein", sequence="MKQ", copies=2),     # auto
                Entity(kind="protein", sequence="HIS", chain_id="X"), # explicit
                Entity(kind="ligand", smiles="O"),                    # auto
            ))
            spec.assigned_chain_ids()
            # → [["A", "B"], ["X"], ["C"]]
        """
        claimed = {e.chain_id for e in self.entities if e.chain_id is not None}
        assigned: list[list[str]] = []
        auto = _ChainIdAllocator(claimed)
        for entity in self.entities:
            if entity.chain_id is not None:
                # Explicit ID. For multi-copy + explicit chain_id, we
                # need additional IDs — allocate them auto-style.
                chain_ids = [entity.chain_id]
                for _ in range(entity.copies - 1):
                    chain_ids.append(auto.next())
                assigned.append(chain_ids)
            else:
                # Auto-assign all copies.
                chain_ids = [auto.next() for _ in range(entity.copies)]
                assigned.append(chain_ids)
        return assigned


# ---------------------------------------------------------------------
# Chain-ID allocator (module-internal, tested via ComplexSpec)
# ---------------------------------------------------------------------


class _ChainIdAllocator:
    """Yields chain IDs in spec order, skipping claimed ones.

    Sequence: A, B, ..., Z, AA, AB, ..., AZ, BA, ...
    """

    def __init__(self, claimed: set[str]) -> None:
        self._claimed = set(claimed)
        self._index = 0

    def next(self) -> str:
        while True:
            cid = _index_to_chain_id(self._index)
            self._index += 1
            if cid not in self._claimed:
                self._claimed.add(cid)
                return cid


def _index_to_chain_id(i: int) -> str:
    """Map 0 → 'A', 1 → 'B', ..., 25 → 'Z', 26 → 'AA', 27 → 'AB', ...

    Uses base-26 with letters A-Z. Sufficient for complexes up to
    676 chains, which is well beyond any practical cofolding use case.
    """
    if i < 0:
        raise ValueError(f"chain_id index must be >= 0, got {i}")
    if i < 26:
        return chr(ord("A") + i)
    # Two-letter range: 26..701 covers AA..ZZ
    first = (i - 26) // 26
    second = (i - 26) % 26
    return chr(ord("A") + first) + chr(ord("A") + second)


__all__ = [
    "ComplexSpec",
    "Entity",
    "EntityKind",
]
