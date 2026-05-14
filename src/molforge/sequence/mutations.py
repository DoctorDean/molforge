"""Sequence-level mutation utilities.

These work on :class:`molforge.core.Protein` objects (and raw strings)
and follow the conventions used in protein engineering literature:

- Mutation strings: ``A123V`` (wild-type A at position 123 to V).
- Multi-mutant strings: ``A123V/T56K`` (slash-delimited).
- Positions are **1-indexed** and refer to the residue's author-assigned
  ``seq_id`` (matches PDB residue numbers), **not** array index.

For structure-level mutations (rebuilding sidechains, energy minimization
after substitution) plug into a wrapper like Rosetta or OpenMM; the
functions here operate purely on sequence identity, which is the right
layer for design-loop bookkeeping.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from molforge.core import Protein
from molforge.core.constants import ONE_TO_THREE, THREE_TO_ONE

if TYPE_CHECKING:
    from collections.abc import Iterable


_MUTATION_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


@dataclass(frozen=True)
class Mutation:
    """A single point mutation.

    Attributes:
        wild_type: One-letter code of the original residue.
        position: 1-indexed residue number (matches PDB ``seq_id``).
        mutant: One-letter code of the new residue.
        chain_id: Optional chain identifier (some workflows need it).
    """

    wild_type: str
    position: int
    mutant: str
    chain_id: str | None = None

    @classmethod
    def parse(cls, spec: str) -> Mutation:
        """Parse a mutation string like ``"A123V"``.

        Supports chain prefix syntax ``"A:K42N"`` for chain-aware
        mutations.

        Raises:
            ValueError: If the spec doesn't match the expected format.
        """
        spec = spec.strip().upper()
        chain_id: str | None = None
        if ":" in spec:
            chain_id, spec = spec.split(":", 1)
            chain_id = chain_id.strip()
        m = _MUTATION_RE.match(spec)
        if not m:
            raise ValueError(
                f"could not parse mutation {spec!r}; expected format e.g. 'A123V' "
                "or 'A:K42N' (with chain prefix)"
            )
        wt, pos, mut = m.group(1), int(m.group(2)), m.group(3)
        if wt not in ONE_TO_THREE:
            raise ValueError(f"unknown wild-type residue code {wt!r}")
        if mut not in ONE_TO_THREE:
            raise ValueError(f"unknown mutant residue code {mut!r}")
        return cls(wild_type=wt, position=pos, mutant=mut, chain_id=chain_id)

    def __str__(self) -> str:
        prefix = f"{self.chain_id}:" if self.chain_id else ""
        return f"{prefix}{self.wild_type}{self.position}{self.mutant}"


def parse_mutations(spec: str) -> list[Mutation]:
    """Parse a slash- (or comma- or whitespace-) delimited mutation string.

    Example:
        >>> parse_mutations("A123V/T56K")
        [Mutation('A', 123, 'V'), Mutation('T', 56, 'K')]
    """
    tokens = re.split(r"[/,\s]+", spec.strip())
    return [Mutation.parse(t) for t in tokens if t]


def apply_mutation(sequence: str, mutation: Mutation | str) -> str:
    """Apply a single point mutation to a sequence string.

    Args:
        sequence: One-letter amino-acid sequence.
        mutation: A :class:`Mutation` object or mutation string (e.g. ``"A123V"``).

    Returns:
        The mutated sequence.

    Raises:
        ValueError: If the position is out of range, or the wild-type
            residue doesn't match what's actually at that position.
    """
    mut = mutation if isinstance(mutation, Mutation) else Mutation.parse(mutation)
    idx = mut.position - 1
    if idx < 0 or idx >= len(sequence):
        raise ValueError(
            f"position {mut.position} out of range for sequence of length {len(sequence)}"
        )
    actual = sequence[idx]
    if actual != mut.wild_type:
        raise ValueError(
            f"wild-type mismatch: mutation {mut} expects {mut.wild_type} at "
            f"position {mut.position}, but found {actual}"
        )
    return sequence[:idx] + mut.mutant + sequence[idx + 1 :]


def apply_mutations(sequence: str, mutations: str | Iterable[Mutation | str]) -> str:
    """Apply multiple mutations to a sequence.

    Accepts a slash-delimited string or an iterable of mutations.
    """
    if isinstance(mutations, str):
        muts = parse_mutations(mutations)
    else:
        muts = [m if isinstance(m, Mutation) else Mutation.parse(m) for m in mutations]
    out = sequence
    for m in muts:
        out = apply_mutation(out, m)
    return out


def mutate_protein(
    protein: Protein,
    mutation: Mutation | str,
    *,
    chain_id: str | None = None,
) -> Protein:
    """Apply a sequence-level mutation to a :class:`Protein`.

    Updates the ``residue_name`` field of the affected residue. **Side
    chain atoms are not rebuilt** — for full structural mutation, route
    through Rosetta, OpenMM, or a side-chain repacker. This function is
    for sequence-bookkeeping and downstream tools that re-fold or repack
    on their own.

    Args:
        protein: The structure to mutate (a copy is returned; original is
            unmodified).
        mutation: A :class:`Mutation` or mutation string.
        chain_id: Which chain to mutate. If omitted, uses the chain
            embedded in the mutation string (``"A:K42N"``) or the first
            chain in the protein.

    Returns:
        A new :class:`Protein` with the residue name updated.

    Raises:
        ValueError: If the position doesn't exist, or the wild-type
            doesn't match the residue actually present.
    """
    from copy import deepcopy

    mut = mutation if isinstance(mutation, Mutation) else Mutation.parse(mutation)
    target_chain = chain_id or mut.chain_id
    if target_chain is None:
        # Default to the first protein chain.
        target_chain = next((c.chain_id for c in protein.chains if c.sequence), None)
    if target_chain is None:
        raise ValueError("protein has no chain to mutate")

    out = deepcopy(protein)
    arr = out.atom_array
    # Find atoms belonging to the target residue.
    import numpy as np

    mask = (arr.chain_id == target_chain) & (arr.residue_id == mut.position)
    if not bool(np.any(mask)):
        raise ValueError(f"no residue at position {mut.position} in chain {target_chain!r}")
    # Validate the wild-type
    current_resname = str(arr.residue_name[np.where(mask)[0][0]])
    current_one = THREE_TO_ONE.get(current_resname, "X")
    if current_one != mut.wild_type:
        raise ValueError(
            f"wild-type mismatch: mutation {mut} expects {mut.wild_type} "
            f"({ONE_TO_THREE.get(mut.wild_type, '???')}) at position {mut.position}, "
            f"but found {current_one} ({current_resname})"
        )
    # Update the residue name. Atoms are kept — this is a sequence-only mutation.
    arr.residue_name[mask] = ONE_TO_THREE[mut.mutant]
    return out
