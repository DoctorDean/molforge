"""Shared provenance-chaining helper for the prep subpackage.

Every :mod:`molforge.prep` function takes a :class:`Protein` and
returns a :class:`Protein` derived from it. Each call should chain a
:class:`molforge.core.Provenance` step onto the output so a full
preparation pipeline (``remove_heterogens → fix_missing_atoms →
add_caps → add_hydrogens``) leaves the final Protein with a 4-deep
provenance chain that reads as the workflow oldest-first.

This module exists to keep that pattern DRY across the five public
prep functions — every one of them does the same wrap-up, and a
single helper means one place to evolve the shape later (e.g. if we
decide to also record the input atom count, or stamp a ``deps``
field with the PDBFixer / OpenMM versions used).

The module is private (``_provenance``) — not part of the public
API. Callers go through :mod:`molforge.core.provenance` directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from molforge.core import metadata_keys as mk
from molforge.core.provenance import Provenance

if TYPE_CHECKING:
    from molforge.core import Protein


def chain_prep_provenance(
    output: Protein,
    *,
    engine: str,
    parameters: dict[str, Any],
    input_protein: Protein,
) -> None:
    """Attach a chained :class:`Provenance` to ``output.metadata``.

    The new step's ``parent`` is the input protein's existing
    Provenance (when present), so calling several prep functions in
    sequence builds a chain.

    Args:
        output: The Protein the prep function is about to return.
            Modified in place: ``output.metadata[PROVENANCE]`` is
            set to a fresh Provenance.
        engine: Producer string. Convention is ``"molforge.prep.<fn>"``
            so the chain reads as a sequence of dotted function names.
        parameters: The function's call-time arguments (must be
            JSON-native; Provenance.from_engine validates).
        input_protein: The Protein the function consumed — its
            existing Provenance becomes the new step's ``parent``.
    """
    parent_obj = input_protein.metadata.get(mk.PROVENANCE)
    parent = parent_obj if isinstance(parent_obj, Provenance) else None
    output.metadata[mk.PROVENANCE] = Provenance.from_engine(
        engine=engine,
        parameters=parameters,
        # The prep functions don't take separate "input" data — the
        # Protein itself is the input, and the parent pointer is how
        # we trace it. Leaving inputs={} keeps the shape clean.
        inputs={},
        parent=parent,
    )
