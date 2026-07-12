"""Numeric descriptors that make a molecule filterable by a Criterion.

A :class:`~molforge.validation.Criterion` evaluates a flat ``{name: value}``
metric dict, so to filter molecules with one we first turn each molecule into
such a dict. :func:`molecule_descriptors` is that mapping — the RDKit-backed
:class:`~molforge.core.Molecule` properties exposed under stable metric names,
so a criterion written for design metrics reads naturally as a molecular
filter::

    from molforge.validation import Criterion
    drug_like = Criterion.lt("molecular_weight", 500) & Criterion.le("formal_charge", 0)
    dataset.filter(drug_like)

The vocabulary (:data:`DESCRIPTOR_NAMES`) is deliberately small — molecular
weight, formal charge, and atom counts — and grows as real filters need it.
Everything here is lazy: computing a descriptor without RDKit raises
:class:`~molforge.core.RDKitNotInstalledError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from molforge.core import Molecule

__all__ = ["DESCRIPTOR_NAMES", "molecule_descriptors"]


_DESCRIPTORS: dict[str, Callable[[Molecule], Any]] = {
    "molecular_weight": lambda m: m.molecular_weight,
    "formal_charge": lambda m: m.formal_charge,
    "n_atoms": lambda m: m.n_atoms,
    "n_heavy_atoms": lambda m: m.n_heavy_atoms,
}

#: The descriptor names a :class:`~molforge.validation.Criterion` may filter on.
DESCRIPTOR_NAMES: frozenset[str] = frozenset(_DESCRIPTORS)


def molecule_descriptors(
    molecule: Molecule, *, names: Iterable[str] | None = None
) -> dict[str, Any]:
    """Compute filterable descriptors for a molecule.

    Args:
        molecule: The molecule to describe.
        names: Which descriptors to compute; defaults to all of
            :data:`DESCRIPTOR_NAMES`. Restricting to the names a filter
            actually references avoids unnecessary RDKit work.

    Returns:
        A flat ``{name: value}`` dict, ready for
        :meth:`molforge.validation.Criterion.evaluate`.

    Raises:
        ValueError: If a requested name isn't a known descriptor.
        RDKitNotInstalledError: If RDKit isn't installed.
    """
    if names is None:
        selected: Iterable[str] = _DESCRIPTORS
    else:
        selected = list(names)
        unknown = sorted(n for n in selected if n not in _DESCRIPTORS)
        if unknown:
            raise ValueError(f"unknown descriptor(s) {unknown}; available: {sorted(_DESCRIPTORS)}")
    return {name: _DESCRIPTORS[name](molecule) for name in selected}
