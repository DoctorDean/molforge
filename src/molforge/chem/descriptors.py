"""Numeric descriptors that make a molecule filterable by a Criterion.

A :class:`~molforge.validation.Criterion` evaluates a flat ``{name: value}``
metric dict, so to filter molecules with one we first turn each molecule into
such a dict. :func:`molecule_descriptors` is that mapping — the RDKit-backed
:class:`~molforge.core.Molecule` properties exposed under stable metric names,
so a criterion written for design metrics reads naturally as a molecular
filter::

    from molforge.validation import Criterion
    drug_like = Criterion.le("lipinski_violations", 1) & Criterion.lt("tpsa", 140)
    dataset.filter(drug_like)

The vocabulary (:data:`DESCRIPTOR_NAMES`) covers the descriptors real
compound filters reach for — molecular weight, formal charge, atom counts,
logP, TPSA, hydrogen-bond donors/acceptors, rotatable bonds, and Lipinski
rule-of-five violations — and grows as new filters need it. Everything here
is lazy: computing an RDKit-backed descriptor without RDKit raises
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
    "logp": lambda m: m.logp,
    "tpsa": lambda m: m.tpsa,
    "n_h_donors": lambda m: m.n_h_donors,
    "n_h_acceptors": lambda m: m.n_h_acceptors,
    "n_rotatable_bonds": lambda m: m.n_rotatable_bonds,
    "lipinski_violations": lambda m: m.lipinski_violations,
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
