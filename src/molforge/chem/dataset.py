"""A lazy, composable collection of molecules.

:class:`MoleculeDataset` wraps any iterable of :class:`~molforge.core.Molecule`
and offers a small set of combinators that each return a *new* dataset
without touching the source until you iterate. It is the "work with a set of
molecules" layer: ingest lazily (e.g. with :func:`molforge.io.iter_molecules`),
transform with chained combinators, then :meth:`~MoleculeDataset.collect`
only what you need. Deliberately a thin lazy pipeline — not a scheduler or
DAG engine.

Laziness has one contract worth stating plainly: a dataset is re-iterable
exactly when its source is. Built over a list it can be traversed
repeatedly; built over a one-shot iterator (like ``iter_molecules``) it is
single-pass, the same behaviour as a generator.

Example:
    >>> from molforge.io import iter_molecules
    >>> from molforge.chem import MoleculeDataset, standardize
    >>> cleaned = (
    ...     MoleculeDataset(iter_molecules("library.sdf"))
    ...     .map(standardize)
    ...     .take(1000)
    ...     .collect()
    ... )
"""

from __future__ import annotations

from itertools import islice
from typing import TYPE_CHECKING

from molforge.chem.quality import _check_key, _iter_unique, is_valid

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from molforge.core import Molecule

__all__ = ["MoleculeDataset"]


class _ReiterableSource:
    """An iterable that rebuilds a fresh iterator from a factory each pass.

    Combinators wrap their result in this so a dataset stays re-iterable
    whenever its ultimate source is (a list re-iterates; a one-shot iterator
    does not) instead of collapsing to a single-use generator.
    """

    __slots__ = ("_factory",)

    def __init__(self, factory: Callable[[], Iterator[Molecule]]) -> None:
        self._factory = factory

    def __iter__(self) -> Iterator[Molecule]:
        return iter(self._factory())


class MoleculeDataset:
    """A lazy, immutable pipeline over a stream of molecules.

    Wrap any iterable of :class:`~molforge.core.Molecule`; the combinators
    (:meth:`map`, :meth:`take`) return new datasets and nothing runs until
    the dataset is iterated or :meth:`collect`-ed.

    Attributes are intentionally hidden: a dataset is defined only by what it
    yields when iterated.
    """

    __slots__ = ("_source",)

    def __init__(self, molecules: Iterable[Molecule]) -> None:
        """Wrap an iterable of molecules (not consumed until iterated)."""
        self._source = molecules

    def __iter__(self) -> Iterator[Molecule]:
        return iter(self._source)

    def map(self, fn: Callable[[Molecule], Molecule]) -> MoleculeDataset:
        """Apply ``fn`` to every molecule, lazily.

        Args:
            fn: A per-molecule transform, e.g. :func:`molforge.chem.standardize`.

        Returns:
            A new dataset yielding ``fn(m)`` for each molecule ``m``.
        """
        source = self._source
        return MoleculeDataset(_ReiterableSource(lambda: (fn(m) for m in source)))

    def take(self, n: int) -> MoleculeDataset:
        """Keep only the first ``n`` molecules.

        Args:
            n: How many molecules to keep; ``take`` short-circuits, so an
                unbounded source is fine.

        Returns:
            A new dataset yielding at most ``n`` molecules.

        Raises:
            ValueError: If ``n`` is negative.
        """
        if n < 0:
            raise ValueError(f"take(n) requires n >= 0, got {n}")
        source = self._source
        return MoleculeDataset(_ReiterableSource(lambda: islice(iter(source), n)))

    def valid(self) -> MoleculeDataset:
        """Keep only molecules that pass RDKit sanitization.

        A lazy filter over :func:`molforge.chem.is_valid` — structures RDKit
        rejects are dropped rather than raising.

        Returns:
            A new dataset yielding only the valid molecules.
        """
        source = self._source
        return MoleculeDataset(_ReiterableSource(lambda: (m for m in source if is_valid(m))))

    def dedup(self, *, key: str = "inchikey") -> MoleculeDataset:
        """Drop duplicate molecules by structural identity, keeping the first.

        Streams with a running set of seen identities, so only the identities
        (not the molecules) are held in memory.

        Args:
            key: Identity to compare on — ``"inchikey"`` (default) or
                ``"smiles"``.

        Returns:
            A new dataset yielding the first molecule of each identity, in
            order.

        Raises:
            ValueError: If ``key`` is neither ``"inchikey"`` nor ``"smiles"``.
        """
        _check_key(key)
        source = self._source
        return MoleculeDataset(_ReiterableSource(lambda: _iter_unique(source, key=key)))

    def collect(self) -> list[Molecule]:
        """Materialize the dataset into a list, running the whole pipeline."""
        return list(self)

    def __repr__(self) -> str:
        return "MoleculeDataset(<lazy>)"
