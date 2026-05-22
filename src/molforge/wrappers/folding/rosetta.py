"""Deprecated ``Rosetta`` stub — redirects to :class:`RoseTTAFold`.

The original placeholder in this file was ambiguous about whether it
wrapped PyRosetta (a sequence-design and ab initio folding library
with restrictive licensing) or RoseTTAFold (the Baker lab's deep-
learning prediction model). The real wrapper now lives in
:mod:`molforge.wrappers.folding.rosettafold` and is exposed as
:class:`molforge.wrappers.folding.RoseTTAFold`.

The :class:`Rosetta` class here is retained as a deprecation alias so
existing imports don't break — instantiating it raises a
:class:`DeprecationWarning` and points at the new class. It will be
removed in a future minor release.

A PyRosetta wrapper, if added, would live in a separate module
(``pyrosetta.py``) with a different class name (``PyRosetta``) since
its API surface is much wider than a folding engine and doesn't fit
the :class:`FoldingEngine` contract.
"""

from __future__ import annotations

import warnings

from molforge.wrappers.folding._base import FoldingEngine
from molforge.wrappers.folding.rosettafold import RoseTTAFold


class Rosetta(RoseTTAFold):
    """Deprecated. Use :class:`RoseTTAFold` instead.

    This class is a thin subclass of :class:`RoseTTAFold` retained
    for backward compatibility with code that imported the previous
    placeholder. It emits a :class:`DeprecationWarning` on
    construction and will be removed in a future release.
    """

    name = "Rosetta"

    def __init__(self, **kwargs: object) -> None:
        warnings.warn(
            "`Rosetta` is deprecated and will be removed in a future "
            "release. Use `molforge.wrappers.folding.RoseTTAFold` "
            "instead. If you were looking for a PyRosetta wrapper, "
            "none exists yet — please open an issue at "
            "https://github.com/DoctorDean/molforge.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**kwargs)  # type: ignore[arg-type]


# Sanity check: make sure Rosetta is still a FoldingEngine subclass
# so existing isinstance() checks keep working.
assert issubclass(Rosetta, FoldingEngine)
