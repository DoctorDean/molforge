"""Reference molforge plugin — registers a minimal but *real* folding engine.

This package is a copy-paste template for extending molforge from an
external package via the ``molforge.plugins`` entry-point group. It is
intentionally tiny, but everything here is the genuine article: swap
:class:`ExtendedChainFolder` for your real engine, keep the
:func:`register` shape and the entry point in ``pyproject.toml``, and you
have a working plugin.

The three things any plugin does:

1. Implement a molforge contract (here the
   :class:`~molforge.wrappers.folding.FoldingEngine` ABC).
2. Expose a no-argument :func:`register` callable that registers it.
3. Declare the entry point in ``pyproject.toml``::

       [project.entry-points."molforge.plugins"]
       example = "example_plugin:register"

Users then run ``molforge.plugins.discover()`` and your engine is available
via ``molforge.plugins.get("engine", "example")``.
"""

from __future__ import annotations

import numpy as np

from molforge.core import AtomArray, Protein
from molforge.core import metadata_keys as mk
from molforge.core.provenance import Provenance
from molforge.plugins import register_engine
from molforge.wrappers.folding import FoldingEngine

__version__ = "0.0.1"

# Approximate Cα–Cα distance along a fully extended chain, in ångström.
_CA_SPACING = 3.8


class ExtendedChainFolder(FoldingEngine):
    """A trivial folding engine that lays the sequence out as a straight chain.

    It implements the real ``predict(sequence) -> Protein`` contract, so
    it's a faithful template rather than a no-op: the returned structure is
    a valid molforge :class:`~molforge.core.Protein` carrying the uniform
    confidence metadata and a :class:`~molforge.core.provenance.Provenance`
    every well-behaved engine should attach. The geometry is meaningless (a
    straight line of Cα atoms) — that's the one thing a real engine
    replaces.
    """

    name = "ExtendedChainFolder"

    def predict(self, sequence: str, **kwargs: object) -> Protein:
        """Return an extended-chain structure for ``sequence``."""
        seq = "".join(sequence.split()).upper()
        if not seq or not seq.isalpha():
            raise ValueError("sequence must be a non-empty string of letters")

        n = len(seq)
        coords = np.zeros((n, 3), dtype=np.float32)
        coords[:, 0] = np.arange(n) * _CA_SPACING

        arr = AtomArray.from_dict(
            {
                "coords": coords,
                "atom_name": np.array(["CA"] * n, dtype="U4"),
                "residue_id": np.arange(1, n + 1, dtype="int32"),
                "chain_id": np.array(["A"] * n, dtype="U4"),
                "entity_type": np.array(["protein"] * n, dtype="U8"),
            }
        )
        protein = Protein(arr, name="extended_chain")

        # The uniform confidence metadata every folding engine sets (a flat
        # 50.0 here — this engine has no real confidence signal).
        confidence = np.full(n, 50.0, dtype=np.float32)
        protein.metadata[mk.ENGINE] = self.name
        protein.metadata[mk.SOURCE_SEQUENCE] = seq
        protein.metadata[mk.CONFIDENCE_PER_RESIDUE] = confidence
        protein.metadata[mk.MEAN_CONFIDENCE] = float(confidence.mean())
        protein.metadata[mk.PROVENANCE] = Provenance.from_engine(
            engine=self.name,
            engine_version=__version__,
            inputs={"sequence": seq},
        )
        return protein


def register() -> None:
    """Entry-point callable. Invoked by :func:`molforge.plugins.discover`."""
    register_engine("example", ExtendedChainFolder)
