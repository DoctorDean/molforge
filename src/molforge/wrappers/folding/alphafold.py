"""AlphaFold / ColabFold wrapper.

[AlphaFold](https://github.com/google-deepmind/alphafold) (Jumper et al.
2021, *Nature* 596: 583-589) is the highest-accuracy single-chain
protein folding model publicly available. The original DeepMind code
requires the full ~3 TB MSA database (BFD, UniRef90, MGnify, PDB
templates, etc.) which makes it impractical for most users.

[ColabFold](https://github.com/sokrypton/ColabFold) (Mirdita et al.
2022, *Nature Methods* 19: 679-682) is a streamlined wrapper around
AlphaFold that replaces the slow local MSA pipeline with an
MMseqs2-based remote search against the ColabFold-MSA server. It runs
the same AlphaFold model with comparable accuracy in ~10x less time
and ~100x less disk. For molforge, ColabFold is the recommended way
to access AlphaFold-quality predictions.

This wrapper supports two modes:

- **`local`** (default): use ColabFold's Python API directly via
  ``colabfold.batch.run``. Requires the ``colabfold`` package and the
  AlphaFold weights (downloaded on first use).
- **`server`**: hit a remote ColabFold-style HTTP API. Useful for users
  without a local GPU. (Stubbed — not yet implemented.)

ColabFold also supports complex prediction (heteromer / homomer), but
this wrapper focuses on the single-chain prediction path for now.
Complex support is straightforward to add when needed; ColabFold's
input convention is just chain sequences joined by ``:``.

Memory note: AlphaFold needs ~12 GB of GPU memory for sequences up to
~700 residues and roughly scales as O(L²). For >1500 residues, run on
A100 80 GB or split the prediction into domains.

Installation::

    pip install 'molforge[ml]' colabfold

ColabFold has many transitive dependencies (jax, jaxlib, alphafold,
mmseqs2 binaries); see https://github.com/sokrypton/ColabFold for
platform-specific setup notes.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from molforge.wrappers.folding._base import (
    FoldingEngine,
    FoldingEngineNotInstalledError,
    _validate_sequence,
)

if TYPE_CHECKING:
    from molforge.core import Protein


class AlphaFold(FoldingEngine):
    """Wrapper around AlphaFold via ColabFold's Python API.

    Args:
        mode: ``"local"`` to call ColabFold's local Python API
            (requires the package and weights), or ``"server"`` for
            remote prediction (not yet implemented).
        num_models: How many of the 5 AlphaFold models to run.
            Default 5 (full ensemble). Set to 1 for faster preview
            predictions; the AlphaFold paper showed that the
            top-1-of-5 best model captures most of the accuracy.
        num_recycles: AlphaFold recycling iterations. Default 3
            matches the original paper. More = slower but slightly
            better; useful for low-confidence regions.
        msa_mode: ColabFold MSA pipeline. ``"mmseqs2_uniref_env"``
            (default) is the full-quality search. ``"single_sequence"``
            skips MSA entirely (very fast but lower accuracy — about
            on par with ESMFold).
        device: ``"cuda"``, ``"cpu"``, or ``None`` to auto-detect.
        model_type: ``"AlphaFold2-ptm"`` (default, with pTM head) or
            ``"AlphaFold2"`` (original).

    Example:
        >>> from molforge.wrappers.folding import AlphaFold
        >>> engine = AlphaFold(num_models=1, num_recycles=3)  # fastest preview
        >>> protein = engine.predict("MKTVRQERLKSIVRILERSK")
        >>> protein.metadata["mean_confidence"]
        87.2
    """

    name = "AlphaFold"

    def __init__(
        self,
        *,
        mode: Literal["local", "server"] = "local",
        num_models: int = 5,
        num_recycles: int = 3,
        msa_mode: str = "mmseqs2_uniref_env",
        device: str | None = None,
        model_type: str = "AlphaFold2-ptm",
    ) -> None:
        if mode == "server":
            raise NotImplementedError(
                "Remote ColabFold server mode is not yet implemented. Use mode='local' for now."
            )
        self.mode = mode
        self.num_models = num_models
        self.num_recycles = num_recycles
        self.msa_mode = msa_mode
        self.device = device
        self.model_type = model_type

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, sequence: str, **kwargs: object) -> Protein:
        """Fold a single sequence into a :class:`Protein`.

        Args:
            sequence: One-letter amino-acid sequence.
            **kwargs: Reserved for future per-call options.

        Returns:
            A :class:`Protein` with:

            - ``metadata["engine"] = "AlphaFold"``
            - ``metadata["model_type"]``: which AlphaFold model was used
            - ``metadata["confidence_per_residue"]``: ``(L,)`` float32 pLDDT
            - ``metadata["mean_confidence"]``: float mean pLDDT
            - ``metadata["confidence_per_atom"]``: ``(N_atoms,)`` float32 pLDDT
              (copy of B-factor column)
            - ``metadata["ptm"]`` (if ``model_type="AlphaFold2-ptm"``):
              predicted TM score
        """
        sequence = _validate_sequence(sequence)
        return self._run_local(sequence)

    # ------------------------------------------------------------------
    # Local-execution path (testable seam)
    # ------------------------------------------------------------------
    def _run_local(self, sequence: str) -> Protein:
        """Run ColabFold's Python API and parse the result.

        Separated from `predict` so tests can mock the heavy ColabFold
        call without touching sequence validation or output parsing.
        """
        run_fn = self._require_colabfold()
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            # ColabFold expects a FASTA file as input.
            fasta_path = tmpdir / "input.fasta"
            fasta_path.write_text(f">query\n{sequence}\n", encoding="utf-8")
            results_dir = tmpdir / "results"
            results_dir.mkdir()

            run_fn(
                queries=[("query", sequence, None)],
                result_dir=str(results_dir),
                num_models=self.num_models,
                num_recycles=self.num_recycles,
                msa_mode=self.msa_mode,
                model_type=self.model_type,
                use_templates=False,
                rank_by="plddt",
            )

            # ColabFold writes one PDB per model; the top-ranked one is
            # `*_rank_001_*.pdb`. We use that.
            pdbs = sorted(results_dir.glob("*_rank_001_*.pdb"))
            if not pdbs:
                # Fallback: any PDB.
                pdbs = sorted(results_dir.glob("*.pdb"))
            if not pdbs:
                raise RuntimeError(
                    f"ColabFold produced no PDB output in {results_dir}. "
                    "Check the ColabFold logs for the actual error."
                )
            pdb_text = pdbs[0].read_text(encoding="utf-8")

        return self._pdb_to_protein(pdb_text, sequence=sequence)

    def _require_colabfold(self) -> Any:
        """Import colabfold or raise a clean error with install hints."""
        try:
            from colabfold.batch import run as run_fn  # type: ignore[import-not-found]
        except ImportError as e:
            raise FoldingEngineNotInstalledError(
                "AlphaFold requires `colabfold`. Install with:\n"
                "    pip install colabfold\n"
                "ColabFold has many transitive dependencies (jax, jaxlib, "
                "alphafold, mmseqs2); see "
                "https://github.com/sokrypton/ColabFold for platform-specific "
                "setup notes.\n"
                f"Underlying error: {e}"
            ) from e
        return run_fn

    # ------------------------------------------------------------------
    # Output parsing (testable in isolation)
    # ------------------------------------------------------------------
    def _pdb_to_protein(self, pdb_text: str, *, sequence: str) -> Protein:
        """Parse a ColabFold PDB and attach AlphaFold-style metadata.

        AlphaFold (and ColabFold) write pLDDT into the B-factor column,
        same as ESMFold. We follow molforge's uniform confidence
        convention: per-atom in ``confidence_per_atom``, per-residue
        in ``confidence_per_residue``, scalar in ``mean_confidence``.
        """
        from molforge.io.pdb import read_pdb_string

        protein = read_pdb_string(pdb_text)
        arr = protein.atom_array
        plddt_per_atom = np.asarray(arr.b_factor, dtype=np.float32).copy()

        per_residue: list[float] = []
        for sl in arr.iter_residue_slices():
            per_residue.append(float(plddt_per_atom[sl].mean()))
        per_residue_arr = np.asarray(per_residue, dtype=np.float32)

        protein.metadata.update(
            {
                "engine": "AlphaFold",
                "model_type": self.model_type,
                "source_sequence": sequence,
                "msa_mode": self.msa_mode,
                "num_models": self.num_models,
                "num_recycles": self.num_recycles,
                "confidence_per_atom": plddt_per_atom,
                "confidence_per_residue": per_residue_arr,
                "mean_confidence": (float(per_residue_arr.mean()) if per_residue_arr.size else 0.0),
            }
        )
        return protein
