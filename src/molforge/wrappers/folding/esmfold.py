"""ESMFold wrapper.

[ESMFold](https://github.com/facebookresearch/esm) is a transformer-based
single-sequence folding model from Meta AI. Unlike AlphaFold it doesn't
require an MSA, which makes it fast (seconds, not minutes) and the most
practical folding engine for high-throughput workflows. It's somewhat
less accurate than AlphaFold for low-pLDDT regions but for most
applications the speed/accuracy trade is worth it.

This wrapper integrates with the official HuggingFace
``facebook/esmfold_v1`` checkpoint via ``transformers``. It exposes a
single :meth:`ESMFold.predict` method that takes a sequence string and
returns a :class:`molforge.core.Protein` with pLDDT in
``protein.metadata["confidence_per_residue"]``.

The heavy dependencies (``torch``, ``transformers``) are imported lazily,
so ``import molforge`` stays cheap. Installing them:

.. code-block:: bash

    pip install "molforge[ml]"

Memory note: ESMFold needs ~10 GB of GPU memory for sequences up to
~500 residues. On CPU it works but is slow (minutes per sequence).
For long sequences, use ``chunk_size`` to trade speed for memory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from molforge.cache import get_default_cache
from molforge.core import metadata_keys as mk
from molforge.core.provenance import Provenance
from molforge.wrappers._versions import check_engine_version
from molforge.wrappers.folding._base import (
    FoldingEngine,
    FoldingEngineNotInstalledError,
    _validate_sequence,
)

if TYPE_CHECKING:
    from molforge.core import Protein


class ESMFold(FoldingEngine):
    """Wrapper around Meta AI's ESMFold (single-sequence transformer folder).

    Args:
        model_name: HuggingFace model identifier. Defaults to
            ``"facebook/esmfold_v1"``, the public ESMFold v1 checkpoint.
        device: Where to run inference. ``"cuda"``, ``"cpu"``, ``"mps"``,
            or ``None`` to auto-detect (CUDA if available, else CPU).
        chunk_size: Axial-attention chunk size (lower = less memory but
            slower). ``None`` for no chunking. ``64`` is a reasonable
            default for sequences > 700 aa on a 24 GB GPU.
        dtype: ``"float32"`` (default) or ``"float16"`` for faster GPU
            inference at the cost of marginal accuracy.

    Example:
        >>> from molforge.wrappers.folding import ESMFold
        >>> engine = ESMFold(device="cuda")
        >>> protein = engine.predict("MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVS")
        >>> protein.metadata["mean_confidence"]
        82.4
    """

    name = "ESMFold"

    def __init__(
        self,
        *,
        model_name: str = "facebook/esmfold_v1",
        device: str | None = None,
        chunk_size: int | None = None,
        dtype: str = "float32",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.chunk_size = chunk_size
        self.dtype = dtype
        # The model + tokenizer are lazily loaded on first predict().
        self._model: Any = None
        self._tokenizer: Any = None

    # ------------------------------------------------------------------
    # Lazy load
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        """Load tokenizer + model if not already in memory.

        Heavy imports (``torch``, ``transformers``) happen here so that
        constructing an ``ESMFold()`` object is free until you actually
        call ``predict``.
        """
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoTokenizer, EsmForProteinFolding
        except ImportError as e:
            raise FoldingEngineNotInstalledError(
                "ESMFold requires `torch` and `transformers`. Install with:\n"
                "    pip install 'molforge[ml]'\n"
                f"Underlying error: {e}"
            ) from e

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        torch_dtype = {"float16": torch.float16, "float32": torch.float32}[self.dtype]
        model = EsmForProteinFolding.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        model = model.eval().to(device)
        if self.chunk_size is not None:
            model.trunk.set_chunk_size(self.chunk_size)
        self._tokenizer = tokenizer
        self._model = model
        self._device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, sequence: str, **kwargs: object) -> Protein:
        """Fold a single sequence into a :class:`Protein`.

        Args:
            sequence: One-letter amino-acid sequence.
            **kwargs: Reserved for future per-call options; currently unused.

        Returns:
            A :class:`Protein` with one chain (``"A"``), the predicted
            structure, and:

            - ``metadata["provenance"]``: :class:`molforge.core.Provenance`
            - ``metadata["engine"] = "ESMFold"``
            - ``metadata["confidence_per_residue"]``: ``(L,)`` float32 pLDDT
            - ``metadata["mean_confidence"]``: float mean pLDDT
            - ``metadata["confidence_per_atom"]``: ``(N_atoms,)`` float32 pLDDT
                (copy of B-factor column for convenience)
        """
        sequence = _validate_sequence(sequence)

        # Cache lookup. Build the Provenance upfront from inputs +
        # constructor parameters; if a previous identical call has
        # cached its result, return it without touching the model.
        provenance = self._build_provenance(sequence)
        cache = get_default_cache()
        cached: Protein | None = cache.get(provenance, "protein")
        if cached is not None:
            return cached

        self._ensure_loaded()

        # Run the model and convert to molforge's representation. The
        # pipeline:
        #   1. Tokenize the sequence.
        #   2. Forward pass to get atom positions + pLDDT.
        #   3. Write a PDB string (using transformers' own output util).
        #   4. Parse that PDB string into a molforge.Protein.
        import torch

        inputs = self._tokenizer(
            [sequence],
            return_tensors="pt",
            add_special_tokens=False,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self._model(**inputs)

        pdb_text = self._output_to_pdb(output)
        result = self._pdb_to_protein(pdb_text, sequence=sequence, provenance=provenance)
        cache.put(provenance, result, "protein")
        return result

    def _build_provenance(self, sequence: str) -> Provenance:
        """Construct the Provenance for a predict() call.

        Factored out of :meth:`_pdb_to_protein` so :meth:`predict`
        can build the Provenance upfront for cache lookup. Pure
        function of inputs + constructor parameters.
        """
        # transformers >= 4.40 is required by the `ml` extra and provides the
        # EsmForProteinFolding + convert_outputs_to_pdb utilities this wrapper
        # parses. Record the version (for provenance / reproducibility) and
        # warn if it's below that floor.
        version = check_engine_version("transformers", engine="ESMFold", minimum="4.40")
        return Provenance.from_engine(
            engine="ESMFold",
            operation="predict",
            engine_version=version,
            parameters={
                "model_name": self.model_name,
                "device": self.device,
                "chunk_size": self.chunk_size,
                "dtype": self.dtype,
            },
            inputs={"sequence": sequence},
        )

    # ------------------------------------------------------------------
    # Helpers (separated for testability — these are what tests mock)
    # ------------------------------------------------------------------
    def _output_to_pdb(self, output: Any) -> str:
        """Convert the model's raw output to a PDB-formatted string.

        Delegates to ``transformers``' own ``convert_outputs_to_pdb`` so
        we stay aligned with upstream changes to the output format.
        """
        # The transformers convenience function returns list[str], one per item
        from transformers.models.esm.openfold_utils import (
            atom14_to_atom37,
        )
        from transformers.models.esm.openfold_utils.protein import (
            Protein as OFProtein,
        )
        from transformers.models.esm.openfold_utils.protein import (
            to_pdb,
        )

        final_atom_positions = atom14_to_atom37(output["positions"][-1], output)
        outputs = {k: v.to("cpu").numpy() for k, v in output.items()}
        final_atom_positions = final_atom_positions.cpu().numpy()
        final_atom_mask = outputs["atom37_atom_exists"]
        of_proteins = []
        for i in range(outputs["aatype"].shape[0]):
            aa = outputs["aatype"][i]
            pred_pos = final_atom_positions[i]
            mask = final_atom_mask[i]
            resid = outputs["residue_index"][i] + 1
            pred = OFProtein(
                aatype=aa,
                atom_positions=pred_pos,
                atom_mask=mask,
                residue_index=resid,
                b_factors=outputs["plddt"][i],
                chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
            )
            of_proteins.append(to_pdb(pred))
        return str(of_proteins[0])

    def _pdb_to_protein(
        self,
        pdb_text: str,
        *,
        sequence: str,
        provenance: Provenance | None = None,
    ) -> Protein:
        """Parse the model's PDB output and attach ESMFold metadata.

        Args:
            pdb_text: PDB-formatted string from :meth:`_output_to_pdb`.
            sequence: The original input sequence (stored in metadata).
            provenance: Pre-built :class:`Provenance` from
                :meth:`_build_provenance`. ``None`` falls back to
                building one here (kept for tests that call this
                method directly).
        """
        from molforge.io.pdb import read_pdb_string

        protein = read_pdb_string(pdb_text)
        arr = protein.atom_array
        plddt_per_atom = np.asarray(arr.b_factor, dtype=np.float32).copy()

        per_residue: list[float] = []
        for sl in arr.iter_residue_slices():
            per_residue.append(float(plddt_per_atom[sl].mean()))
        per_residue_arr = np.asarray(per_residue, dtype=np.float32)

        # First-class provenance record. Engine config goes into
        # `parameters`; the sequence (the actual input data) into
        # `inputs`. The engine name and model_name remain in the ad-hoc
        # keys below for backwards compatibility.
        prov = provenance if provenance is not None else self._build_provenance(sequence)

        protein.metadata.update(
            {
                mk.PROVENANCE: prov,
                mk.ENGINE: "ESMFold",
                mk.MODEL_NAME: self.model_name,
                mk.SOURCE_SEQUENCE: sequence,
                mk.CONFIDENCE_PER_ATOM: plddt_per_atom,
                mk.CONFIDENCE_PER_RESIDUE: per_residue_arr,
                mk.MEAN_CONFIDENCE: float(per_residue_arr.mean()) if per_residue_arr.size else 0.0,
            }
        )
        return protein
