"""Protein language model embeddings (ESM-2 wrapper).

Wraps [ESM-2](https://github.com/facebookresearch/esm) (Lin et al. 2023,
*Science* 379: 1123-1130) — Meta's protein language model. ESM-2 is the
de-facto pre-trained model for protein representations: train a head
on its embeddings and you have a strong baseline for almost any
protein-prediction task.

We support the HuggingFace `facebook/esm2_*` checkpoint family. The
relevant model sizes:

  - ``esm2_t33_650M_UR50D`` — 650M params, 1280-dim embedding (default).
    The sweet spot of accuracy vs. cost in most papers.
  - ``esm2_t30_150M_UR50D`` — 150M params, 640-dim. Fast preview.
  - ``esm2_t36_3B_UR50D`` — 3B params, 2560-dim. Highest accuracy.
  - ``esm2_t48_15B_UR50D`` — 15B params, 5120-dim. Research scale.

Memory: ESM-2 650M needs ~4 GB of GPU memory for sequences up to ~700
residues, scaling roughly as O(L²) with sequence length.

Installation::

    pip install 'molforge[ml]'
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Sequence


class EmbeddingNotInstalledError(ImportError):
    """Raised when the heavy embedding dependencies aren't installed."""


class ESM2Embedder:
    """Per-residue and per-sequence embeddings via ESM-2.

    Args:
        model_name: HuggingFace model identifier (default
            ``"facebook/esm2_t33_650M_UR50D"``).
        device: ``"cuda"``, ``"cpu"``, ``"mps"``, or ``None`` for
            auto-detect.
        layer: which transformer layer to extract embeddings from.
            Defaults to the model's last layer. Final-layer embeddings
            are most discriminative for most downstream tasks; mid-layer
            embeddings often work better for structure prediction.
        dtype: ``"float32"`` (default) or ``"float16"`` for faster GPU
            inference.

    Example:
        >>> from molforge.ml import ESM2Embedder
        >>> embedder = ESM2Embedder(device="cuda")
        >>> emb = embedder.embed("MKTVRQERLKSIVRILERSK")
        >>> emb.shape
        (20, 1280)
    """

    def __init__(
        self,
        *,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        device: str | None = None,
        layer: int = -1,
        dtype: str = "float32",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.layer = layer
        self.dtype = dtype
        self._model: Any = None
        self._tokenizer: Any = None
        self._device_resolved: str | None = None

    # ------------------------------------------------------------------
    # Lazy load
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise EmbeddingNotInstalledError(
                "ESM-2 embeddings require `torch` and `transformers`. "
                "Install with:\n"
                "    pip install 'molforge[ml]'\n"
                f"Underlying error: {e}"
            ) from e

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = {"float16": torch.float16, "float32": torch.float32}[self.dtype]

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            output_hidden_states=True,
        )
        model = model.eval().to(device)
        self._tokenizer = tokenizer
        self._model = model
        self._device_resolved = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def embed(self, sequence: str) -> NDArray[np.float32]:
        """Per-residue embeddings for a single sequence.

        Args:
            sequence: one-letter amino-acid sequence.

        Returns:
            ``(L, D)`` float32 array, where ``L`` is sequence length
            and ``D`` is the model's embedding dimensionality
            (e.g. 1280 for the 650M model). The CLS / EOS tokens
            are stripped before returning so the leading axis aligns
            with residue index.
        """
        self._ensure_loaded()
        import torch

        inputs = self._tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        inputs = {k: v.to(self._device_resolved) for k, v in inputs.items()}
        with torch.no_grad():
            output = self._model(**inputs)
        hidden = output.hidden_states[self.layer]  # (1, L+2, D)
        # Strip the leading CLS and trailing EOS to align with residues.
        residue_emb = hidden[0, 1:-1, :].cpu().numpy().astype(np.float32)
        return cast("NDArray[np.float32]", residue_emb)

    def embed_many(self, sequences: Sequence[str]) -> list[NDArray[np.float32]]:
        """Per-residue embeddings for a batch of sequences.

        Sequences of different lengths can't be stacked into a single
        tensor without padding, so we return a list of (L_i, D) arrays.
        For length-padded batched inference, drop down to the underlying
        ``self._model`` directly via ``self._tokenizer`` with
        ``padding=True``.
        """
        return [self.embed(s) for s in sequences]

    def embed_pooled(
        self,
        sequence: str,
        *,
        pooling: str = "mean",
    ) -> NDArray[np.float32]:
        """Per-sequence embedding (a single fixed-size vector per protein).

        Args:
            sequence: one-letter amino-acid sequence.
            pooling: ``"mean"`` (default), ``"max"``, or ``"cls"``.
                ``"cls"`` returns the CLS token's embedding without
                averaging.

        Returns:
            ``(D,)`` float32 array.
        """
        self._ensure_loaded()
        import torch

        inputs = self._tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        inputs = {k: v.to(self._device_resolved) for k, v in inputs.items()}
        with torch.no_grad():
            output = self._model(**inputs)
        hidden = output.hidden_states[self.layer][0]  # (L+2, D)
        if pooling == "mean":
            # Average over actual residues (skip CLS, EOS)
            return cast(
                "NDArray[np.float32]",
                hidden[1:-1].mean(dim=0).cpu().numpy().astype(np.float32),
            )
        if pooling == "max":
            return cast(
                "NDArray[np.float32]",
                hidden[1:-1].amax(dim=0).cpu().numpy().astype(np.float32),
            )
        if pooling == "cls":
            return cast(
                "NDArray[np.float32]",
                hidden[0].cpu().numpy().astype(np.float32),
            )
        raise ValueError(f"unknown pooling {pooling!r}; expected 'mean', 'max', or 'cls'")

    def __repr__(self) -> str:
        return f"ESM2Embedder(model={self.model_name!r}, layer={self.layer})"
