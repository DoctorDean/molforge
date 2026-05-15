"""Tests for ESM2 embedding wrapper.

These don't require torch/transformers to be installed. They test
construction, lazy import behaviour, and the missing-dep error path.
End-to-end tests require the real model and are marked
``@pytest.mark.slow``.
"""

from __future__ import annotations

import importlib.util

import pytest

from molforge.ml import EmbeddingNotInstalledError, ESM2Embedder


def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


class TestConstruction:
    def test_defaults(self) -> None:
        e = ESM2Embedder()
        assert e.model_name == "facebook/esm2_t33_650M_UR50D"
        assert e.layer == -1
        assert e.dtype == "float32"

    def test_custom_settings(self) -> None:
        e = ESM2Embedder(
            model_name="facebook/esm2_t30_150M_UR50D",
            device="cpu",
            layer=10,
            dtype="float16",
        )
        assert e.model_name == "facebook/esm2_t30_150M_UR50D"
        assert e.layer == 10

    def test_construction_does_not_load_model(self) -> None:
        e = ESM2Embedder()
        assert e._model is None
        assert e._tokenizer is None

    def test_repr(self) -> None:
        e = ESM2Embedder(layer=20)
        assert "ESM2Embedder" in repr(e)
        assert "layer=20" in repr(e)


class TestMissingDependency:
    @pytest.mark.skipif(_torch_available(), reason="torch is installed")
    def test_embed_without_torch_raises_clear_error(self) -> None:
        e = ESM2Embedder()
        with pytest.raises(EmbeddingNotInstalledError, match="molforge\\[ml\\]"):
            e.embed("MKTV")


@pytest.mark.slow
@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
class TestEndToEnd:
    def test_embed_short_sequence(self) -> None:
        # Use the smallest model for speed
        e = ESM2Embedder(
            model_name="facebook/esm2_t6_8M_UR50D",
            device="cpu",
        )
        emb = e.embed("MKTV")
        # 8M model has 320-dim embedding
        assert emb.shape == (4, 320)
