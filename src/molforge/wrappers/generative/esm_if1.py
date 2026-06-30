"""ESM-IF1 inverse-folding wrapper.

ESM-IF1 is the inverse-folding model from Meta AI's ESM series (Hsu
et al. 2022, "Learning inverse folding from millions of predicted
structures"). Given a protein backbone, it predicts sequences that
would fold into that structure. The model architecture combines
invariant geometric input layers (GVP-GNN) with a sequence-to-
sequence transformer; it was trained on ~12M AlphaFold2-predicted
structures and achieves ~51% native sequence recovery on
structurally held-out backbones.

This wrapper sits alongside :class:`ProteinMPNN` in
``molforge.wrappers.generative``. Both solve the same inverse-folding
problem with different architectures and different training; running
both on the same backbone and taking the intersection (or comparing
refold quality) is a common practice — they often disagree, and the
agreement tends to be the more reliable signal.

The wrapper is fundamentally simpler than ProteinMPNN's: ESM-IF1
ships as a Python library (``fair-esm``), not a cloned repo, so
there's no subprocess shell-out and no ``ESMIF1_HOME`` env-var
song-and-dance. Same lazy-import pattern as ESMFold.

What this wrapper does and doesn't do
-------------------------------------

It runs the single-chain inverse-folding pipeline: load model,
extract (N, CA, C) coords from a Protein, sample N sequences at
the given temperature, optionally score each sequence's
log-likelihood. Returns a list of :class:`DesignedSequence`,
sorted best-first (lowest negative log-likelihood = highest
likelihood under the model).

It does *not* expose:

- **Multi-chain conditioning.** ESM-IF1 has ``multichain_util.sample_sequence_in_complex``
  for designing one chain conditioned on the others; that needs a
  different input shape (dict of chain_id → coords) and is worth a
  follow-up commit.
- **Partial sequence conditioning.** ESM-IF1's "partial sequence"
  mode (mask some positions, design others) is supported by the
  underlying model but would expand the molforge API surface for
  little immediate benefit; deferred until concrete user needs.
- **Custom model checkpoints.** ESM-IF1 ships one production
  checkpoint (``esm_if1_gvp4_t16_142M_UR50``); a constructor
  ``model_name`` argument is exposed for future-proofing but
  defaults to the standard one.

For multi-chain or partial-sequence design, fall back to running
the ``fair-esm`` library directly with the model handle exposed via
:attr:`ESMIF1.model` (lazily loaded on first :meth:`generate`).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from molforge.cache import get_default_cache
from molforge.core import Protein
from molforge.core import metadata_keys as mk
from molforge.core.provenance import Provenance
from molforge.generative import (
    DesignedSequence,
    GenerativeEngine,
    GenerativeEngineNotInstalledError,
)


class ESMIF1(GenerativeEngine):
    """Wrapper around the ESM-IF1 inverse-folding model.

    Args:
        model_name: Pre-trained model identifier. ESM-IF1 currently
            ships exactly one production checkpoint
            (``"esm_if1_gvp4_t16_142M_UR50"``); the argument exists
            so future checkpoint releases can be selected without
            breaking the constructor signature.
        device: Torch device string (``"cpu"`` / ``"cuda"`` /
            ``"mps"``). ``None`` lets PyTorch auto-pick; with the
            fair-esm distribution the default is CPU unless CUDA is
            visible.
        num_seqs: Number of sequences to sample per call. Defaults
            to 8 — matches ProteinMPNN's default for a smooth
            cross-engine comparison.
        temperature: Sampling temperature. ESM-IF1's own
            recommendation: 1e-6 to optimise native-sequence
            recovery, 1.0 (the model default) for diversity. molforge
            defaults to 1.0 to keep ESM-IF1 behaviour aligned with
            the upstream library; pass ``temperature=1e-6`` for
            recovery-style use.
        score_sequences: When ``True`` (default), each sampled
            sequence is run through ``score_sequence`` to compute its
            log-likelihood under the model, populating the
            :attr:`DesignedSequence.score` field. When ``False``, the
            score is recorded as ``0.0`` and the per-sample scoring
            cost (a second forward pass) is skipped — useful when
            sampling many sequences and the score doesn't matter
            downstream.
        compute_recovery: When ``True``, the native-sequence recovery
            fraction is computed for each sample (requires the native
            sequence from the input PDB). Defaults to ``True``.
        seed: PyTorch random seed for reproducibility. ``None``
            leaves the global seed untouched. Note: ESM-IF1 sampling
            still has some non-determinism from CUDA ops; pin to CPU
            for byte-identical output.
    """

    name = "ESM-IF1"

    # The canonical pre-trained checkpoint. ESM-IF1 currently ships
    # this single production model; if/when Meta release new
    # checkpoints, add them to a frozenset like ProteinMPNN does.
    _DEFAULT_MODEL = "esm_if1_gvp4_t16_142M_UR50"

    def __init__(
        self,
        *,
        model_name: str = _DEFAULT_MODEL,
        device: str | None = None,
        num_seqs: int = 8,
        temperature: float = 1.0,
        score_sequences: bool = True,
        compute_recovery: bool = True,
        seed: int | None = None,
    ) -> None:
        if num_seqs < 1:
            raise ValueError(f"num_seqs must be >= 1, got {num_seqs}")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.model_name = model_name
        self.device = device
        self.num_seqs = num_seqs
        self.temperature = temperature
        self.score_sequences = score_sequences
        self.compute_recovery = compute_recovery
        self.seed = seed

        # Model + alphabet lazily loaded on first .generate() call.
        # Weights are ~145 MB and download on first use; we don't
        # want construction to touch the network or heavy deps.
        self._model: Any = None
        self._alphabet: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(  # type: ignore[override]  # engine-specific kwargs + refined return type vs the ABC
        self,
        backbone: Protein | str | Path,
        *,
        chain_id: str = "A",
        **_kwargs: object,
    ) -> list[DesignedSequence]:
        """Sample ``self.num_seqs`` designs for the given backbone.

        Args:
            backbone: A :class:`Protein` or a path to a PDB / mmCIF
                file with the backbone to design for. If a Protein,
                it's written to a temp PDB so ESM-IF1's coord loader
                can read it.
            chain_id: Which chain to design (default ``"A"``).
                Multi-chain conditioning isn't yet supported; the
                other chains in a multi-chain backbone are ignored
                by this single-chain wrapper.

        Returns:
            A list of :class:`DesignedSequence`, length
            ``self.num_seqs``, sorted best-first by negative log-
            likelihood (lower = better, matching molforge's
            sequence-design convention).

        Raises:
            GenerativeEngineNotInstalledError: If ``fair-esm`` (and
                its peer ``torch-geometric``) aren't installed.
        """
        # Cache lookup before loading the model. We use ``self.device``
        # (the user-specified value) rather than ``_effective_device()``
        # so the cache key stays stable independent of load-time
        # auto-detection.
        provenance = self._build_provenance(backbone, chain_id=chain_id)
        cache = get_default_cache()
        cached: list[DesignedSequence] | None = cache.get(provenance, "designed_sequences")
        if cached is not None:
            return cached

        self._ensure_loaded()

        # We rely on the upstream ``load_coords`` helper because it
        # handles things like residue numbering gaps and chain breaks
        # the way ESM-IF1 expects. Writing the Protein to a temp PDB
        # and passing the path is the most robust path.
        with tempfile.TemporaryDirectory(prefix="molforge_esmif1_") as tmp:
            backbone_path = self._materialise_backbone(backbone, Path(tmp))
            coords, native_seq = self._load_coords(backbone_path, chain_id)

        designs = self._sample_designs(coords, native_seq)
        designs.sort(key=lambda d: d.score)
        # Attach the same shared Provenance built above to each design
        # (frozen + immutable, safe to share by reference).
        for d in designs:
            d.metadata[mk.PROVENANCE] = provenance
        cache.put(provenance, designs, "designed_sequences")
        return designs

    def _build_provenance(self, backbone: Protein | str | Path, *, chain_id: str) -> Provenance:
        """Construct the Provenance for a generate() call.

        Pure function of inputs + constructor parameters — used as
        the cache key. Note: ``device`` is the user-specified value,
        not ``_effective_device()``, to keep the cache key stable
        without needing to load the model.
        """
        backbone_ref = self._provenance_ref(backbone)
        parent = (
            backbone.metadata.get(mk.PROVENANCE)
            if isinstance(backbone, Protein)
            and isinstance(backbone.metadata.get(mk.PROVENANCE), Provenance)
            else None
        )
        return Provenance.from_engine(
            engine="ESM-IF1",
            parameters={
                "model_name": self.model_name,
                "device": self.device,
                "num_seqs": self.num_seqs,
                "temperature": self.temperature,
                "score_sequences": self.score_sequences,
                "compute_recovery": self.compute_recovery,
                "chain_id": chain_id,
                "seed": self.seed,
            },
            inputs={"backbone": backbone_ref},
            parent=parent,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load the model + alphabet if not already in memory.

        The weights download from S3 on first use (~145 MB). The
        ``fair-esm`` package keeps a cache under
        ``~/.cache/torch/hub`` so subsequent calls are fast.
        """
        if self._model is not None and self._alphabet is not None:
            return

        # Lazy import. fair-esm + torch + torch-geometric are
        # heavy and we don't want them required for construction.
        try:
            import esm
            import torch
        except ImportError as e:
            raise GenerativeEngineNotInstalledError(
                "ESM-IF1 requires the fair-esm package (and torch + "
                "torch-geometric as transitive deps).\n"
                "Install with: pip install 'molforge[ml]' "
                "(which pulls fair-esm), then separately "
                "`pip install torch-geometric` for the GVP-GNN layers.\n"
                "See https://github.com/facebookresearch/esm for the "
                "ESM-IF1 environment setup notes."
            ) from e

        if self.seed is not None:
            torch.manual_seed(self.seed)

        loader = getattr(esm.pretrained, self.model_name, None)
        if loader is None:
            raise GenerativeEngineNotInstalledError(
                f"ESM-IF1 model {self.model_name!r} is not available "
                "in this fair-esm install. The standard checkpoint is "
                "'esm_if1_gvp4_t16_142M_UR50'."
            )

        model, alphabet = loader()
        model = model.eval()
        if self.device is not None:
            model = model.to(self.device)
        elif torch.cuda.is_available():
            model = model.to("cuda")

        self._model = model
        self._alphabet = alphabet

    # ------------------------------------------------------------------
    # Coordinate extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _materialise_backbone(backbone: Protein | str | Path, tmp_dir: Path) -> Path:
        """Return a filesystem path to the backbone, writing a temp
        PDB if given a Protein object.

        ESM-IF1's ``load_coords`` reads from a file path, so we go
        through a temp PDB rather than reaching into the Protein's
        atom array directly. This is slightly slower than a direct
        coord extraction but uses the upstream's own loader which
        handles all the edge cases (residue numbering gaps, chain
        breaks, alternate locations, etc.) the way ESM-IF1 expects.
        """
        if isinstance(backbone, Protein):
            from molforge.io import save

            path = tmp_dir / "backbone.pdb"
            save(backbone, path)
            return path
        return Path(backbone)

    def _load_coords(self, pdb_path: Path, chain_id: str) -> tuple[Any, str]:
        """Read coords + native sequence via ESM-IF1's own loader."""
        import esm.inverse_folding

        coords, native_seq = esm.inverse_folding.util.load_coords(str(pdb_path), chain_id)
        return coords, native_seq

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_designs(self, coords: Any, native_seq: str) -> list[DesignedSequence]:
        """Sample, optionally score and recovery-compute, then wrap
        each sample in a :class:`DesignedSequence`.

        Separated from :meth:`generate` so tests can drive this
        seam with mocked coords / native_seq without needing the
        real model to run.
        """
        designs: list[DesignedSequence] = []
        for i in range(self.num_seqs):
            sampled = self._sample_one(coords)

            if self.score_sequences:
                ll_fullseq = self._score_one(coords, sampled)
                # Negative log-likelihood: lower = better, matching
                # molforge's DesignedSequence.score convention.
                score = -float(ll_fullseq)
            else:
                score = 0.0

            recovery = (
                _compute_recovery(sampled, native_seq)
                if self.compute_recovery and native_seq
                else None
            )

            designs.append(
                DesignedSequence(
                    sequence=sampled,
                    score=score,
                    recovery=recovery,
                    metadata={
                        "engine": "ESM-IF1",
                        "model_name": self.model_name,
                        "temperature": self.temperature,
                        "sample_index": i,
                    },
                )
            )
        return designs

    def _sample_one(self, coords: Any) -> str:
        """Sample a single sequence from the loaded model.

        Single-point seam to ``esm.inverse_folding.util.sample_sequence``.
        Tests patch this method on the engine to drive
        :meth:`_sample_designs` without needing the real model.
        """
        import esm.inverse_folding

        result: str = esm.inverse_folding.util.sample_sequence(
            self._model,
            coords,
            temperature=self.temperature,
        )
        return result

    def _score_one(self, coords: Any, sequence: str) -> float:
        """Score a single sequence's average log-likelihood.

        Single-point seam to ``esm.inverse_folding.util.score_sequence``,
        returning just the ``ll_fullseq`` value (the average
        log-likelihood across all positions). The second return value
        — ``ll_withcoord``, masking out the coord-conditioned
        positions — is discarded; users who need it can call the
        upstream function directly.
        """
        import esm.inverse_folding

        ll_fullseq, _ = esm.inverse_folding.util.score_sequence(
            self._model, self._alphabet, coords, sequence
        )
        return float(ll_fullseq)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _effective_device(self) -> str:
        """Return a JSON-safe device string for provenance.

        ``device`` may be None when construction was lazy; the actual
        choice is resolved at load time. This helper avoids leaking
        ``None`` into the provenance.parameters when we know what
        was actually used.
        """
        if self.device is not None:
            return self.device
        # Best-effort: if we've loaded the model, report what it's on.
        if self._model is not None:
            try:
                next_param = next(self._model.parameters(), None)
                if next_param is not None:
                    return str(next_param.device)
            except Exception:
                pass
        return "auto"

    @staticmethod
    def _provenance_ref(backbone: Protein | str | Path) -> str:
        """Return a JSON-safe string identifier for the backbone input."""
        if isinstance(backbone, Protein):
            return backbone.name or "<Protein>"
        return str(backbone)


def _compute_recovery(designed: str, native: str) -> float:
    """Fraction of positions where designed matches native.

    Handles length mismatches (which shouldn't happen for proper
    inverse-folding outputs, but defensive coding here) by comparing
    only the overlapping prefix.
    """
    if not designed or not native:
        return 0.0
    n = min(len(designed), len(native))
    if n == 0:
        return 0.0
    matches = sum(1 for i in range(n) if designed[i] == native[i])
    return matches / n
