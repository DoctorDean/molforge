"""Chai-1 wrapper.

`Chai-1 <https://github.com/chaidiscovery/chai-lab>`_ (Chai Discovery,
2024) is an open-weights re-implementation of AlphaFold-3-style
biomolecular structure prediction. Like :class:`Boltz`, it predicts
proteins, nucleic acids (DNA/RNA), and small-molecule ligands (via
SMILES) in a single forward pass, and reaches AlphaFold-3-level
accuracy on the PoseBusters protein-ligand benchmark.

The two engines are direct competitors and natural cross-checks:
different teams, different training pipelines, similar overall
capabilities. Running both on a hard target and taking the
intersection (or comparing ranking scores) is a common practice
when single-engine confidence is borderline.

Unlike :class:`Boltz` (which is invoked through a CLI subprocess),
Chai-1 ships a clean Python entry point — ``chai_lab.chai1.run_inference``
— so this wrapper imports and calls it directly. No subprocess
plumbing, no YAML hand-rolling; just FASTA in, structures out.

Installation::

    pip install chai_lab

Chai-1 requires Linux, Python 3.10+, and a GPU with CUDA + bfloat16
support (A100/H100/L40S recommended; A10/A30 work for smaller
complexes; RTX 4090 also works). CPU-only inference is not
supported by the upstream package.

Weights download automatically on first call into
``~/.cache/chai/downloads`` (overridable via the
``CHAI_DOWNLOADS_DIR`` env var). Expect a slow first call (~3 GB
download); subsequent calls reuse the local cache.

Output convention
-----------------

Chai-1 always emits 5 diffusion samples per call (this is hard-coded
in the upstream and is *not* configurable in v1). Each sample is a
full mmCIF with per-residue pLDDT in the B-factor column, paired
with a ``scores.model_idx_N.npz`` archive containing the headline
ranking metrics (``aggregate_score``, ``ptm``, ``iptm``,
``per_chain_ptm``, ``per_chain_pair_iptm``, ``has_inter_chain_clashes``).

:meth:`Chai1.predict` picks the sample with the highest
``aggregate_score`` as the canonical returned :class:`Protein`, which
is what most callers want. The other four are not thrown away:
:meth:`Chai1.predict_samples` and :meth:`Chai1.predict_complex_samples`
return all five, ranked, from the same single inference — the GPU
time is spent regardless, so an ensemble costs no more than a
single structure. Both shapes share a cache, so asking for one
after the other doesn't re-run the model.

Multi-component scope
---------------------

This v1 wrapper mirrors :class:`Boltz`'s v1: single protein chain
only. Chai-1 natively supports multi-component complexes (the
FASTA header is ``>protein|name=...`` / ``>ligand|name=...`` /
``>dna|name=...`` / ``>rna|name=...``) and that capability will
land alongside multi-component support in the other folding
wrappers in a future commit.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from molforge.cache import get_default_cache
from molforge.core import metadata_keys as mk
from molforge.core.provenance import Provenance
from molforge.folding import ComplexSpec, Entity
from molforge.wrappers._versions import engine_version
from molforge.wrappers.folding._base import (
    FoldingEngine,
    FoldingEngineNotInstalledError,
    _reject_unknown_kwargs,
    _validate_sequence,
)

if TYPE_CHECKING:
    from molforge.cache import Cache
    from molforge.core import Protein


# Chai-1 hard-codes 5 diffusion samples per call. Documented here
# so the test that uses this constant doesn't have to re-state the
# magic number.
_CHAI_NUM_SAMPLES = 5


# Score keys we extract from each sample's NPZ. We don't enforce
# their presence — Chai's API may evolve — and missing keys produce
# ``None`` in metadata rather than crashing.
_HEADLINE_SCORE_KEYS = (
    "aggregate_score",
    "ptm",
    "iptm",
)


class Chai1(FoldingEngine):
    """Wrapper around the Chai-1 biomolecular prediction model.

    Args:
        device: Torch device string (``"cuda"`` / ``"cuda:0"``).
            Defaults to ``None`` which lets the wrapper auto-detect
            CUDA on first call. Chai-1 doesn't support CPU inference;
            passing ``"cpu"`` will surface as a Chai-side error.
        use_msa_server: If ``True``, Chai-1 hits the ColabFold MMseqs2
            server for MSA generation. Substantially improves accuracy
            for natural proteins; adds network round-trips and depends
            on a shared community resource. Defaults to ``False`` (the
            Chai-lab default), which uses MSA-free inference — faster,
            no network, lower accuracy.
        msa_server_url: Override the MSA server URL. Only used when
            ``use_msa_server=True``. ``None`` uses Chai-lab's default
            (the ColabFold server).
        num_trunk_recycles: Trunk-recycling rounds. ``None`` uses
            Chai-lab's default (3). Higher = slower, marginally
            better accuracy.
        num_diffn_timesteps: Diffusion denoising steps. ``None`` uses
            Chai-lab's default (200). Higher = slower, marginally
            better structure quality.
        seed: PyTorch random seed for reproducibility. ``None`` leaves
            the global seed untouched. Note: even with a seed, exact
            reproducibility depends on CUDA non-determinism settings;
            small numerical drift across runs is expected.
        cache_dir: Override the Chai-lab weights cache directory by
            setting the ``CHAI_DOWNLOADS_DIR`` env var when calling.
            ``None`` uses the upstream default (typically inside the
            ``chai_lab`` package install).
    """

    name = "Chai-1"

    def __init__(
        self,
        *,
        device: str | None = None,
        use_msa_server: bool = False,
        msa_server_url: str | None = None,
        num_trunk_recycles: int | None = None,
        num_diffn_timesteps: int | None = None,
        seed: int | None = None,
        cache_dir: str | None = None,
    ) -> None:
        if num_trunk_recycles is not None and num_trunk_recycles < 1:
            raise ValueError(f"num_trunk_recycles must be >= 1, got {num_trunk_recycles}")
        if num_diffn_timesteps is not None and num_diffn_timesteps < 1:
            raise ValueError(f"num_diffn_timesteps must be >= 1, got {num_diffn_timesteps}")
        self.device = device
        self.use_msa_server = use_msa_server
        self.msa_server_url = msa_server_url
        self.num_trunk_recycles = num_trunk_recycles
        self.num_diffn_timesteps = num_diffn_timesteps
        self.seed = seed
        self.cache_dir = cache_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, sequence: str, **kwargs: object) -> Protein:
        """Fold a single sequence into a :class:`Protein` via Chai-1.

        Args:
            sequence: One-letter amino-acid sequence.
            **kwargs: None accepted. Chai-1's options — including
                ``seed`` — are set on the constructor; any keyword
                passed here raises :class:`TypeError` rather than
                being silently ignored.

        Returns:
            The best of Chai-1's 5 diffusion samples (by
            ``aggregate_score``), as a :class:`Protein` whose
            ``metadata`` includes:

            - ``engine = "Chai-1"``
            - ``source_sequence``: the input sequence
            - ``confidence_per_residue``: ``(L,)`` float32 pLDDT
                (extracted from the CIF's B-factor column)
            - ``mean_confidence``: scalar mean pLDDT
            - ``ptm`` / ``iptm`` / ``aggregate_score``: Chai-1's
                ranking metrics for the chosen sample
            - ``best_sample_index``: 0–4 index of the chosen sample
            - ``per_sample_scores``: list of 5 dicts (one per
                diffusion sample, in Chai-1's own sample order) with
                ``aggregate_score``, ``ptm``, ``iptm`` — for inspecting
                ranking spread. For the non-best *structures*, use
                :meth:`predict_samples`.
            - ``provenance``: :class:`Provenance` capturing all
                constructor kwargs

        Raises:
            FoldingEngineNotInstalledError: If ``chai_lab`` isn't
                installed or fails to import.
            RuntimeError: If Chai-1 produces no parseable output.
            TypeError: If any keyword argument is passed.
        """
        _reject_unknown_kwargs(
            kwargs,
            engine="Chai1",
            method="predict",
            constructor_example="Chai1(seed=7)",
        )
        sequence = _validate_sequence(sequence)
        # Delegate to the spec-based code path. Multi-component
        # machinery is the canonical implementation; single-sequence
        # is a degenerate special case.
        spec = ComplexSpec.from_protein(sequence)
        return self._predict_spec(spec, single_sequence=sequence)

    def predict_complex(self, spec: ComplexSpec, **kwargs: object) -> Protein:
        """Fold a multi-component complex via Chai-1.

        The headline AlphaFold-3-style capability: predict the
        structure of multiple protein chains, DNA/RNA, and/or
        small-molecule ligands in a single forward pass. See
        :class:`molforge.folding.ComplexSpec` for input shape.

        Args:
            spec: A :class:`ComplexSpec` describing the entities to
                fold. The returned :class:`Protein` has one chain in
                its ``atom_array`` per polymer entity (or per copy,
                for homo-oligomers). Ligand atoms appear as
                hetero-atoms with chain IDs assigned by the spec.
            **kwargs: None accepted. Chai-1's options — including
                ``seed`` — are set on the constructor; any keyword
                passed here raises :class:`TypeError` rather than
                being silently ignored.

        Returns:
            A :class:`Protein` with multi-chain ``atom_array`` and
            the metadata documented in :meth:`predict`, plus:

            - ``metadata["complex_spec"]``: the :class:`ComplexSpec`
                passed in, for traceability.
            - ``metadata["per_chain_ptm"]``: per-chain pTM values
                from Chai's scores NPZ (when present).
            - ``metadata["per_chain_pair_iptm"]``: pairwise
                interface pTM matrix (when present).

            Only the best-scoring of Chai-1's five diffusion samples;
            :meth:`predict_complex_samples` returns all five.

        Raises:
            FoldingEngineNotInstalledError: If ``chai_lab`` isn't
                installed.
            RuntimeError: If Chai-1 produces no parseable output.
            TypeError: If any keyword argument is passed.

        Examples:
            Protein-ligand complex::

                from molforge.folding import ComplexSpec
                from molforge.wrappers.folding import Chai1

                spec = ComplexSpec.protein_ligand(
                    protein_sequence="MVTPEG...",
                    ligand_smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
                )
                complex_struct = Chai1().predict_complex(spec)
        """
        _reject_unknown_kwargs(
            kwargs,
            engine="Chai1",
            method="predict_complex",
            constructor_example="Chai1(seed=7)",
        )
        return self._predict_spec(spec, single_sequence=None)

    def predict_samples(self, sequence: str) -> list[Protein]:
        """Fold a sequence and return *every* diffusion sample, best first.

        Chai-1 always computes five diffusion samples per call;
        :meth:`predict` returns the highest-scoring one and drops the
        other four structures. They cost nothing extra — the GPU time is
        already spent — so for ensemble, occupancy, or pose-diversity
        work this is five times the usable output per run.

        Args:
            sequence: One-letter amino-acid sequence.

        Returns:
            The run's :class:`Protein` samples ordered by
            ``aggregate_score``, best first, so ``predict_samples(seq)[0]``
            is what :meth:`predict` returns. Each carries the metadata
            documented in :meth:`predict` — with its *own* scores — plus:

            - ``metadata["sample_index"]``: 0-4, Chai-1's own index for
                this sample.
            - ``metadata["sample_rank"]``: 0-4, its position in the
                returned ranking.

        Raises:
            FoldingEngineNotInstalledError: If ``chai_lab`` isn't
                installed or fails to import.
            RuntimeError: If Chai-1 produces no parseable output.

        Examples:
            A conformational ensemble from one call::

                from molforge.wrappers.folding import Chai1

                samples = Chai1().predict_samples("MVTPEG...")
                len(samples)  # 5
                spread = [s.metadata["mean_confidence"] for s in samples]
        """
        sequence = _validate_sequence(sequence)
        spec = ComplexSpec.from_protein(sequence)
        return self._predict_spec_samples(spec, single_sequence=sequence)

    def predict_complex_samples(self, spec: ComplexSpec) -> list[Protein]:
        """Fold a complex and return *every* diffusion sample, best first.

        The :meth:`predict_complex` counterpart of :meth:`predict_samples`.
        For ligand work this is the interesting one: five poses per call
        rather than one, at no extra inference cost.

        Args:
            spec: A :class:`ComplexSpec` describing the entities to fold.

        Returns:
            The run's :class:`Protein` samples ordered by
            ``aggregate_score``, best first, so
            ``predict_complex_samples(spec)[0]`` is what
            :meth:`predict_complex` returns. Metadata is as documented in
            :meth:`predict_complex`, plus the ``sample_index`` and
            ``sample_rank`` keys from :meth:`predict_samples`.

        Raises:
            FoldingEngineNotInstalledError: If ``chai_lab`` isn't installed.
            RuntimeError: If Chai-1 produces no parseable output.

        Examples:
            Five ligand poses from one call::

                from molforge.folding import ComplexSpec
                from molforge.wrappers.folding import Chai1

                spec = ComplexSpec.protein_ligand(
                    protein_sequence="MVTPEG...",
                    ligand_smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
                )
                poses = Chai1().predict_complex_samples(spec)
        """
        return self._predict_spec_samples(spec, single_sequence=None)

    # ------------------------------------------------------------------
    # Local-execution path (testable seam)
    # ------------------------------------------------------------------
    def _predict_spec(
        self,
        spec: ComplexSpec,
        *,
        single_sequence: str | None,
    ) -> Protein:
        """The shared spec-based execution path.

        Both :meth:`predict` and :meth:`predict_complex` route through
        here. ``single_sequence`` is non-None when the caller was the
        single-sequence ``predict()`` path; that lets us preserve the
        existing ``metadata["source_sequence"]`` contract.
        """
        # Cache lookup before chai_lab GPU inference.
        provenance = self._build_provenance(spec, single_sequence=single_sequence)
        cache = get_default_cache()
        cached: Protein | None = cache.get(provenance, "protein")
        if cached is not None:
            return cached

        samples = self._run_and_collect(spec)

        result = self._parse_outputs(
            samples=samples,
            sequence=single_sequence,
            spec=spec,
            provenance=provenance,
        )
        cache.put(provenance, result, "protein")
        self._bank_samples(samples, spec=spec, single_sequence=single_sequence, cache=cache)
        return result

    def _predict_spec_samples(
        self,
        spec: ComplexSpec,
        *,
        single_sequence: str | None,
    ) -> list[Protein]:
        """The shared spec-based execution path, keeping every sample.

        Same inference as :meth:`_predict_spec`; the difference is only
        what survives it. Cached under its own Provenance ``operation``
        so the ensemble and the single best result don't share an entry.
        """
        provenance = self._build_provenance(spec, single_sequence=single_sequence, all_samples=True)
        cache = get_default_cache()
        cached: list[Protein] | None = cache.get(provenance, "protein_list")
        if cached is not None:
            return cached

        samples = self._run_and_collect(spec)

        results = self._parse_all_outputs(
            samples=samples,
            sequence=single_sequence,
            spec=spec,
            provenance=provenance,
        )
        cache.put(provenance, results, "protein_list")
        self._bank_best(samples, spec=spec, single_sequence=single_sequence, cache=cache)
        return results

    def _run_and_collect(self, spec: ComplexSpec) -> list[dict[str, Any]]:
        """Run chai_lab on ``spec`` and return its samples, tempdir cleaned up."""
        with tempfile.TemporaryDirectory(prefix="molforge_chai1_") as td:
            tmpdir = Path(td)
            fasta_path = tmpdir / "input.fasta"
            output_dir = tmpdir / "out"
            output_dir.mkdir()

            fasta_path.write_text(
                self._build_fasta_from_spec(spec),
                encoding="utf-8",
            )

            self._run_inference(fasta_path, output_dir)
            return self._collect_samples(output_dir)

    def _bank_samples(
        self,
        samples: list[dict[str, Any]],
        *,
        spec: ComplexSpec,
        single_sequence: str | None,
        cache: Cache,
    ) -> None:
        """Cache the full ensemble that a :meth:`predict` call also produced.

        The GPU time is spent either way — Chai-1 always runs five
        diffusion samples — so banking them means a later
        :meth:`predict_samples` on the same input is a cache hit rather
        than a second identical run. Parsing four extra CIFs is free next
        to the inference that made them.
        """
        provenance = self._build_provenance(spec, single_sequence=single_sequence, all_samples=True)
        cache.put(
            provenance,
            self._parse_all_outputs(
                samples=samples, sequence=single_sequence, spec=spec, provenance=provenance
            ),
            "protein_list",
        )

    def _bank_best(
        self,
        samples: list[dict[str, Any]],
        *,
        spec: ComplexSpec,
        single_sequence: str | None,
        cache: Cache,
    ) -> None:
        """The mirror of :meth:`_bank_samples`: an ensemble run answers a
        plain :meth:`predict` too, so a later one shouldn't re-run."""
        provenance = self._build_provenance(spec, single_sequence=single_sequence)
        cache.put(
            provenance,
            self._parse_outputs(
                samples=samples, sequence=single_sequence, spec=spec, provenance=provenance
            ),
            "protein",
        )

    def _build_provenance(
        self,
        spec: ComplexSpec,
        *,
        single_sequence: str | None,
        all_samples: bool = False,
    ) -> Provenance:
        """Construct the Provenance for a predict / predict_complex call.

        Pure function of inputs + constructor parameters — used as
        the cache key. Does not touch chai_lab.

        ``all_samples`` marks the full-ensemble variant. It's recorded
        in the *parameters*, not only in ``operation``, because the
        cache key hashes parameters and not the operation — the same
        trick :meth:`Boltz._build_provenance` uses to keep an affinity
        run from colliding with a plain fold. The key is only added
        when set, so a plain ``predict()`` keeps the Provenance (and
        therefore the cache entry) it has always had.
        """
        prov_inputs: dict[str, object]
        if single_sequence is not None:
            prov_inputs = {"sequence": single_sequence}
        else:
            prov_inputs = {"complex_spec": _serialize_spec_for_provenance(spec)}
        parameters: dict[str, object] = {
            "device": self.device,
            "use_msa_server": self.use_msa_server,
            "msa_server_url": self.msa_server_url,
            "num_trunk_recycles": self.num_trunk_recycles,
            "num_diffn_timesteps": self.num_diffn_timesteps,
            "seed": self.seed,
            "cache_dir": self.cache_dir,
        }
        if all_samples:
            parameters["return_all_samples"] = True
        return Provenance.from_engine(
            engine="Chai-1",
            operation="predict_samples" if all_samples else "predict",
            engine_version=engine_version("chai_lab"),
            parameters=parameters,
            inputs=prov_inputs,
        )

    def _run_local(self, sequence: str) -> Protein:
        """Single-sequence local execution (legacy API).

        Preserved for any external callers of the private seam. New
        code should use :meth:`_predict_spec` instead.
        """
        return self._predict_spec(ComplexSpec.from_protein(sequence), single_sequence=sequence)

    # ------------------------------------------------------------------
    # Process plumbing (each step a testable seam)
    # ------------------------------------------------------------------
    def _build_fasta(self, sequence: str, *, name: str) -> str:
        """Single-sequence FASTA builder (legacy API).

        Preserved as a thin wrapper around
        :meth:`_build_fasta_from_spec` so existing tests of the
        single-sequence path keep working.
        """
        return self._build_fasta_from_spec(
            ComplexSpec(entities=(Entity(kind="protein", sequence=sequence, name=name),))
        )

    def _build_fasta_from_spec(self, spec: ComplexSpec) -> str:
        """Construct the Chai-1 FASTA input for a ComplexSpec.

        Chai-1's FASTA uses typed headers like ``>protein|name=foo``,
        ``>ligand|name=...``, ``>dna|name=...``, ``>rna|name=...``.
        The type prefix is what tells Chai whether each record is a
        polymer or a ligand.

        Unlike Boltz's YAML which can express multi-copy with a list-
        of-IDs shape, Chai-1's FASTA needs each copy as a separate
        record. A homodimer becomes two ``>protein|name=...`` records
        with the same sequence and distinct ``name=`` suffixes. We
        embed the chain ID into the ``name=`` field so users can map
        FASTA records back to chains in the output structure.
        """
        records: list[str] = []
        chain_ids_per_entity = spec.assigned_chain_ids()

        for entity, chain_ids in zip(spec.entities, chain_ids_per_entity, strict=True):
            records.extend(_chai_fasta_entity(entity, chain_ids))

        return "".join(records)

    def _run_inference(self, fasta_path: Path, output_dir: Path) -> None:
        """Single seam to ``chai_lab.chai1.run_inference``.

        Tests patch this method to drive the full :meth:`_run_local`
        pipeline without needing ``chai_lab`` or a GPU. Returns
        nothing; the upstream writes ``pred.model_idx_N.cif`` and
        ``scores.model_idx_N.npz`` for ``N`` in 0..4 directly under
        ``output_dir``.
        """
        try:
            import torch
            from chai_lab.chai1 import run_inference
        except ImportError as e:
            raise FoldingEngineNotInstalledError(
                "Chai-1 requires the chai_lab package. Install with:\n"
                "    pip install chai_lab\n"
                "Note: Chai-1 requires Linux, Python 3.10+, and a CUDA-"
                "capable GPU with bfloat16 support. CPU-only inference "
                "is not supported by upstream.\n"
                "See https://github.com/chaidiscovery/chai-lab for "
                "setup notes."
            ) from e

        # Resolve device. Chai-1 demands a torch.device; we accept a
        # string in our constructor and convert here. None means
        # auto-detect at call time (cuda if available).
        if self.device is not None:
            device = torch.device(self.device)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            # Chai-1 will reject this, but raising upfront gives a
            # cleaner error than the upstream torch error.
            raise FoldingEngineNotInstalledError(
                "Chai-1 requires a CUDA GPU but no CUDA device was "
                "detected. Run on a GPU machine or pass an explicit "
                "device= override."
            )

        if self.seed is not None:
            torch.manual_seed(self.seed)

        # Build the kwargs for run_inference. None-valued constructor
        # params are omitted so chai_lab uses its own defaults.
        kwargs: dict[str, Any] = {
            "fasta_file": fasta_path,
            "output_dir": output_dir,
            "device": device,
            "use_msa_server": self.use_msa_server,
        }
        if self.msa_server_url is not None:
            kwargs["msa_server_url"] = self.msa_server_url
        if self.num_trunk_recycles is not None:
            kwargs["num_trunk_recycles"] = self.num_trunk_recycles
        if self.num_diffn_timesteps is not None:
            kwargs["num_diffn_timesteps"] = self.num_diffn_timesteps
        if self.seed is not None:
            kwargs["seed"] = self.seed

        run_inference(**kwargs)

    def _collect_samples(self, output_dir: Path) -> list[dict[str, Any]]:
        """Gather Chai-1's 5 diffusion samples from the output dir.

        Returns a list of dicts, each with keys:

        - ``index``: 0..4
        - ``cif_text``: the mmCIF text for ``pred.model_idx_N.cif``
        - ``scores``: dict parsed from ``scores.model_idx_N.npz``

        We read the CIF into memory (rather than holding onto the
        path) so the caller can clean up the tempdir before parsing.
        Missing CIF or NPZ files for any sample are surfaced as
        :class:`RuntimeError` — Chai-1 producing fewer than 5 samples
        is a sign something went wrong in the inference and the
        result shouldn't be trusted.
        """
        samples: list[dict[str, Any]] = []
        for idx in range(_CHAI_NUM_SAMPLES):
            cif_path = output_dir / f"pred.model_idx_{idx}.cif"
            scores_path = output_dir / f"scores.model_idx_{idx}.npz"
            if not cif_path.is_file():
                raise RuntimeError(
                    f"Chai-1 did not produce {cif_path.name}. Check the chai_lab output for errors."
                )
            if not scores_path.is_file():
                raise RuntimeError(
                    f"Chai-1 did not produce {scores_path.name}. "
                    f"Check the chai_lab output for errors."
                )
            scores = _load_scores_npz(scores_path)
            samples.append(
                {
                    "index": idx,
                    "cif_text": cif_path.read_text(encoding="utf-8"),
                    "scores": scores,
                }
            )

        return samples

    def _parse_outputs(
        self,
        *,
        samples: list[dict[str, Any]],
        sequence: str | None,
        spec: ComplexSpec | None = None,
        provenance: Provenance | None = None,
    ) -> Protein:
        """Pick the best sample by ``aggregate_score`` and build the
        canonical :class:`Protein`.

        The non-best samples' headline scores are preserved in
        ``metadata["per_sample_scores"]`` so users wanting to inspect
        ranking spread can do so without re-running Chai-1;
        :meth:`_parse_all_outputs` keeps their structures too.

        Args:
            samples: Output of :meth:`_collect_samples`.
            sequence: The original single-sequence input, when the
                caller is the single-sequence :meth:`predict` path.
                ``None`` for multi-component :meth:`predict_complex`
                calls (the spec carries that information instead).
            spec: The :class:`ComplexSpec` used for this prediction.
                Always set; for single-sequence predict() it's the
                trivial single-entity spec.
        """
        ranked = _rank_samples(samples)
        return self._protein_from_sample(
            ranked[0],
            ranked=ranked,
            sequence=sequence,
            spec=spec,
            provenance=provenance,
            rank=0,
        )

    def _parse_all_outputs(
        self,
        *,
        samples: list[dict[str, Any]],
        sequence: str | None,
        spec: ComplexSpec | None = None,
        provenance: Provenance | None = None,
    ) -> list[Protein]:
        """Build a :class:`Protein` for *every* sample, best first.

        Same ranking as :meth:`_parse_outputs`, so element 0 is the
        structure that method would have returned on its own; the rest
        are the samples Chai-1 computed and molforge used to throw away.

        Args:
            samples: Output of :meth:`_collect_samples`.
            sequence: The original single-sequence input, when the caller
                is the single-sequence path. ``None`` for complexes.
            spec: The :class:`ComplexSpec` used for this prediction.
            provenance: The prebuilt Provenance for the call.
        """
        ranked = _rank_samples(samples)
        return [
            self._protein_from_sample(
                sample,
                ranked=ranked,
                sequence=sequence,
                spec=spec,
                provenance=provenance,
                rank=rank,
            )
            for rank, sample in enumerate(ranked)
        ]

    def _protein_from_sample(
        self,
        sample: dict[str, Any],
        *,
        ranked: list[dict[str, Any]],
        sequence: str | None,
        spec: ComplexSpec | None,
        provenance: Provenance | None,
        rank: int,
    ) -> Protein:
        """Build one sample's :class:`Protein`, with the run's metadata attached.

        ``ranked`` is the whole run, so every returned structure carries
        the same run-level view (``per_sample_scores``,
        ``best_sample_index``) alongside its own scores and position.
        """
        # Read the CIF through molforge's own reader to get a proper
        # Protein with AtomArray, then layer on Chai's confidence +
        # ranking metadata. We use read_cif_string (not read_cif) so
        # the parsing works on the in-memory CIF text captured during
        # _collect_samples — the upstream tempdir is gone by now.
        from molforge.io.mmcif import read_cif_string

        protein = read_cif_string(sample["cif_text"])

        # pLDDT is in the CIF's B-factor column (AlphaFold convention,
        # which Chai-1 follows). Extract per-residue mean over CA atoms.
        confidence_per_residue = _per_residue_plddt_from_cif(protein)
        mean_confidence = (
            float(confidence_per_residue.mean()) if confidence_per_residue.size > 0 else 0.0
        )

        # Per-chain confidence stats. Chai-1's NPZ writes
        # ``per_chain_ptm`` and ``per_chain_pair_iptm`` for multi-chain
        # predictions; pass through verbatim when present.
        per_chain_ptm = sample["scores"].get("per_chain_ptm")
        per_chain_pair_iptm = sample["scores"].get("per_chain_pair_iptm")

        # Provenance: use the prebuilt one when supplied (the normal
        # _predict_spec path), otherwise build a fresh one (legacy
        # test path that calls _parse_outputs directly).
        if provenance is None:
            if spec is None and sequence is not None:
                spec = ComplexSpec.from_protein(sequence)
            assert spec is not None, "_parse_outputs needs spec or sequence"
            provenance = self._build_provenance(spec, single_sequence=sequence)

        metadata_update: dict[str, object] = {
            "engine": "Chai-1",
            mk.CONFIDENCE_PER_RESIDUE: confidence_per_residue,
            mk.MEAN_CONFIDENCE: mean_confidence,
            "aggregate_score": sample["scores"].get("aggregate_score"),
            "ptm": sample["scores"].get("ptm"),
            "iptm": sample["scores"].get("iptm"),
            "sample_index": sample["index"],
            "sample_rank": rank,
            "best_sample_index": ranked[0]["index"],
            # Indexed by Chai-1's own sample index, not by rank: callers
            # read per_sample_scores[i] as "sample i", so this stays in
            # Chai's order even though the structures come back ranked.
            "per_sample_scores": [
                {key: s["scores"].get(key) for key in _HEADLINE_SCORE_KEYS}
                for s in sorted(ranked, key=lambda s: s["index"])
            ],
            mk.PROVENANCE: provenance,
        }
        # Preserve "source_sequence" only for single-sequence calls.
        # For complexes, surface the spec instead.
        if sequence is not None:
            metadata_update["source_sequence"] = sequence
        if spec is not None and sequence is None:
            metadata_update["complex_spec"] = spec
        if per_chain_ptm is not None:
            metadata_update["per_chain_ptm"] = per_chain_ptm
        if per_chain_pair_iptm is not None:
            metadata_update["per_chain_pair_iptm"] = per_chain_pair_iptm

        protein.metadata.update(metadata_update)
        return protein


# ---------------------------------------------------------------------
# NPZ + CIF helpers (module-level so tests can exercise them directly)
# ---------------------------------------------------------------------


def _rank_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order a run's samples best-first by ``aggregate_score``.

    Samples missing an ``aggregate_score`` (unusual) sort with ``-inf``
    so they lose every tiebreak. The sort is stable, so equal scores
    keep Chai-1's own sample order.

    Raises:
        RuntimeError: If ``samples`` is empty.
    """
    if not samples:
        raise RuntimeError("Chai-1 produced no samples — nothing to parse.")

    def _aggregate(sample: dict[str, Any]) -> float:
        v = sample["scores"].get("aggregate_score")
        return float(v) if v is not None else float("-inf")

    return sorted(samples, key=_aggregate, reverse=True)


def _load_scores_npz(path: Path) -> dict[str, Any]:
    """Load a Chai-1 scores NPZ into a JSON-shaped dict.

    Chai-1 stores its ranking outputs as a numpy archive with the
    same keys you'd find in the equivalent JSON file
    (``aggregate_score``, ``ptm``, ``iptm``, ``per_chain_ptm``,
    ``per_chain_pair_iptm``, ``has_inter_chain_clashes``,
    ``chain_intra_clashes``, ``chain_chain_inter_clashes``).

    We extract every key into a Python-native dict so the calling
    code doesn't have to deal with 0-d numpy arrays.
    """
    out: dict[str, Any] = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            arr = archive[key]
            if arr.shape == ():
                # 0-d arrays unwrap to scalars; .item() converts
                # numpy types to native Python (float / int / bool).
                out[key] = arr.item()
            else:
                # Multi-dim arrays stay as numpy for users who
                # want them; downstream code can convert as needed.
                out[key] = arr
    return out


def _per_residue_plddt_from_cif(protein: Protein) -> np.ndarray:
    """Extract per-residue pLDDT from a Protein whose B-factor column
    carries Chai-1's pLDDT values (AlphaFold convention).

    Returns a ``(n_residues,)`` float32 array. Uses CA atom b-factors
    where available, falling back to per-residue mean over all atoms
    in residues lacking a CA (which can happen for non-standard
    residues, but normally there's a CA per residue).
    """
    arr = protein.atom_array
    if arr.n_atoms == 0:
        return np.array([], dtype=np.float32)

    # Find CA atoms per residue. The atom_name array is a NumPy
    # str array; ``== "CA"`` gives a bool mask we can use to slice.
    is_ca = arr.atom_name == "CA"
    if not is_ca.any():
        # No CAs (e.g. all-ligand structure or unusual chemistry).
        # Fall back to per-residue mean B-factor.
        return _residue_mean_bfactor(arr).astype(np.float32)

    # b_factor on CA atoms, in the residue order the CAs appear.
    return np.asarray(arr.b_factor[is_ca], dtype=np.float32)


def _residue_mean_bfactor(arr: Any) -> np.ndarray:
    """Per-residue mean over all atoms. Fallback when no CAs are
    present.

    Groups by ``(chain_id, residue_id, insertion_code)`` since
    residue_id alone isn't unique across chains. Returns one mean
    per residue in the order they appear in the AtomArray.
    """
    # Build a per-atom group id by concatenating chain + residue_id +
    # insertion_code, then take per-group mean using numpy.unique's
    # ``inverse_indices`` machinery.
    chain = arr.chain_id.astype(str)
    resid = arr.residue_id.astype(str)
    icode = arr.insertion_code.astype(str)
    keys = np.array([f"{c}|{r}|{i}" for c, r, i in zip(chain, resid, icode, strict=False)])
    _, first_idx, inverse = np.unique(keys, return_index=True, return_inverse=True)
    # Preserve original residue order using first_idx.
    order = np.argsort(first_idx)
    n_residues = order.size

    sums = np.zeros(n_residues, dtype=np.float64)
    counts = np.zeros(n_residues, dtype=np.int64)
    # Map each atom's inverse index into the order-preserved index.
    remap = np.empty(n_residues, dtype=np.int64)
    remap[order] = np.arange(n_residues)
    per_atom_residue_idx = remap[inverse]
    np.add.at(sums, per_atom_residue_idx, arr.b_factor)
    np.add.at(counts, per_atom_residue_idx, 1)
    return sums / np.maximum(counts, 1)


# ---------------------------------------------------------------------
# Module-level FASTA serialization helpers (testable without a Chai1 instance)
# ---------------------------------------------------------------------


def _chai_fasta_entity(entity: Entity, chain_ids: list[str]) -> list[str]:
    """Render one ComplexSpec.Entity as Chai-1 FASTA records.

    Returns a list of FASTA-record strings (each already
    newline-terminated). Each copy of the entity becomes its own
    FASTA record — Chai-1's FASTA format doesn't have an id-list
    shape like Boltz's YAML.

    The ``name=`` field embeds the chain_id, falling back to the
    user-supplied :attr:`Entity.name`. This is what lets users map
    output chains back to input entities.

    Example output for a homodimer::

        >protein|name=A
        MKQH...
        >protein|name=B
        MKQH...

    Example output for a protein + SMILES ligand::

        >protein|name=A
        MKQH...
        >ligand|name=ligand_B
        CC(=O)OC1=CC=CC=C1C(=O)O
    """
    records: list[str] = []
    base_name = entity.name

    for chain_id in chain_ids:
        # Choose a name suffix. We prefer the chain_id (it's unique
        # and traceable) but respect user-provided entity.name when
        # set on single-copy entities. For multi-copy, the user-
        # supplied name only serves as a base; chain_id distinguishes
        # the copies.
        if base_name is not None and len(chain_ids) == 1:
            record_name = base_name
        elif base_name is not None:
            record_name = f"{base_name}_{chain_id}"
        else:
            # Default: kind + chain_id (e.g. "ligand_B"). For protein,
            # using just the chain_id is more conventional.
            record_name = chain_id if entity.kind == "protein" else f"{entity.kind}_{chain_id}"

        header = f">{entity.kind}|name={record_name}\n"

        if entity.is_polymer:
            body = entity.normalized_sequence() + "\n"
        elif entity.smiles is not None:
            body = entity.smiles + "\n"
        else:
            # CCD code. Chai-1 accepts CCD codes via the ligand
            # entity type using the same record-body convention.
            assert entity.ccd is not None  # validated upstream
            body = entity.ccd + "\n"

        records.append(header + body)

    return records


def _serialize_spec_for_provenance(spec: ComplexSpec | None) -> object:
    """Render a ComplexSpec to a JSON-safe shape for Provenance.inputs.

    Identical contract to the helper in boltz.py: flattens the spec
    into a list of dicts with engine-agnostic keys, so the same
    structure shows up in Provenance regardless of which engine
    produced the prediction. This is what lets users compare Boltz
    and Chai-1 provenance side-by-side.
    """
    if spec is None:
        return None
    entities_payload: list[dict[str, object]] = []
    for entity, chain_ids in zip(spec.entities, spec.assigned_chain_ids(), strict=True):
        payload: dict[str, object] = {
            "kind": entity.kind,
            "chain_ids": chain_ids,
        }
        if entity.is_polymer:
            payload["sequence"] = entity.normalized_sequence()
        elif entity.smiles is not None:
            payload["smiles"] = entity.smiles
        else:
            payload["ccd"] = entity.ccd
        if entity.name is not None:
            payload["name"] = entity.name
        entities_payload.append(payload)
    return {"entities": entities_payload}
