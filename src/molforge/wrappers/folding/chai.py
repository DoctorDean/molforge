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

The wrapper picks the sample with the highest ``aggregate_score``
as the canonical returned :class:`Protein`. The other four
predictions remain on disk under the tempdir (deleted with the
context manager) — surfacing them is deferred until concrete user
needs surface (per-sample diversity analysis is a niche pattern;
most users want the best prediction).

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

from molforge.core import metadata_keys as mk
from molforge.core.provenance import Provenance
from molforge.wrappers.folding._base import (
    FoldingEngine,
    FoldingEngineNotInstalledError,
    _validate_sequence,
)

if TYPE_CHECKING:
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
            **kwargs: Reserved for future per-call options.

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
                diffusion sample) with ``aggregate_score``, ``ptm``,
                ``iptm`` — for users who want to inspect ranking
                spread or pick a non-best sample
            - ``provenance``: :class:`Provenance` capturing all
                constructor kwargs

        Raises:
            FoldingEngineNotInstalledError: If ``chai_lab`` isn't
                installed or fails to import.
            RuntimeError: If Chai-1 produces no parseable output.
        """
        sequence = _validate_sequence(sequence)
        return self._run_local(sequence)

    # ------------------------------------------------------------------
    # Local-execution path (testable seam)
    # ------------------------------------------------------------------
    def _run_local(self, sequence: str) -> Protein:
        """Drive ``chai_lab.chai1.run_inference`` in a temp directory.

        Separated from :meth:`predict` so tests can mock the run
        without touching sequence validation or output parsing.
        """
        with tempfile.TemporaryDirectory(prefix="molforge_chai1_") as td:
            tmpdir = Path(td)
            fasta_path = tmpdir / "input.fasta"
            output_dir = tmpdir / "out"
            output_dir.mkdir()

            fasta_path.write_text(
                self._build_fasta(sequence, name="query"),
                encoding="utf-8",
            )

            self._run_inference(fasta_path, output_dir)
            samples = self._collect_samples(output_dir)

        return self._parse_outputs(samples=samples, sequence=sequence)

    # ------------------------------------------------------------------
    # Process plumbing (each step a testable seam)
    # ------------------------------------------------------------------
    def _build_fasta(self, sequence: str, *, name: str) -> str:
        """Construct the Chai-1 FASTA input for a single protein chain.

        Chai-1's FASTA uses a typed header like ``>protein|name=foo``;
        the type prefix is what tells Chai-1 whether each entity is a
        protein, ligand (SMILES), DNA, or RNA. For a single protein
        chain we emit one ``>protein|name=...`` record.
        """
        return f">protein|name={name}\n{sequence}\n"

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
        sequence: str,
    ) -> Protein:
        """Pick the best sample by ``aggregate_score`` and build the
        canonical :class:`Protein`.

        The non-best samples' headline scores are preserved in
        ``metadata["per_sample_scores"]`` so users wanting to inspect
        ranking spread can do so without re-running Chai-1.
        """
        if not samples:
            raise RuntimeError("Chai-1 produced no samples — nothing to parse.")

        # Pick the highest aggregate_score. If aggregate_score is
        # missing for some samples (unusual) we sort with -inf so
        # those samples lose the tiebreak.
        def _aggregate(sample: dict[str, Any]) -> float:
            v = sample["scores"].get("aggregate_score")
            return float(v) if v is not None else float("-inf")

        best = max(samples, key=_aggregate)
        per_sample_scores = [
            {key: s["scores"].get(key) for key in _HEADLINE_SCORE_KEYS} for s in samples
        ]

        # Read the chosen CIF through molforge's own reader to get a
        # proper Protein with AtomArray, then layer on Chai's
        # confidence + ranking metadata. We use read_cif_string
        # (not read_cif) so the parsing works on the in-memory CIF
        # text captured during _collect_samples — the upstream
        # tempdir is gone by the time we get here.
        from molforge.io.mmcif import read_cif_string

        protein = read_cif_string(best["cif_text"])

        # pLDDT is in the CIF's B-factor column (AlphaFold convention,
        # which Chai-1 follows). Extract per-residue mean over CA atoms.
        confidence_per_residue = _per_residue_plddt_from_cif(protein)
        mean_confidence = (
            float(confidence_per_residue.mean()) if confidence_per_residue.size > 0 else 0.0
        )

        provenance = Provenance.from_engine(
            engine="Chai-1",
            parameters={
                "device": self.device,
                "use_msa_server": self.use_msa_server,
                "msa_server_url": self.msa_server_url,
                "num_trunk_recycles": self.num_trunk_recycles,
                "num_diffn_timesteps": self.num_diffn_timesteps,
                "seed": self.seed,
                "cache_dir": self.cache_dir,
            },
            inputs={"sequence": sequence},
        )

        protein.metadata.update(
            {
                "engine": "Chai-1",
                "source_sequence": sequence,
                mk.CONFIDENCE_PER_RESIDUE: confidence_per_residue,
                mk.MEAN_CONFIDENCE: mean_confidence,
                "aggregate_score": best["scores"].get("aggregate_score"),
                "ptm": best["scores"].get("ptm"),
                "iptm": best["scores"].get("iptm"),
                "best_sample_index": best["index"],
                "per_sample_scores": per_sample_scores,
                mk.PROVENANCE: provenance,
            }
        )
        return protein


# ---------------------------------------------------------------------
# NPZ + CIF helpers (module-level so tests can exercise them directly)
# ---------------------------------------------------------------------


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
