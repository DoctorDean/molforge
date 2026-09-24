"""Boltz / Boltz-2 wrapper.

[Boltz](https://github.com/jwohlwend/boltz) (Wohlwend et al. 2024) is
an MIT-licensed open-source biomolecular structure prediction model
from the MIT Jameel Clinic. It handles proteins, nucleic acids
(DNA/RNA), and small molecules (via SMILES or CCD codes), plus
modified residues, covalent ligands, and glycans. Boltz-2 (2025) adds
state-of-the-art binding-affinity prediction and brings overall
accuracy close to AlphaFold-3.

Unlike ESMFold (single-sequence transformer) or AlphaFold-via-
ColabFold (MSA-based with a clean Python API), Boltz is invoked
through the ``boltz`` command-line entry point. This wrapper drives
the CLI via :mod:`subprocess` against a temporary directory and parses
the resulting mmCIF + JSON confidence output.

Installation::

    pip install boltz

GPU is strongly recommended. Boltz auto-downloads weights to
``~/.boltz`` on first use; first-call latency is dominated by that
download.

Confidence convention:

Boltz writes a JSON sidecar with several confidence metrics. We
populate :class:`Protein.metadata` with the uniform
``confidence_per_residue`` / ``mean_confidence`` keys (sourced from
per-residue pLDDT) and also surface Boltz-specific metrics
(``ptm``, ``iptm``, ``confidence_score``) for callers that want them.

Multi-chain note: this wrapper focuses on the single-chain prediction
path in v1, matching how AlphaFold and ESMFold are exposed today.
Boltz natively supports complex prediction; that's a one-extra-line
extension via a multi-entity YAML input and will land alongside
multi-chain support in the other folding wrappers.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from molforge.cache import get_default_cache
from molforge.core import metadata_keys as mk
from molforge.core.provenance import Provenance
from molforge.folding import ComplexSpec, Entity, TemplatePolicy, _coerce_template_policy
from molforge.wrappers._versions import engine_version
from molforge.wrappers.folding._base import (
    FoldingEngine,
    FoldingEngineNotInstalledError,
    _optional_sampling_parameters,
    _reject_unknown_kwargs,
    _reject_unsupported_template_mode,
    _validate_msa_depth,
    _validate_sequence,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from molforge.core import Protein

#: The only keywords the public predict methods accept per call. Anything
#: else is a caller mistake — and a silently dropped ``seed=`` is the worst
#: kind, since the run looks reproducible and isn't — so unknown keywords
#: raise instead of being swallowed.
_PER_CALL_OPTIONS = frozenset({"seed"})


class Boltz(FoldingEngine):
    """Wrapper around the Boltz biomolecular prediction model.

    Args:
        model_version: ``"boltz1"`` or ``"boltz2"``. The CLI defaults
            to ``boltz2`` when available. Set explicitly for
            reproducibility.
        use_msa_server: If ``True`` (default), Boltz hits the MMseqs2
            MSA server for protein chains. Set ``False`` for fast
            single-sequence inference (lower accuracy, no internet
            required after weight download).
        msa_depth: Cap the number of MSA sequences the model may use, as
            ``--subsample_msa --num_subsampled_msa <n>``. ``None``
            (default) leaves Boltz's own behaviour alone — the full
            alignment, up to its ``max_msa_seqs`` ceiling. A shallow
            alignment makes the model lean less on coevolution, which
            is how you sweep a depth ladder (8 / 16 / 32 / 64 / full)
            and watch a prediction move. Recorded in provenance, so
            each rung of the ladder gets its own cache slot instead of
            colliding with the others. To drop the MSA entirely, set
            ``use_msa_server=False`` rather than ``msa_depth``.
        templates: A :class:`~molforge.folding.TemplatePolicy` (or the
            string ``"none"``). ``None`` (default) leaves Boltz's own
            behaviour alone, which is to run without templates. Boltz
            supports ``TemplatePolicy.none()`` and
            ``TemplatePolicy.from_structures(...)``, which it receives as
            a ``templates:`` block in the input YAML; the ``"hits"`` and
            ``"server"`` modes are Chai-1-only and raise here. Recorded
            in provenance, so a templated and an untemplated run of the
            same target never share a cached result.
        recycling_steps: How many trunk-recycling rounds Boltz runs.
            Default ``None`` lets Boltz choose its own (3 for boltz1,
            10 for boltz2 currently).
        diffusion_samples: Number of diffusion samples drawn per
            prediction. Default ``None`` uses Boltz's default (1).
            Higher = more thorough sampling, slower.
        sampling_steps: Number of diffusion sampling steps. Default
            ``None`` uses Boltz's own (200 for boltz1, 30 for boltz2).
        seed: Seed for Boltz's random number generator, passed through as
            ``--seed`` so diffusion sampling is reproducible — the same
            spec, engine settings, and seed give the same structure.
            ``None`` (default) leaves Boltz unseeded, matching the CLI.
            The seed is recorded in the returned structure's provenance
            (and so is part of the cache key). Note: as with any
            GPU-accelerated sampler, exact bit-for-bit reproducibility
            also depends on device and non-determinism settings; small
            numerical drift across machines is expected.
            Can be overridden per call, e.g. ``predict(seq, seed=7)``,
            which is the cheap way to draw a seeded ensemble.
        device: Which device to use. Default ``None`` lets Boltz
            auto-detect (CUDA → CPU fallback). Pass ``"cpu"`` to
            force CPU even when a GPU is present.
        executable: Path to the ``boltz`` CLI binary. ``None``
            (default) means look it up on ``$PATH``. Override only
            for testing or non-standard installs.
        cache_dir: Where Boltz looks for / downloads its weights.
            ``None`` (default) uses Boltz's own default
            (``~/.boltz``).

    Example:
        >>> from molforge.wrappers.folding import Boltz
        >>> engine = Boltz(model_version="boltz2", use_msa_server=True)
        >>> protein = engine.predict("MKTVRQERLKSIVRILERSK")
        >>> protein.metadata["mean_confidence"]
        87.3
        >>> protein.metadata["ptm"]
        0.84
    """

    name = "Boltz"

    def __init__(
        self,
        *,
        model_version: Literal["boltz1", "boltz2"] = "boltz2",
        use_msa_server: bool = True,
        msa_depth: int | None = None,
        templates: TemplatePolicy | str | None = None,
        recycling_steps: int | None = None,
        diffusion_samples: int | None = None,
        sampling_steps: int | None = None,
        seed: int | None = None,
        device: str | None = None,
        executable: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        if model_version not in ("boltz1", "boltz2"):
            raise ValueError(f"model_version must be 'boltz1' or 'boltz2', got {model_version!r}")
        self.model_version = model_version
        self.use_msa_server = use_msa_server
        self.msa_depth = _validate_msa_depth(msa_depth)
        self.templates = _validate_boltz_templates(templates)
        self.recycling_steps = recycling_steps
        self.diffusion_samples = diffusion_samples
        self.sampling_steps = sampling_steps
        self.seed = _validate_seed(seed)
        self.device = device
        self.executable = executable
        self.cache_dir = cache_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, sequence: str, **kwargs: object) -> Protein:
        """Fold a single sequence into a :class:`Protein` via the boltz CLI.

        Args:
            sequence: One-letter amino-acid sequence.
            **kwargs: Per-call options. Only ``seed`` is accepted — an
                ``int`` (or ``None``) overriding the constructor's ``seed``
                for this call. Any other keyword raises :class:`TypeError`
                rather than being silently ignored.

        Returns:
            A :class:`Protein` with:

            - ``metadata["engine"] = "Boltz"``
            - ``metadata["model_version"]``: ``"boltz1"`` or ``"boltz2"``
            - ``metadata["source_sequence"]``: the input sequence
            - ``metadata["confidence_per_residue"]``: ``(L,)`` float32 pLDDT
            - ``metadata["mean_confidence"]``: scalar float pLDDT
            - ``metadata["ptm"]``: predicted TM-score
            - ``metadata["iptm"]``: interface pTM (only meaningful
                for complexes; usually 0 for single-chain inputs)
            - ``metadata["confidence_score"]``: Boltz's composite
                confidence (``0.8 * pLDDT + 0.2 * iPTM`` for boltz2)

        Raises:
            FoldingEngineNotInstalledError: If the ``boltz`` CLI isn't
                on ``$PATH`` (or at the configured ``executable``).
            RuntimeError: If the CLI runs but produces no output, or
                its output can't be parsed.
            TypeError: If an unsupported keyword argument is passed.
        """
        seed = self._resolve_call_seed(kwargs, method="predict")
        sequence = _validate_sequence(sequence)
        # Delegate to the spec-based code path with a single-entity
        # spec. This keeps the multi-component machinery as the one
        # canonical implementation — single-sequence is just a
        # degenerate special case.
        spec = ComplexSpec.from_protein(sequence)
        return self._predict_spec(spec, single_sequence=sequence, seed=seed)

    def predict_complex(self, spec: ComplexSpec, **kwargs: object) -> Protein:
        """Fold a multi-component complex via the boltz CLI.

        This is the headline AlphaFold-3-style capability: predict
        the structure of multiple protein chains, DNA/RNA, and/or
        small-molecule ligands in a single forward pass. See
        :class:`molforge.folding.ComplexSpec` for input shape and
        examples.

        Args:
            spec: A :class:`ComplexSpec` describing the entities to
                fold. The returned :class:`Protein` has one chain in
                its ``atom_array`` per polymer entity (or per copy,
                for homo-oligomers). Ligand atoms appear as
                hetero-atoms with chain IDs assigned by the spec.
            **kwargs: Per-call options. Only ``seed`` is accepted — an
                ``int`` (or ``None``) overriding the constructor's ``seed``
                for this call. Any other keyword raises :class:`TypeError`
                rather than being silently ignored.

        Returns:
            A :class:`Protein` with multi-chain ``atom_array`` and
            metadata as documented in :meth:`predict`, plus:

            - ``metadata["complex_spec"]``: the :class:`ComplexSpec`
                passed in, for traceability.
            - ``metadata["per_chain_ptm"]``: per-chain pTM values
                from Boltz's confidence JSON.

        Raises:
            FoldingEngineNotInstalledError: If the ``boltz`` CLI
                isn't installed.
            RuntimeError: If the CLI runs but produces no parseable
                output.
            TypeError: If an unsupported keyword argument is passed.

        Examples:
            Protein-ligand complex::

                from molforge.folding import ComplexSpec
                from molforge.wrappers.folding import Boltz

                spec = ComplexSpec.protein_ligand(
                    protein_sequence="MVTPEG...",
                    ligand_smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
                )
                complex_struct = Boltz().predict_complex(spec)
                # complex_struct.atom_array has chain A (protein) and
                # chain B (ligand atoms).
        """
        seed = self._resolve_call_seed(kwargs, method="predict_complex")
        return self._predict_spec(spec, single_sequence=None, seed=seed)

    def predict_affinity(self, spec: ComplexSpec, **kwargs: object) -> Protein:
        """Predict a protein-ligand complex *and its binding affinity* (Boltz-2).

        Boltz-2's headline capability: alongside the folded complex, it
        predicts how tightly the ligand binds. This method folds ``spec``
        with an ``affinity`` property on the ligand and surfaces the result
        in the returned structure's metadata.

        Args:
            spec: A :class:`ComplexSpec` with exactly one ligand entity —
                the binder whose affinity is predicted — plus at least one
                protein chain. Build it with
                :meth:`ComplexSpec.protein_ligand`.
            **kwargs: Per-call options. Only ``seed`` is accepted — an
                ``int`` (or ``None``) overriding the constructor's ``seed``
                for this call. Any other keyword raises :class:`TypeError`
                rather than being silently ignored.

        Returns:
            The folded complex as a :class:`Protein` whose ``metadata`` adds,
            on top of the usual confidence keys:

            - ``affinity_value``: Boltz-2's ``affinity_pred_value`` (log-scale IC50-like; lower = stronger binding).
            - ``affinity_probability``: probability the ligand is a binder (0-1).
            - ``affinity``: the full affinity JSON, verbatim.

        Raises:
            ValueError: If this engine wasn't constructed with
                ``model_version="boltz2"`` (affinity is a Boltz-2 feature),
                or ``spec`` doesn't have exactly one ligand entity.
            TypeError: If an unsupported keyword argument is passed.
        """
        seed = self._resolve_call_seed(kwargs, method="predict_affinity")
        if self.model_version != "boltz2":
            raise ValueError(
                "affinity prediction requires Boltz-2; construct the engine with "
                f'Boltz(model_version="boltz2"). Got {self.model_version!r}.'
            )
        binder = _single_ligand_chain_id(spec)
        return self._predict_spec(spec, single_sequence=None, affinity_binder=binder, seed=seed)

    # ------------------------------------------------------------------
    # Local-execution path (testable seam)
    # ------------------------------------------------------------------
    def _resolve_call_seed(self, kwargs: Mapping[str, object], *, method: str) -> int | None:
        """Validate a call's keywords and return the seed it should run with.

        A per-call ``seed=`` wins over the constructor's; absent (or
        ``None``) means "use the engine's". Unrecognized keywords raise
        rather than being dropped — a swallowed ``seed=`` is how a run that
        looks reproducible quietly isn't.
        """
        _reject_unknown_kwargs(
            kwargs,
            engine="Boltz",
            method=method,
            supported=_PER_CALL_OPTIONS,
            constructor_example="Boltz(diffusion_samples=3)",
        )
        return self._seed_for(_validate_seed(kwargs.get("seed")))

    def _seed_for(self, seed: int | None) -> int | None:
        """The effective seed: an explicit per-call value, else the engine's."""
        return self.seed if seed is None else seed

    def _predict_spec(
        self,
        spec: ComplexSpec,
        *,
        single_sequence: str | None,
        affinity_binder: str | None = None,
        seed: int | None = None,
    ) -> Protein:
        """The shared spec-based execution path.

        Both :meth:`predict` and :meth:`predict_complex` route through
        here. ``single_sequence`` is non-None when the caller was the
        single-sequence ``predict()`` path; that lets us preserve the
        existing ``metadata["source_sequence"]`` contract for users
        that depend on it. For multi-entity calls it's None and the
        spec itself goes into ``metadata["complex_spec"]``.
        """
        # Cache lookup. Build the Provenance upfront so we can
        # check the cache *before* spawning the boltz subprocess.
        provenance = self._build_provenance(
            spec, single_sequence=single_sequence, affinity_binder=affinity_binder, seed=seed
        )
        cache = get_default_cache()
        cached: Protein | None = cache.get(provenance, "protein")
        if cached is not None:
            return cached

        binary = self._require_boltz()

        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            input_yaml = tmpdir / "input.yaml"
            output_dir = tmpdir / "out"
            output_dir.mkdir()

            input_yaml.write_text(
                self._build_input_yaml_from_spec(spec, affinity_binder=affinity_binder),
                encoding="utf-8",
            )

            cmd = self._build_command(binary, input_yaml, output_dir, seed=seed)
            env = self._build_env()
            self._invoke(cmd, env=env)

            cif_text, confidence_json = self._collect_outputs(output_dir)
            affinity_json = (
                self._collect_affinity(output_dir) if affinity_binder is not None else {}
            )

        result = self._parse_outputs(
            cif_text=cif_text,
            confidence_json=confidence_json,
            sequence=single_sequence,
            spec=spec,
            provenance=provenance,
            affinity_json=affinity_json,
        )
        cache.put(provenance, result, "protein")
        return result

    def _build_provenance(
        self,
        spec: ComplexSpec,
        *,
        single_sequence: str | None,
        affinity_binder: str | None = None,
        seed: int | None = None,
    ) -> Provenance:
        """Construct the Provenance for a predict / predict_complex /
        predict_affinity call.

        Factored out of :meth:`_parse_outputs` so :meth:`_predict_spec`
        can build it upfront for cache lookup. Pure function of inputs
        + constructor parameters — does not touch the boltz CLI. The
        ``affinity_binder`` is recorded in the parameters so an affinity
        run doesn't collide with a plain fold in the cache; ``seed`` is
        recorded for the same reason (two seeds are two different runs)
        and so a replayed manifest re-runs with the seed it was folded
        with. ``seed=None`` falls back to the engine's own.
        """
        prov_inputs: dict[str, object]
        if single_sequence is not None:
            prov_inputs = {"sequence": single_sequence}
        else:
            prov_inputs = {"complex_spec": _serialize_spec_for_provenance(spec)}
        return Provenance.from_engine(
            engine="Boltz",
            operation="predict",
            engine_version=engine_version("boltz"),
            parameters={
                "model_version": self.model_version,
                "use_msa_server": self.use_msa_server,
                **_optional_sampling_parameters(self.msa_depth, self.templates),
                "recycling_steps": self.recycling_steps,
                "diffusion_samples": self.diffusion_samples,
                "sampling_steps": self.sampling_steps,
                "seed": self._seed_for(seed),
                "device": self.device,
                "affinity_binder": affinity_binder,
            },
            inputs=prov_inputs,
        )

    def _run_local(self, sequence: str) -> Protein:
        """Single-sequence local execution (legacy API).

        Preserved for any external code that may have been calling
        the private seam directly. New code should use
        :meth:`_predict_spec` instead.
        """
        sequence = _validate_sequence(sequence)
        return self._predict_spec(
            ComplexSpec.from_protein(sequence), single_sequence=sequence, seed=self.seed
        )

    # ------------------------------------------------------------------
    # Process plumbing (each step a testable seam)
    # ------------------------------------------------------------------
    def _require_boltz(self) -> str:
        """Resolve the boltz binary or raise a clean error with install hints."""
        binary = self.executable or shutil.which("boltz")
        if not binary:
            raise FoldingEngineNotInstalledError(
                "Boltz requires the `boltz` CLI to be on $PATH. Install with:\n"
                "    pip install boltz\n"
                "Or pass an explicit `executable=` path to the Boltz constructor. "
                "See https://github.com/jwohlwend/boltz for setup notes."
            )
        return binary

    def _build_input_yaml(self, sequence: str, *, name: str) -> str:
        """Single-sequence YAML builder (legacy API).

        Preserved as a thin wrapper around :meth:`_build_input_yaml_from_spec`
        so existing tests of the single-sequence path keep working.
        """
        return self._build_input_yaml_from_spec(ComplexSpec.from_protein(sequence))

    def _build_input_yaml_from_spec(
        self, spec: ComplexSpec, *, affinity_binder: str | None = None
    ) -> str:
        """Construct the Boltz YAML input for a ComplexSpec.

        Boltz's input format is a YAML document with a top-level
        ``sequences`` list of entities. Each entity is keyed by its
        type (``protein`` / ``ligand`` / ``dna`` / ``rna``). For
        single-protein inputs this collapses to the trivial
        single-entity YAML the v1 wrapper originally produced.

        Multi-copy entities (``Entity.copies > 1``) are emitted with
        ``id: [A, B, ...]`` — Boltz's documented shape for declaring
        multiple chains share the same input sequence.

        Hand-built YAML rather than pulling PyYAML for one schema
        we entirely control.
        """
        lines: list[str] = ["version: 1", "sequences:"]
        chain_ids_per_entity = spec.assigned_chain_ids()

        for entity, chain_ids in zip(spec.entities, chain_ids_per_entity, strict=True):
            lines.extend(_boltz_yaml_entity(entity, chain_ids))

        # Templates: a top-level `templates` list of structure files.
        # Boltz matches each against the query chains itself unless a
        # chain_id is given, which is a refinement molforge doesn't
        # model yet. mode="none" emits nothing, which is already Boltz's
        # behaviour — the point is that it is now stated rather than
        # assumed, and recorded in provenance either way.
        if self.templates is not None and self.templates.mode == "structures":
            lines.append("templates:")
            for path in self.templates.structures:
                lines.append(f"  - {_boltz_template_key(path)}: {path}")

        # Affinity request: a top-level `properties` block naming the
        # binder chain. Boltz-2 computes affinity when this is present.
        if affinity_binder is not None:
            lines += ["properties:", "  - affinity:", f"      binder: {affinity_binder}"]

        # Trailing newline keeps Boltz's parser happy on some
        # versions and makes diffing the file pleasant.
        return "\n".join(lines) + "\n"

    def _collect_affinity(self, output_dir: Path) -> dict[str, Any]:
        """Locate and parse Boltz-2's ``affinity_*.json`` sidecar.

        Missing / malformed affinity output isn't fatal — the folded
        structure is still returned; the affinity metadata is simply
        absent. Boltz writes the file alongside the model outputs.
        """
        candidates = sorted(output_dir.rglob("affinity_*.json"))
        if not candidates:
            return {}
        try:
            data = json.loads(candidates[0].read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def _build_command(
        self,
        binary: str,
        input_path: Path,
        output_dir: Path,
        *,
        seed: int | None = None,
    ) -> list[str]:
        """Assemble the ``boltz predict ...`` command line.

        Args:
            binary: The resolved ``boltz`` executable.
            input_path: The YAML spec written for this call.
            output_dir: Where Boltz should write its output.
            seed: Seed for this call; ``None`` falls back to the engine's
                ``seed``, and an unseeded engine omits ``--seed`` so Boltz
                keeps its own (unseeded) default.
        """
        cmd: list[str] = [
            binary,
            "predict",
            str(input_path),
            "--out_dir",
            str(output_dir),
            "--model",
            self.model_version,
            "--output_format",
            "mmcif",
            # Boltz uses overwrite-the-output semantics by default;
            # `--override` is safe in our tempdir.
            "--override",
        ]
        if self.use_msa_server:
            cmd.append("--use_msa_server")
        if self.msa_depth is not None:
            # --subsample_msa is the switch; --num_subsampled_msa is the
            # count. Boltz ignores the count without the flag, so both
            # go together or neither does.
            cmd.extend(["--subsample_msa", "--num_subsampled_msa", str(self.msa_depth)])
        if self.recycling_steps is not None:
            cmd.extend(["--recycling_steps", str(self.recycling_steps)])
        if self.diffusion_samples is not None:
            cmd.extend(["--diffusion_samples", str(self.diffusion_samples)])
        if self.sampling_steps is not None:
            cmd.extend(["--sampling_steps", str(self.sampling_steps)])
        effective_seed = self._seed_for(seed)
        if effective_seed is not None:
            cmd.extend(["--seed", str(effective_seed)])
        if self.device == "cpu":
            cmd.extend(["--accelerator", "cpu"])
        elif self.device is not None and self.device.startswith("cuda"):
            cmd.extend(["--accelerator", "gpu"])
        return cmd

    def _build_env(self) -> dict[str, str]:
        """Prepare environment variables for the subprocess.

        Propagates the parent environment, then overrides BOLTZ_CACHE
        if the user set a custom cache_dir.
        """
        env = dict(os.environ)
        if self.cache_dir is not None:
            env["BOLTZ_CACHE"] = self.cache_dir
        return env

    def _invoke(self, cmd: list[str], *, env: dict[str, str]) -> None:
        """Run the subprocess and raise on non-zero exit.

        Captured stdout/stderr is surfaced in the RuntimeError message
        so users see what Boltz reported when something fails.
        """
        try:
            subprocess.run(
                cmd,
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"`boltz predict` failed (exit code {e.returncode}).\n"
                f"stderr:\n{e.stderr}\n"
                f"stdout:\n{e.stdout}"
            ) from e

    def _collect_outputs(self, output_dir: Path) -> tuple[str, dict[str, Any]]:
        """Locate the Boltz output CIF + confidence JSON.

        Boltz writes into a nested directory structure under
        ``output_dir`` named after the input file. We don't hard-code
        the layout — we just glob for the relevant files. If multiple
        models were sampled, we take ``model_0`` (the highest-ranked).
        """
        cif_candidates = sorted(output_dir.rglob("*_model_0.cif")) or sorted(
            output_dir.rglob("*.cif")
        )
        if not cif_candidates:
            raise RuntimeError(
                f"Boltz produced no .cif output in {output_dir}. "
                "Check the Boltz logs for the actual error."
            )
        cif_text = cif_candidates[0].read_text(encoding="utf-8")

        # Confidence JSON lives alongside the CIF, named confidence_*.json.
        cif_path = cif_candidates[0]
        json_candidates = sorted(cif_path.parent.glob("confidence_*model_0.json")) or sorted(
            cif_path.parent.glob("confidence_*.json")
        )
        confidence_json: dict[str, Any] = {}
        if json_candidates:
            try:
                confidence_json = json.loads(json_candidates[0].read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                # Confidence is nice-to-have, not blocking. The
                # structure is still useful.
                confidence_json = {}
        return cif_text, confidence_json

    # ------------------------------------------------------------------
    # Output parsing (testable in isolation)
    # ------------------------------------------------------------------
    def _parse_outputs(
        self,
        *,
        cif_text: str,
        confidence_json: dict[str, Any],
        sequence: str | None,
        spec: ComplexSpec | None = None,
        provenance: Provenance | None = None,
        affinity_json: dict[str, Any] | None = None,
    ) -> Protein:
        """Parse a Boltz mmCIF + confidence JSON into a Protein with metadata.

        Boltz writes per-atom pLDDT to the B-factor column of the CIF,
        same as AlphaFold and ESMFold. We follow molforge's uniform
        confidence convention: per-atom in ``confidence_per_atom``,
        per-residue in ``confidence_per_residue``, scalar mean in
        ``mean_confidence``. Boltz-specific scalars (pTM, iPTM,
        composite confidence score) are surfaced verbatim.

        Args:
            cif_text: The raw mmCIF text from Boltz's prediction.
            confidence_json: The parsed confidence sidecar JSON.
            sequence: The original single-sequence input, when the
                caller is the single-sequence :meth:`predict` path.
                ``None`` for multi-component :meth:`predict_complex`
                calls (the spec carries that information instead).
            spec: The :class:`ComplexSpec` used for this prediction.
                Always set; for single-sequence predict() it's the
                trivial single-entity spec.
        """
        from molforge.io.mmcif import read_cif_string

        protein = read_cif_string(cif_text)
        arr = protein.atom_array
        plddt_per_atom = np.asarray(arr.b_factor, dtype=np.float32).copy()

        per_residue: list[float] = []
        for sl in arr.iter_residue_slices():
            per_residue.append(float(plddt_per_atom[sl].mean()))
        per_residue_arr = np.asarray(per_residue, dtype=np.float32)
        mean_conf = float(per_residue_arr.mean()) if per_residue_arr.size else 0.0

        # Prefer JSON-provided scalars where present (more precise than
        # column-derived means).
        ptm = float(confidence_json.get("ptm", 0.0))
        iptm = float(confidence_json.get("iptm", 0.0))
        composite = float(
            confidence_json.get(
                "confidence_score",
                # Boltz-2 default composite if JSON didn't carry one.
                0.8 * (mean_conf / 100.0) + 0.2 * iptm,
            )
        )

        # Per-chain confidence stats. Boltz writes a chains_ptm dict
        # (chain_id -> pTM) in its confidence JSON for multi-chain
        # predictions; pass through verbatim when present.
        per_chain_ptm = confidence_json.get("chains_ptm")
        pair_chains_iptm = confidence_json.get("pair_chains_iptm")

        # Provenance: use the prebuilt one when supplied (the normal
        # _predict_spec path), otherwise build a fresh one (legacy
        # test path that calls _parse_outputs directly).
        if provenance is None:
            if spec is None and sequence is not None:
                spec = ComplexSpec.from_protein(sequence)
            assert spec is not None, "_parse_outputs needs spec or sequence"
            provenance = self._build_provenance(spec, single_sequence=sequence)

        metadata_update: dict[str, object] = {
            mk.PROVENANCE: provenance,
            mk.ENGINE: "Boltz",
            mk.MODEL_VERSION: self.model_version,
            mk.USE_MSA_SERVER: self.use_msa_server,
            mk.CONFIDENCE_PER_ATOM: plddt_per_atom,
            mk.CONFIDENCE_PER_RESIDUE: per_residue_arr,
            mk.MEAN_CONFIDENCE: mean_conf,
            mk.PTM: ptm,
            mk.IPTM: iptm,
            mk.CONFIDENCE_SCORE: composite,
        }
        # Preserve the SOURCE_SEQUENCE key only for single-sequence
        # calls (where it has a clear meaning). For complexes, surface
        # the spec instead under a dedicated key.
        if sequence is not None:
            metadata_update[mk.SOURCE_SEQUENCE] = sequence
        if spec is not None and sequence is None:
            metadata_update["complex_spec"] = spec
        if per_chain_ptm is not None:
            metadata_update["per_chain_ptm"] = per_chain_ptm
        if pair_chains_iptm is not None:
            metadata_update["pair_chains_iptm"] = pair_chains_iptm

        # Boltz-2 affinity (only present on predict_affinity calls).
        if affinity_json:
            value = _affinity_value(affinity_json)
            probability = _affinity_probability(affinity_json)
            if value is not None:
                metadata_update[mk.AFFINITY_VALUE] = value
            if probability is not None:
                metadata_update[mk.AFFINITY_PROBABILITY] = probability
            metadata_update["affinity"] = affinity_json

        protein.metadata.update(metadata_update)
        return protein


# ---------------------------------------------------------------------
# Module-level helpers (testable without a Boltz instance)
# ---------------------------------------------------------------------


def _single_ligand_chain_id(spec: ComplexSpec) -> str:
    """Return the chain id of ``spec``'s sole ligand — the affinity binder.

    Raises:
        ValueError: If the spec doesn't have exactly one ligand entity.
    """
    chain_ids_per_entity = spec.assigned_chain_ids()
    ligand_chains = [
        chain_ids[0]
        for entity, chain_ids in zip(spec.entities, chain_ids_per_entity, strict=True)
        if entity.is_ligand and chain_ids
    ]
    if len(ligand_chains) != 1:
        raise ValueError(
            "affinity prediction needs exactly one ligand entity (the binder); "
            f"the spec has {len(ligand_chains)}. Build it with "
            "ComplexSpec.protein_ligand(...)."
        )
    return ligand_chains[0]


def _affinity_value(affinity_json: dict[str, Any]) -> float | None:
    """Boltz-2's headline ``affinity_pred_value`` (log-scale, lower = stronger)."""
    return _maybe_float(affinity_json.get("affinity_pred_value"))


def _affinity_probability(affinity_json: dict[str, Any]) -> float | None:
    """Boltz-2's ``affinity_probability_binary`` (probability of being a binder)."""
    return _maybe_float(affinity_json.get("affinity_probability_binary"))


def _validate_boltz_templates(templates: TemplatePolicy | str | None) -> TemplatePolicy | None:
    """Coerce and check the ``templates`` argument for Boltz.

    Boltz reads templates as structure files named in its input YAML, so
    it can honour ``"none"`` and ``"structures"``. It has no template
    search of its own and no notion of a hits file, so the two Chai-1
    modes are refused here rather than silently ignored.
    """
    return _reject_unsupported_template_mode(
        _coerce_template_policy(templates),
        engine="Boltz",
        supported={"none", "structures"},
        hint=(
            "Boltz takes template structures directly — "
            "TemplatePolicy.from_structures('4hhb.cif'). The 'hits' and "
            "'server' modes are Chai-1's."
        ),
    )


def _boltz_template_key(path: str) -> str:
    """Whether Boltz should read this template as ``cif:`` or ``pdb:``.

    Boltz keys the YAML entry by format, and assigns template chain IDs
    differently for the two, so the suffix has to pick the key rather
    than defaulting to one. Anything that isn't recognisably a PDB file
    is offered as mmCIF, which is the format Boltz documents first and
    the one molforge writes.
    """
    return "pdb" if path.lower().endswith((".pdb", ".ent")) else "cif"


def _validate_seed(seed: object) -> int | None:
    """Check a user-supplied seed is an ``int`` (or ``None``).

    Typed on the constructor, but checked at runtime too: a seed arriving as
    a string from a config file, or as a stray ``bool``, would otherwise sail
    into the command line and be rejected by the CLI several minutes later.
    """
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int or None, got {type(seed).__name__}: {seed!r}")
    return seed


def _maybe_float(value: object) -> float | None:
    """Coerce ``value`` to float, or ``None`` when missing / non-numeric."""
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _boltz_yaml_entity(entity: Entity, chain_ids: list[str]) -> list[str]:
    """Render one ComplexSpec.Entity as Boltz YAML lines.

    Returns a list of lines (without trailing newlines) ready to be
    joined into the larger document. The Boltz YAML schema is::

        - protein:
            id: A
            sequence: MKQH...
        - ligand:
            id: B
            smiles: 'CCO'
        - ligand:
            id: C
            ccd: ATP
        - dna:
            id: D
            sequence: ATCG
        - rna:
            id: E
            sequence: AUCG

    For multi-copy entities, ``id`` becomes a YAML list:
    ``id: [A, B]``. Boltz documents this as the canonical shape for
    declaring identical chains share a single input sequence.
    """
    # Boltz's entity type names exactly match Entity.kind, so no
    # mapping needed.
    kind_key = entity.kind

    # id: A   (single)   or   id: [A, B]   (multi-copy)
    if len(chain_ids) == 1:
        id_line = f"      id: {chain_ids[0]}"
    else:
        id_line = "      id: [{}]".format(", ".join(chain_ids))

    body: list[str] = [f"  - {kind_key}:", id_line]

    if entity.is_polymer:
        body.append(f"      sequence: {entity.normalized_sequence()}")
    elif entity.is_ligand:
        if entity.smiles is not None:
            # YAML strings with special chars (e.g. SMILES with
            # backslashes or quotes) need single-quoting. Smiles
            # are well-defined ASCII so single-quote is safe.
            body.append(f"      smiles: '{entity.smiles}'")
        else:
            assert entity.ccd is not None  # validated upstream
            body.append(f"      ccd: {entity.ccd}")

    return body


def _serialize_spec_for_provenance(spec: ComplexSpec | None) -> object:
    """Render a ComplexSpec to a JSON-safe shape for Provenance.inputs.

    Provenance.inputs must be JSON-serializable; the dataclass itself
    is not (it's a frozen dataclass with Entity tuples). We flatten
    it to a list of dicts where each entry has the key fields. The
    Provenance still cross-references back to the actual spec via
    ``Protein.metadata["complex_spec"]`` for callers that want the
    rich type.
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
