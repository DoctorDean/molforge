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

from molforge.core import metadata_keys as mk
from molforge.core.provenance import Provenance
from molforge.wrappers.folding._base import (
    FoldingEngine,
    FoldingEngineNotInstalledError,
    _validate_sequence,
)

if TYPE_CHECKING:
    from molforge.core import Protein


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
        recycling_steps: How many trunk-recycling rounds Boltz runs.
            Default ``None`` lets Boltz choose its own (3 for boltz1,
            10 for boltz2 currently).
        diffusion_samples: Number of diffusion samples drawn per
            prediction. Default ``None`` uses Boltz's default (1).
            Higher = more thorough sampling, slower.
        sampling_steps: Number of diffusion sampling steps. Default
            ``None`` uses Boltz's own (200 for boltz1, 30 for boltz2).
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
        recycling_steps: int | None = None,
        diffusion_samples: int | None = None,
        sampling_steps: int | None = None,
        device: str | None = None,
        executable: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        if model_version not in ("boltz1", "boltz2"):
            raise ValueError(f"model_version must be 'boltz1' or 'boltz2', got {model_version!r}")
        self.model_version = model_version
        self.use_msa_server = use_msa_server
        self.recycling_steps = recycling_steps
        self.diffusion_samples = diffusion_samples
        self.sampling_steps = sampling_steps
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
            **kwargs: Reserved for future per-call options.

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
        """
        sequence = _validate_sequence(sequence)
        return self._run_local(sequence)

    # ------------------------------------------------------------------
    # Local-execution path (testable seam)
    # ------------------------------------------------------------------
    def _run_local(self, sequence: str) -> Protein:
        """Drive the boltz CLI in a temporary directory and parse the result.

        Separated from `predict` so tests can mock the subprocess call
        without touching sequence validation or output parsing.
        """
        binary = self._require_boltz()

        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            input_yaml = tmpdir / "input.yaml"
            output_dir = tmpdir / "out"
            output_dir.mkdir()

            input_yaml.write_text(
                self._build_input_yaml(sequence, name="query"),
                encoding="utf-8",
            )

            cmd = self._build_command(binary, input_yaml, output_dir)
            env = self._build_env()
            self._invoke(cmd, env=env)

            cif_text, confidence_json = self._collect_outputs(output_dir)

        return self._parse_outputs(
            cif_text=cif_text,
            confidence_json=confidence_json,
            sequence=sequence,
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
        """Construct the Boltz YAML input for a single protein chain.

        Boltz's input format is a YAML document with a top-level
        ``sequences`` list of entities. For a single protein chain we
        write one entity with ``protein`` type, a chain ID of ``A``,
        and the raw sequence string. MSA is requested implicitly via
        ``--use_msa_server`` on the CLI (no per-entity MSA field).
        """
        # Hand-built YAML to avoid a PyYAML dependency for one trivial doc.
        return f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {sequence}\n"

    def _build_command(
        self,
        binary: str,
        input_path: Path,
        output_dir: Path,
    ) -> list[str]:
        """Assemble the ``boltz predict ...`` command line."""
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
        if self.recycling_steps is not None:
            cmd.extend(["--recycling_steps", str(self.recycling_steps)])
        if self.diffusion_samples is not None:
            cmd.extend(["--diffusion_samples", str(self.diffusion_samples)])
        if self.sampling_steps is not None:
            cmd.extend(["--sampling_steps", str(self.sampling_steps)])
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
        sequence: str,
    ) -> Protein:
        """Parse a Boltz mmCIF + confidence JSON into a Protein with metadata.

        Boltz writes per-atom pLDDT to the B-factor column of the CIF,
        same as AlphaFold and ESMFold. We follow molforge's uniform
        confidence convention: per-atom in ``confidence_per_atom``,
        per-residue in ``confidence_per_residue``, scalar mean in
        ``mean_confidence``. Boltz-specific scalars (pTM, iPTM,
        composite confidence score) are surfaced verbatim.
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

        protein.metadata.update(
            {
                mk.PROVENANCE: Provenance.from_engine(
                    engine="Boltz",
                    parameters={
                        "model_version": self.model_version,
                        "use_msa_server": self.use_msa_server,
                        "recycling_steps": self.recycling_steps,
                        "diffusion_samples": self.diffusion_samples,
                        "sampling_steps": self.sampling_steps,
                        "device": self.device,
                    },
                    inputs={"sequence": sequence},
                ),
                mk.ENGINE: "Boltz",
                mk.MODEL_VERSION: self.model_version,
                mk.SOURCE_SEQUENCE: sequence,
                mk.USE_MSA_SERVER: self.use_msa_server,
                mk.CONFIDENCE_PER_ATOM: plddt_per_atom,
                mk.CONFIDENCE_PER_RESIDUE: per_residue_arr,
                mk.MEAN_CONFIDENCE: mean_conf,
                mk.PTM: ptm,
                mk.IPTM: iptm,
                mk.CONFIDENCE_SCORE: composite,
            }
        )
        return protein
