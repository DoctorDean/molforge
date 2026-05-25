"""RoseTTAFold All-Atom (RFAA) wrapper.

[RoseTTAFold All-Atom](https://github.com/baker-laboratory/RoseTTAFold-All-Atom)
(Krishna et al. 2024, *Science* 384: eadl2528) is the Baker lab's
biomolecular structure prediction model. RFAA handles a broad surface:
proteins, nucleic acids (DNA/RNA), small molecules, covalent
modifications, and metals, all in one network. Its protein-folding
accuracy is comparable to AlphaFold for ordered regions, with the
added advantage of producing useful per-prediction error estimates
(particularly ``pae_inter`` for protein-ligand interfaces).

Unlike ESMFold (one-line pip install) or Boltz (PyPI ``boltz`` package),
RFAA is **not** pip-installable. The supported deployment is a cloned
GitHub repo + Conda environment + ~400 GB of databases (BFD, UniRef30,
PDB templates) + downloaded weights + a licensed signalp6 install.
This wrapper assumes the user has done that setup work themselves and
exposes a clean Python entry point on top of the Hydra-driven CLI.

Invocation::

    python -m rf2aa.run_inference --config-name protein

This wrapper writes a small Hydra config file, drives the inference
script via :mod:`subprocess` from the configured repo directory, and
parses the resulting PDB + ``*_aux.pt`` confidence file.

v1 scope: **single-chain protein prediction only.** RFAA's headline
strengths — small-molecule co-folding, nucleic acid complexes,
covalent modifications — are out of scope here; they each require a
slightly different config shape and warrant a separate
``predict_complex()`` method (planned). This matches the v1 scope of
the AlphaFold, ESMFold, and Boltz wrappers.

Setup pointers (see the RFAA README for full details):

1. Clone ``baker-laboratory/RoseTTAFold-All-Atom``.
2. Create the ``RFAA`` Conda environment (``mamba env create -f environment.yaml``).
3. Install SE3-Transformer (``cd rf2aa/SE3Transformer && pip install -r requirements.txt && python setup.py install``).
4. Register signalp6 (separate licensed download).
5. Run ``bash install_dependencies.sh`` for input-prep tools.
6. Download ``RFAA_paper_weights.pt`` and the sequence databases.
7. Edit ``rf2aa/config/inference/base.yaml`` with the database / weights paths.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from molforge.core import metadata_keys as mk
from molforge.wrappers.folding._base import (
    FoldingEngine,
    FoldingEngineNotInstalledError,
    _validate_sequence,
)

if TYPE_CHECKING:
    from molforge.core import Protein


class RoseTTAFold(FoldingEngine):
    """Wrapper around RoseTTAFold All-Atom (RFAA) for single-chain protein folding.

    Args:
        repo_dir: Path to the cloned ``RoseTTAFold-All-Atom`` repo. If
            ``None`` (default), the wrapper looks at the ``RFAA_HOME``
            environment variable. If neither is set,
            :meth:`predict` raises
            :class:`FoldingEngineNotInstalledError`.
        python_executable: Path to the Python interpreter that has the
            RFAA environment activated. Default ``None`` uses
            ``sys.executable`` (the same Python ``molforge`` is running
            in). Override when RFAA lives in a different conda env.
        max_cycle: Hydra override for ``loader_params.MAXCYCLE``. The
            RFAA README recommends ``10`` for hard cases (default is
            4). ``None`` keeps the model default.
        job_name: Name used for output files. Defaults to
            ``"molforge_prediction"``.
        extra_overrides: Additional Hydra-style overrides (e.g.
            ``["recycling_steps=8"]``) passed verbatim to the CLI.
            Use this for any RFAA config knob the wrapper doesn't
            expose explicitly.

    Example:
        >>> from molforge.wrappers.folding import RoseTTAFold
        >>> engine = RoseTTAFold(repo_dir="/opt/RoseTTAFold-All-Atom",
        ...                      max_cycle=10)
        >>> protein = engine.predict("MKTVRQERLKSIVRILERSK")
        >>> protein.metadata["mean_confidence"]
        82.4
        >>> protein.metadata["pae_inter"]  # RFAA's headline confidence
        4.8
    """

    name = "RoseTTAFold"

    def __init__(
        self,
        *,
        repo_dir: str | None = None,
        python_executable: str | None = None,
        max_cycle: int | None = None,
        job_name: str = "molforge_prediction",
        extra_overrides: list[str] | None = None,
    ) -> None:
        self.repo_dir = repo_dir
        self.python_executable = python_executable
        self.max_cycle = max_cycle
        self.job_name = job_name
        self.extra_overrides = list(extra_overrides) if extra_overrides else []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, sequence: str, **kwargs: object) -> Protein:
        """Fold a single sequence into a :class:`Protein` via RFAA.

        Args:
            sequence: One-letter amino-acid sequence.
            **kwargs: Reserved for future per-call options.

        Returns:
            A :class:`Protein` with:

            - ``metadata["engine"] = "RoseTTAFold"``
            - ``metadata["source_sequence"]``: the input sequence
            - ``metadata["confidence_per_residue"]``: ``(L,)`` float32 pLDDT
            - ``metadata["confidence_per_atom"]``: ``(N_atoms,)`` float32 pLDDT
            - ``metadata["mean_confidence"]``: scalar float pLDDT (0–100)
            - ``metadata["pae"]``: ``(L, L)`` predicted aligned error
                (only populated when the aux file is readable)
            - ``metadata["pae_inter"]``: scalar mean inter-frame PAE
                (RFAA's headline interface-quality metric; < 10 typically
                indicates high quality)
            - ``metadata["mean_pae"]``: scalar mean PAE over the matrix
            - ``metadata["pae_prot"]``: scalar mean PAE over protein-only
                residues

        Raises:
            FoldingEngineNotInstalledError: If ``repo_dir`` isn't set
                (via constructor or ``RFAA_HOME``) or doesn't exist.
            RuntimeError: If the CLI fails or produces no output.
        """
        sequence = _validate_sequence(sequence)
        return self._run_local(sequence)

    # ------------------------------------------------------------------
    # Local-execution path (testable seam)
    # ------------------------------------------------------------------
    def _run_local(self, sequence: str) -> Protein:
        """Drive the RFAA CLI in a temporary directory and parse the result."""
        repo = self._require_rfaa()

        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            # RFAA needs the FASTA on disk, and the Hydra config has to
            # point at a specific file path. We stash both inside a
            # working directory and run the inference from there.
            workdir = tmpdir / "work"
            workdir.mkdir()
            fasta_path = workdir / "query.fasta"
            fasta_path.write_text(f">query\n{sequence}\n", encoding="utf-8")

            config_dir = workdir / "configs"
            config_dir.mkdir()
            config_path = config_dir / f"{self.job_name}.yaml"
            config_path.write_text(
                self._build_config(fasta_path=fasta_path),
                encoding="utf-8",
            )

            cmd = self._build_command(config_dir=config_dir)
            env = self._build_env()
            self._invoke(cmd, cwd=workdir, env=env, repo_dir=repo)

            pdb_path, aux_path = self._collect_outputs(workdir)
            pdb_text = pdb_path.read_text(encoding="utf-8")
            confidence = (
                self._load_aux_file(aux_path) if aux_path is not None else {}
            )

        return self._parse_outputs(
            pdb_text=pdb_text,
            confidence=confidence,
            sequence=sequence,
        )

    # ------------------------------------------------------------------
    # Process plumbing (each step a testable seam)
    # ------------------------------------------------------------------
    def _require_rfaa(self) -> Path:
        """Resolve the RFAA repo directory or raise a clean error."""
        repo_dir = self.repo_dir or os.environ.get("RFAA_HOME")
        if not repo_dir:
            raise FoldingEngineNotInstalledError(
                "RoseTTAFold All-Atom requires a cloned repo directory.\n"
                "Set $RFAA_HOME or pass repo_dir=... to the constructor.\n"
                "See https://github.com/baker-laboratory/RoseTTAFold-All-Atom "
                "for setup instructions (note: ~400 GB of database "
                "downloads required)."
            )
        repo = Path(repo_dir)
        if not repo.is_dir():
            raise FoldingEngineNotInstalledError(
                f"RFAA repo_dir={repo!s} is not a directory. "
                "Clone https://github.com/baker-laboratory/RoseTTAFold-All-Atom."
            )
        if not (repo / "rf2aa").is_dir():
            raise FoldingEngineNotInstalledError(
                f"RFAA repo_dir={repo!s} does not contain a 'rf2aa/' "
                "subdirectory. Make sure repo_dir points at the cloned "
                "RoseTTAFold-All-Atom repository root."
            )
        return repo

    def _build_config(self, *, fasta_path: Path) -> str:
        """Construct the Hydra config YAML for a single-chain protein job.

        Mirrors ``rf2aa/config/inference/protein.yaml`` from the RFAA
        repo, inheriting all the database / weights paths from the
        installed-default ``base`` config the user has already
        configured.
        """
        lines = [
            "defaults:",
            "  - base",
            "",
            f'job_name: "{self.job_name}"',
            "protein_inputs:",
            "  A:",
            f"    fasta_file: {fasta_path}",
        ]
        if self.max_cycle is not None:
            lines.extend([
                "",
                "loader_params:",
                f"  MAXCYCLE: {self.max_cycle}",
            ])
        return "\n".join(lines) + "\n"

    def _build_command(self, *, config_dir: Path) -> list[str]:
        """Assemble the ``python -m rf2aa.run_inference ...`` command line."""
        python = self.python_executable or sys.executable
        cmd: list[str] = [
            python,
            "-m",
            "rf2aa.run_inference",
            "--config-dir",
            str(config_dir),
            "--config-name",
            self.job_name,
        ]
        # Hydra overrides go on the end, positional. Use 'append' syntax
        # for adding non-overrides (none in our case) and bare key=value
        # for value overrides.
        cmd.extend(self.extra_overrides)
        return cmd

    def _build_env(self) -> dict[str, str]:
        """Prepare environment variables for the subprocess.

        Propagates the parent environment. RFAA reads its database
        paths from variables that should already be set by the user
        (``DB_UR30``, ``DB_BFD``, ``BLASTMAT``); we don't override
        them.
        """
        return dict(os.environ)

    def _invoke(
        self,
        cmd: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        repo_dir: Path,
    ) -> None:
        """Run the subprocess and raise on non-zero exit.

        Runs from ``cwd`` but with ``PYTHONPATH`` extended to include
        the RFAA repo, so ``python -m rf2aa.run_inference`` finds the
        module even when the user hasn't installed it system-wide.
        """
        env = dict(env)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{repo_dir}{os.pathsep}{existing}" if existing else str(repo_dir)
        )
        try:
            subprocess.run(  # noqa: S603 - inputs are constructed internally
                cmd,
                check=True,
                cwd=str(cwd),
                env=env,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"`python -m rf2aa.run_inference` failed (exit code {e.returncode}).\n"
                f"stderr:\n{e.stderr}\n"
                f"stdout:\n{e.stdout}"
            ) from e

    def _collect_outputs(
        self,
        workdir: Path,
    ) -> tuple[Path, Path | None]:
        """Locate the PDB output and the optional aux confidence file.

        RFAA writes ``<job_name>.pdb`` and ``<job_name>_aux.pt`` into
        the working directory by default.
        """
        pdb_candidates = sorted(workdir.rglob(f"{self.job_name}.pdb")) or sorted(
            workdir.rglob("*.pdb")
        )
        if not pdb_candidates:
            raise RuntimeError(
                f"RFAA produced no .pdb output in {workdir}. "
                "Check the RFAA logs for the actual error."
            )

        aux_candidates = sorted(workdir.rglob(f"{self.job_name}_aux.pt")) or sorted(
            workdir.rglob("*_aux.pt")
        )
        return pdb_candidates[0], (aux_candidates[0] if aux_candidates else None)

    def _load_aux_file(self, aux_path: Path) -> dict[str, object]:
        """Load the RFAA confidence aux file.

        The aux file is a PyTorch state dict with confidence tensors.
        We try to load it with :func:`torch.load`; if torch isn't
        installed in the calling environment we still return the PDB-
        derived confidence so the wrapper degrades gracefully.
        """
        try:
            import torch
        except ImportError:
            return {}
        try:
            data = torch.load(aux_path, map_location="cpu", weights_only=False)
        except (RuntimeError, OSError, ValueError):
            # If the aux file is malformed we still want the PDB.
            return {}
        # Convert torch tensors to numpy for the consumer.
        result: dict[str, object] = {}
        for key, value in data.items():
            if hasattr(value, "detach"):
                result[key] = value.detach().cpu().numpy()
            else:
                result[key] = value
        return result

    # ------------------------------------------------------------------
    # Output parsing (testable in isolation)
    # ------------------------------------------------------------------
    def _parse_outputs(
        self,
        *,
        pdb_text: str,
        confidence: dict[str, object],
        sequence: str,
    ) -> Protein:
        """Parse RFAA PDB + aux confidence into a Protein with metadata.

        RFAA writes per-atom pLDDT to the B-factor column of the PDB
        (same as AlphaFold, ESMFold, and Boltz). We follow molforge's
        uniform confidence convention and additionally expose the
        RFAA-specific PAE matrix and scalars.
        """
        from molforge.io.pdb import read_pdb_string

        protein = read_pdb_string(pdb_text)
        arr = protein.atom_array
        plddt_per_atom = np.asarray(arr.b_factor, dtype=np.float32).copy()

        per_residue: list[float] = []
        for sl in arr.iter_residue_slices():
            per_residue.append(float(plddt_per_atom[sl].mean()))
        per_residue_arr = np.asarray(per_residue, dtype=np.float32)
        mean_conf = (
            float(per_residue_arr.mean()) if per_residue_arr.size else 0.0
        )

        meta: dict[str, object] = {
            mk.ENGINE: "RoseTTAFold",
            mk.SOURCE_SEQUENCE: sequence,
            mk.JOB_NAME: self.job_name,
            mk.CONFIDENCE_PER_ATOM: plddt_per_atom,
            mk.CONFIDENCE_PER_RESIDUE: per_residue_arr,
            mk.MEAN_CONFIDENCE: mean_conf,
        }

        # Surface the RFAA-specific tensors when available.
        for key in (mk.PAE, mk.PDE, mk.MEAN_PAE, mk.PAE_PROT, mk.PAE_INTER, mk.MEAN_PLDDT):
            if key in confidence:
                meta[key] = confidence[key]

        protein.metadata.update(meta)
        return protein
