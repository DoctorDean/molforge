"""ProteinMPNN wrapper for protein sequence design.

Reference: Dauparas, J. et al. (2022) "Robust deep learning-based
protein sequence design using ProteinMPNN." *Science* 378: 49-56.

ProteinMPNN is a message-passing neural network that, given a protein
backbone, proposes amino-acid sequences that should fold to it. It's
the standard *inverse-folding* tool — the natural follow-up to a
backbone generator like :class:`RFdiffusion`. The typical workflow:

  1. Generate or load a backbone (`Protein` with N/CA/C/O atoms).
  2. Call :meth:`generate` to sample sequences from ProteinMPNN.
  3. (Recommended) Fold each sequence with ESMFold and check that
     the resulting structure matches the input backbone (low RMSD).
  4. Score the designs by ProteinMPNN's per-sequence score plus
     downstream metrics (TM-score, lDDT, DockQ).

What this wrapper supports:

  - **Monomer design**: design every residue of a single chain.
  - **Multi-chain design**: choose which chains are designable and
    which serve as fixed context (e.g. binder design).
  - **Fixed positions**: keep specific residues at their wild-type
    identities (e.g. preserve a known motif or active-site residues).
  - **Sampling control**: temperature, number of sequences, omitted
    amino acids.
  - **Model variants**: vanilla (full-backbone), CA-only,
    soluble-only.

This wrapper invokes the official ``protein_mpnn_run.py`` from the
[dauparas/ProteinMPNN](https://github.com/dauparas/ProteinMPNN)
repository via subprocess. The Python API is also available via the
``protein-mpnn-pip`` PyPI package, but the CLI is the more stable
interface.

Installation
------------

Either clone the GitHub repo::

    git clone https://github.com/dauparas/ProteinMPNN

and pass ``proteinmpnn_dir="/path/to/ProteinMPNN"`` to the constructor,
or set ``PROTEINMPNN_HOME``.

Or install the pip package (less common but works)::

    pip install protein-mpnn-pip
"""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from molforge.generative import (
    DesignedSequence,
    GenerativeEngine,
    GenerativeEngineNotInstalledError,
)

if TYPE_CHECKING:
    from molforge.core import Protein


# Valid model names per the official repo
_VALID_MODELS = ("v_48_002", "v_48_010", "v_48_020", "v_48_030")


class ProteinMPNN(GenerativeEngine):
    """ProteinMPNN sequence-design engine.

    Args:
        proteinmpnn_dir: Path to the cloned dauparas/ProteinMPNN repo.
            If ``None``, reads ``PROTEINMPNN_HOME`` from the environment.
        python_executable: Python interpreter. Default ``sys.executable``.
        model_name: Which checkpoint to use. ``v_48_020`` is the
            standard default (48 edges, 0.20 Å backbone noise during
            training). Lower noise (``v_48_002``) for very high-
            quality backbones; higher noise (``v_48_030``) for noisier
            inputs.
        use_soluble_model: Use the soluble-protein-only checkpoint.
            Better for designs intended to be soluble.
        ca_only: Use the CA-only checkpoint. Required if your input
            structure is CA-only.
        num_seqs: How many sequences to sample per call. ProteinMPNN
            samples are independent, so more samples = better
            sequence-recovery odds.
        sampling_temp: Sampling temperature. The paper recommends
            0.1 for high-fidelity recovery, 0.2-0.3 for diversity.
        omit_aas: String of one-letter codes to omit from the
            generated sequences. Default ``"X"`` (don't sample the
            unknown token); add ``"C"`` to avoid cysteines, etc.
        seed: Random seed. ``0`` uses a fresh random seed.

    Example:
        >>> from molforge.wrappers.generative import ProteinMPNN
        >>> from molforge.io import read_pdb
        >>>
        >>> backbone = read_pdb("backbone.pdb")
        >>> engine = ProteinMPNN(num_seqs=8, sampling_temp=0.1)
        >>> designs = engine.generate(backbone)
        >>>
        >>> # Each design is a DesignedSequence
        >>> best = min(designs, key=lambda d: d.score)
        >>> print(f"{best.sequence}  (score {best.score:.3f})")
    """

    name = "ProteinMPNN"

    def __init__(
        self,
        *,
        proteinmpnn_dir: str | os.PathLike[str] | None = None,
        python_executable: str | None = None,
        model_name: str = "v_48_020",
        use_soluble_model: bool = False,
        ca_only: bool = False,
        num_seqs: int = 8,
        sampling_temp: float = 0.1,
        omit_aas: str = "X",
        seed: int = 0,
    ) -> None:
        if model_name not in _VALID_MODELS:
            raise ValueError(f"unknown model_name {model_name!r}; expected one of {_VALID_MODELS}")
        if not (0.0 < sampling_temp <= 2.0):
            raise ValueError(f"sampling_temp must be in (0, 2], got {sampling_temp}")
        if num_seqs < 1:
            raise ValueError(f"num_seqs must be >= 1, got {num_seqs}")

        self.proteinmpnn_dir = Path(proteinmpnn_dir) if proteinmpnn_dir is not None else None
        self.python_executable = python_executable
        self.model_name = model_name
        self.use_soluble_model = use_soluble_model
        self.ca_only = ca_only
        self.num_seqs = num_seqs
        self.sampling_temp = sampling_temp
        self.omit_aas = omit_aas
        self.seed = seed

    # ------------------------------------------------------------------
    # Locate the installation
    # ------------------------------------------------------------------
    def _resolve_proteinmpnn_dir(self) -> Path:
        """Return the path to the ProteinMPNN repo, or raise."""
        if self.proteinmpnn_dir is not None:
            d = Path(self.proteinmpnn_dir)
        else:
            env = os.environ.get("PROTEINMPNN_HOME")
            if not env:
                raise GenerativeEngineNotInstalledError(
                    "ProteinMPNN not found. Either pass "
                    "`proteinmpnn_dir=` to the constructor, or set the "
                    "PROTEINMPNN_HOME environment variable to the cloned "
                    "dauparas/ProteinMPNN repo. Install with:\n"
                    "    git clone https://github.com/dauparas/ProteinMPNN"
                )
            d = Path(env)

        run_script = d / "protein_mpnn_run.py"
        if not run_script.exists():
            raise GenerativeEngineNotInstalledError(
                f"ProteinMPNN directory {d} doesn't contain "
                "protein_mpnn_run.py. Re-clone or fix the path."
            )
        return d

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(  # type: ignore[override]  # engine-specific kwargs + refined return type vs the ABC
        self,
        backbone: Protein | str | os.PathLike[str],
        *,
        chains_to_design: str | None = None,
        fixed_positions: dict[str, list[int]] | None = None,
        timeout: float | None = None,
    ) -> list[DesignedSequence]:
        """Design sequences for ``backbone``.

        Args:
            backbone: A :class:`molforge.core.Protein` or a path to a
                PDB file. The structure must have backbone atoms
                (N/CA/C/O); side chains are ignored.
            chains_to_design: Space-separated chain IDs to design
                (e.g. ``"A"`` or ``"A B"``). ``None`` = all chains.
                Other chains, if present, serve as fixed context.
            fixed_positions: Dict mapping chain ID to a list of
                **1-indexed** residue positions to keep at their
                wild-type identity. Example: ``{"A": [10, 11, 12]}``.
                Indices count from the first residue of the chain,
                not the PDB residue number.
            timeout: Optional subprocess timeout in seconds.

        Returns:
            A list of :class:`DesignedSequence` instances, sorted by
            score (lowest = best per ProteinMPNN's convention).

        Raises:
            GenerativeEngineNotInstalledError: If ProteinMPNN isn't found.
            RuntimeError: If the subprocess fails.
        """
        pmpnn_dir = self._resolve_proteinmpnn_dir()
        return self._run_cli(
            backbone=backbone,
            pmpnn_dir=pmpnn_dir,
            chains_to_design=chains_to_design,
            fixed_positions=fixed_positions,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # CLI invocation (testable seam)
    # ------------------------------------------------------------------
    def _run_cli(
        self,
        *,
        backbone: Protein | str | os.PathLike[str],
        pmpnn_dir: Path,
        chains_to_design: str | None,
        fixed_positions: dict[str, list[int]] | None,
        timeout: float | None,
    ) -> list[DesignedSequence]:
        """Run the ProteinMPNN CLI and parse outputs."""
        import sys

        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            out_folder = tmpdir / "out"
            out_folder.mkdir()

            # Materialize the backbone to a PDB file
            if hasattr(backbone, "atom_array"):
                from molforge.io import write_pdb

                pdb_path = tmpdir / "backbone.pdb"
                write_pdb(backbone, pdb_path)  # type: ignore[arg-type]
            else:
                pdb_path = Path(backbone).resolve()

            python = self.python_executable or sys.executable
            run_script = pmpnn_dir / "protein_mpnn_run.py"

            cmd = [
                python,
                str(run_script),
                "--pdb_path",
                str(pdb_path),
                "--out_folder",
                str(out_folder),
                "--num_seq_per_target",
                str(self.num_seqs),
                "--sampling_temp",
                str(self.sampling_temp),
                "--model_name",
                self.model_name,
                "--omit_AAs",
                self.omit_aas,
                "--seed",
                str(self.seed),
                "--batch_size",
                "1",
            ]
            if self.use_soluble_model:
                cmd.append("--use_soluble_model")
            if self.ca_only:
                cmd.append("--ca_only")
            if chains_to_design:
                cmd.extend(["--pdb_path_chains", chains_to_design])
            if fixed_positions:
                # ProteinMPNN reads fixed positions from a JSON file built
                # by helper_scripts/make_fixed_positions_dict.py. To keep the
                # wrapper self-contained, build the JSON directly.
                fp_path = tmpdir / "fixed_positions.jsonl"
                self._write_fixed_positions(fp_path, pdb_path, fixed_positions)
                cmd.extend(["--fixed_positions_jsonl", str(fp_path)])

            try:
                subprocess.run(
                    cmd,
                    check=True,
                    cwd=str(pmpnn_dir),
                    timeout=timeout,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"ProteinMPNN failed (exit {e.returncode}). "
                    f"Command: {shlex.join(cmd)}\n"
                    f"stderr: {e.stderr}"
                ) from e

            return self._parse_outputs(out_folder, pdb_path.stem)

    # ------------------------------------------------------------------
    # Output parsing (testable in isolation)
    # ------------------------------------------------------------------
    def _parse_outputs(
        self,
        out_folder: Path,
        pdb_stem: str,
    ) -> list[DesignedSequence]:
        """Parse the FASTA output ProteinMPNN writes.

        Output structure::

            out_folder/seqs/{pdb_stem}.fa

        Each FASTA record has a header like::

            >T=0.1, sample=1, score=1.23, global_score=1.45, ...
            DESIGNEDSEQUENCEHERE

        The first record is always the native sequence (or poly-G for
        an all-glycine backbone); we skip it and return the designs.
        """
        seqs_dir = out_folder / "seqs"
        candidates = list(seqs_dir.glob("*.fa")) + list(seqs_dir.glob("*.fasta"))
        if not candidates:
            raise RuntimeError(f"ProteinMPNN produced no FASTA output in {seqs_dir}")
        # If multiple files (multi-PDB run), pick the one matching pdb_stem
        fasta = next(
            (p for p in candidates if p.stem == pdb_stem),
            candidates[0],
        )
        text = fasta.read_text(encoding="utf-8")
        return self._parse_fasta(text)

    @staticmethod
    def _parse_fasta(text: str) -> list[DesignedSequence]:
        """Parse the FASTA text into a list of DesignedSequence."""
        records: list[tuple[str, str]] = []
        header: str | None = None
        body: list[str] = []
        for line in text.splitlines():
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(body)))
                header = line[1:]
                body = []
            elif header is not None:
                body.append(line.strip())
        if header is not None:
            records.append((header, "".join(body)))

        designs: list[DesignedSequence] = []
        # First record is the native; skip it.
        for hdr, seq in records[1:]:
            score = ProteinMPNN._parse_score(hdr)
            meta = ProteinMPNN._parse_metadata(hdr)
            designs.append(
                DesignedSequence(
                    sequence=seq,
                    score=score,
                    metadata=meta,
                )
            )
        # Sort by score, lowest first (best by convention)
        designs.sort(key=lambda d: d.score)
        return designs

    @staticmethod
    def _parse_score(header: str) -> float:
        """Extract the `score=...` value from a ProteinMPNN FASTA header."""
        for tok in header.split(","):
            tok = tok.strip()
            if tok.startswith("score="):
                return float(tok[len("score=") :])
        return float("nan")

    @staticmethod
    def _parse_metadata(header: str) -> dict[str, object]:
        """Pull all key=value pairs out of a ProteinMPNN FASTA header."""
        meta: dict[str, object] = {"engine": "ProteinMPNN"}
        for tok in header.split(","):
            tok = tok.strip()
            if "=" in tok:
                key, val = tok.split("=", 1)
                key = key.strip()
                val = val.strip()
                # Try numeric conversion
                try:
                    meta[key] = float(val)
                except ValueError:
                    meta[key] = val
        return meta

    # ------------------------------------------------------------------
    # Helper: write the fixed-positions JSONL
    # ------------------------------------------------------------------
    @staticmethod
    def _write_fixed_positions(
        path: Path,
        pdb_path: Path,
        fixed_positions: dict[str, list[int]],
    ) -> None:
        """Write ProteinMPNN's fixed-positions JSONL file.

        Format (per `make_fixed_positions_dict.py`)::

            {"<pdb_stem>": {"A": [1, 2, 5], "B": [...]}}
        """
        import json

        payload = {pdb_path.stem: fixed_positions}
        path.write_text(json.dumps(payload), encoding="utf-8")
