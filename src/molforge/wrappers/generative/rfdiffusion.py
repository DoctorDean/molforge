"""RFdiffusion wrapper for *de novo* protein backbone generation.

Reference: Watson, J. L. et al. (2023) "De novo design of protein
structure and function with RFdiffusion." *Nature* 620: 1089-1100.

RFdiffusion is a diffusion model trained on the RoseTTAFold structure
prediction network. It generates novel protein backbones either
unconditionally (just specify a length) or conditionally on a motif,
target, or symmetry constraint. The output is a backbone-only PDB
(poly-glycine, since RFdiffusion only generates the C-alpha trace plus
backbone atoms — no side chains). Sequence design is then done by a
separate tool, typically :class:`ProteinMPNN`.

What this wrapper supports:

  - **Unconditional generation**: produce ``N`` random backbones of a
    requested length range.
  - **Motif scaffolding**: given an input PDB, scaffold the specified
    residues into a new backbone of a chosen length.
  - **Binder design**: given a target, generate a complementary
    binder of a chosen length.
  - **Symmetric design**: cyclic, dihedral, or tetrahedral symmetry.

This wrapper invokes the official ``scripts/run_inference.py`` from
the [RosettaCommons/RFdiffusion](https://github.com/RosettaCommons/RFdiffusion)
repository via subprocess. RFdiffusion uses
[Hydra](https://hydra.cc/) for config; we translate Python kwargs into
Hydra's ``key=value`` syntax internally so users don't need to know
the syntax.

Installation
------------

There's no pip-installable RFdiffusion; you need the GitHub repo plus
model weights::

    git clone https://github.com/RosettaCommons/RFdiffusion
    cd RFdiffusion
    pip install -e .
    bash scripts/download_models.sh models

Then either set ``RFDIFFUSION_HOME`` to that clone, or pass
``rfdiffusion_dir="/path/to/RFdiffusion"`` to the constructor.

Memory
------

RFdiffusion needs ~10 GB of GPU memory for 150-residue designs. CPU
inference works but is roughly 50x slower.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from molforge.core import metadata_keys as mk
from molforge.core.provenance import Provenance
from molforge.generative import GenerativeEngine, GenerativeEngineNotInstalledError

if TYPE_CHECKING:
    from molforge.core import Protein


class RFdiffusion(GenerativeEngine):
    """RFdiffusion backbone generator.

    Args:
        rfdiffusion_dir: Path to the cloned RosettaCommons/RFdiffusion
            repository. If ``None``, reads ``RFDIFFUSION_HOME`` from
            the environment.
        python_executable: Python interpreter to invoke. Default
            ``sys.executable``. Override if RFdiffusion's dependencies
            live in a different env.
        num_designs: How many backbones to generate per call.
        diffusion_steps: Reverse-diffusion steps (default 50, RFdiffusion's
            standard).
        device: ``"cuda"``, ``"cpu"``, or ``None`` to use the default.
        config_name: Hydra config file name. Defaults to ``base``;
            use ``symmetry`` for symmetric designs.

    Example:
        >>> from molforge.wrappers.generative import RFdiffusion
        >>> engine = RFdiffusion(num_designs=4)
        >>>
        >>> # Unconditional generation: 4 backbones of length 100
        >>> backbones = engine.generate(length=100)
        >>> len(backbones)
        4
        >>>
        >>> # Motif scaffolding from a target PDB
        >>> backbones = engine.generate(
        ...     target_pdb="my_motif.pdb",
        ...     contigs=["10-40/A20-35/10-40"],
        ... )

    Note:
        RFdiffusion outputs are backbone-only (N/CA/C/O, no side chains,
        all residues labeled GLY). Run :class:`ProteinMPNN` on each
        backbone to get a designable sequence.
    """

    name = "RFdiffusion"

    def __init__(
        self,
        *,
        rfdiffusion_dir: str | os.PathLike[str] | None = None,
        python_executable: str | None = None,
        num_designs: int = 1,
        diffusion_steps: int = 50,
        device: str | None = None,
        config_name: str = "base",
    ) -> None:
        self.rfdiffusion_dir = Path(rfdiffusion_dir) if rfdiffusion_dir is not None else None
        self.python_executable = python_executable
        self.num_designs = num_designs
        self.diffusion_steps = diffusion_steps
        self.device = device
        self.config_name = config_name

    # ------------------------------------------------------------------
    # Locate the installation
    # ------------------------------------------------------------------
    def _resolve_rfdiffusion_dir(self) -> Path:
        """Return the path to the RFdiffusion repo, or raise."""
        if self.rfdiffusion_dir is not None:
            d = Path(self.rfdiffusion_dir)
        else:
            env = os.environ.get("RFDIFFUSION_HOME")
            if not env:
                raise GenerativeEngineNotInstalledError(
                    "RFdiffusion not found. Either pass "
                    "`rfdiffusion_dir=` to the RFdiffusion(...) constructor, or "
                    "set the RFDIFFUSION_HOME environment variable to the "
                    "cloned RosettaCommons/RFdiffusion repo. Install with:\n"
                    "    git clone https://github.com/RosettaCommons/RFdiffusion\n"
                    "    cd RFdiffusion && pip install -e .\n"
                    "    bash scripts/download_models.sh models"
                )
            d = Path(env)

        # Sanity check: the repo's inference script should exist
        inference_script = d / "scripts" / "run_inference.py"
        if not inference_script.exists():
            raise GenerativeEngineNotInstalledError(
                f"RFdiffusion directory {d} doesn't contain "
                f"scripts/run_inference.py. Re-clone the repo or fix the path."
            )
        return d

    # ------------------------------------------------------------------
    # Build the Hydra command-line args
    # ------------------------------------------------------------------
    def _build_hydra_args(
        self,
        *,
        output_prefix: Path,
        length: int | None = None,
        target_pdb: str | os.PathLike[str] | None = None,
        contigs: Sequence[str] | None = None,
        hotspot_residues: Sequence[str] | None = None,
        symmetry: str | None = None,
        extra: dict[str, str] | None = None,
    ) -> list[str]:
        """Translate Python kwargs into RFdiffusion's Hydra `key=value` args."""
        args: list[str] = [
            f"--config-name={self.config_name}",
            f"inference.output_prefix={output_prefix}",
            f"inference.num_designs={self.num_designs}",
            f"diffuser.T={self.diffusion_steps}",
        ]
        if target_pdb is not None:
            args.append(f"inference.input_pdb={Path(target_pdb).resolve()}")
        if contigs is not None:
            # Hydra wants the list as a single string with [...] wrapping
            contigs_str = "[" + ",".join(contigs) + "]"
            args.append(f"contigmap.contigs={contigs_str}")
        elif length is not None:
            args.append(f"contigmap.contigs=[{length}-{length}]")
        if hotspot_residues is not None:
            hot_str = "[" + ",".join(hotspot_residues) + "]"
            args.append(f"ppi.hotspot_res={hot_str}")
        if symmetry is not None:
            args.append(f"inference.symmetry={symmetry}")
        if extra is not None:
            for k, v in extra.items():
                args.append(f"{k}={v}")
        return args

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(  # type: ignore[override]  # engine-specific kwargs + refined return type vs the ABC
        self,
        *,
        length: int | None = None,
        target_pdb: str | os.PathLike[str] | None = None,
        contigs: Sequence[str] | None = None,
        hotspot_residues: Sequence[str] | None = None,
        symmetry: str | None = None,
        extra_hydra_args: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> list[Protein]:
        """Generate ``num_designs`` backbones.

        Args:
            length: Number of residues for unconditional generation.
                Mutually exclusive with ``contigs``.
            target_pdb: Path to an input PDB for motif scaffolding or
                binder design.
            contigs: Hydra-style contig list, e.g.
                ``["10-40/A20-35/10-40"]`` for motif scaffolding. See
                the RFdiffusion README for the full grammar.
            hotspot_residues: For binder design, residues on the
                target that the binder should contact (e.g.
                ``["A32", "A33", "A34"]``).
            symmetry: For symmetric design — ``"cyclic"``,
                ``"dihedral"``, ``"tetrahedral"``, etc. Requires
                ``config_name="symmetry"`` on the engine.
            extra_hydra_args: Additional ``key=value`` overrides
                passed directly to Hydra. Use this for any setting
                not exposed above.
            timeout: Optional subprocess timeout in seconds.

        Returns:
            List of :class:`Protein` instances, one per design.
            Each ``Protein`` has ``metadata["engine"] = "RFdiffusion"``
            and ``metadata["source_args"]`` recording the call.

        Raises:
            GenerativeEngineNotInstalledError: If RFdiffusion can't be
                found.
            ValueError: For incompatible argument combinations.
            subprocess.CalledProcessError: If RFdiffusion errors out.
        """
        if length is not None and contigs is not None:
            raise ValueError("pass either `length` or `contigs`, not both")
        if length is None and contigs is None and target_pdb is None:
            raise ValueError("specify at least one of `length`, `contigs`, or `target_pdb`")

        rfdir = self._resolve_rfdiffusion_dir()
        return self._run_cli(
            rfdiffusion_dir=rfdir,
            length=length,
            target_pdb=target_pdb,
            contigs=contigs,
            hotspot_residues=hotspot_residues,
            symmetry=symmetry,
            extra_hydra_args=extra_hydra_args,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # CLI invocation (testable seam)
    # ------------------------------------------------------------------
    def _run_cli(
        self,
        *,
        rfdiffusion_dir: Path,
        length: int | None,
        target_pdb: str | os.PathLike[str] | None,
        contigs: Sequence[str] | None,
        hotspot_residues: Sequence[str] | None,
        symmetry: str | None,
        extra_hydra_args: dict[str, str] | None,
        timeout: float | None,
    ) -> list[Protein]:
        """Run the RFdiffusion CLI and parse outputs into Proteins.

        Separated from :meth:`generate` so tests can mock this without
        repeating arg validation.
        """
        import sys

        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            output_prefix = tmpdir / "design"

            hydra_args = self._build_hydra_args(
                output_prefix=output_prefix,
                length=length,
                target_pdb=target_pdb,
                contigs=contigs,
                hotspot_residues=hotspot_residues,
                symmetry=symmetry,
                extra=extra_hydra_args,
            )

            python = self.python_executable or sys.executable
            script = rfdiffusion_dir / "scripts" / "run_inference.py"
            cmd = [python, str(script), *hydra_args]

            try:
                subprocess.run(
                    cmd,
                    check=True,
                    cwd=str(rfdiffusion_dir),
                    timeout=timeout,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"RFdiffusion failed (exit {e.returncode}). "
                    f"Command: {shlex.join(cmd)}\n"
                    f"stderr: {e.stderr}"
                ) from e

            return self._parse_outputs(
                tmpdir,
                source_args={
                    "length": length,
                    "target_pdb": str(target_pdb) if target_pdb else None,
                    "contigs": list(contigs) if contigs else None,
                    "hotspot_residues": (list(hotspot_residues) if hotspot_residues else None),
                    "symmetry": symmetry,
                    "diffusion_steps": self.diffusion_steps,
                    "num_designs": self.num_designs,
                },
            )

    # ------------------------------------------------------------------
    # Output parsing (testable in isolation)
    # ------------------------------------------------------------------
    def _parse_outputs(
        self,
        tmpdir: Path,
        *,
        source_args: dict[str, object],
    ) -> list[Protein]:
        """Load the design_N.pdb files RFdiffusion produces."""
        from molforge.io import read_pdb

        pdbs = sorted(tmpdir.glob("design_*.pdb"))
        if not pdbs:
            raise RuntimeError(
                f"RFdiffusion produced no PDB output in {tmpdir}. "
                "Check the RFdiffusion logs for the actual error."
            )

        designs: list[Protein] = []
        for i, pdb in enumerate(pdbs):
            protein = read_pdb(pdb)
            # Each design is its own independent output, so each gets
            # its own Provenance. The engine parameters are the same
            # across designs (same call, same config); design_index
            # stays as a separate metadata key — it identifies *which*
            # of N designs this is, not part of the engine config.
            prov = Provenance.from_engine(
                engine="RFdiffusion",
                parameters={
                    **source_args,
                    "rfdiffusion_dir": (
                        str(self.rfdiffusion_dir) if self.rfdiffusion_dir else None
                    ),
                    "python_executable": self.python_executable,
                    "device": self.device,
                },
                inputs=(
                    {"target_pdb": source_args["target_pdb"]}
                    if source_args.get("target_pdb")
                    else {}
                ),
            )
            protein.metadata.update(
                {
                    mk.PROVENANCE: prov,
                    "engine": "RFdiffusion",
                    "design_index": i,
                    "source_args": source_args,
                }
            )
            designs.append(protein)
        return designs
