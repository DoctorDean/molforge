"""DiffDock molecular-docking wrapper.

[DiffDock](https://github.com/gcorso/DiffDock) (Corso et al. 2023,
*ICLR*) is a diffusion-generative model for blind protein-ligand
docking. Unlike search-based engines such as AutoDock Vina, it does
**not** need a search box — it samples ligand poses directly over the
whole receptor and ranks them with a learned confidence model.

DiffDock ships as a research repository, not a pip package: the model
runs via ``python -m inference`` from a cloned checkout of the repo
(plus its model weights). This wrapper therefore drives DiffDock as a
subprocess — the same shape as the RoseTTAFold wrapper — rather than
importing it as a library.

What the wrapper handles:
  - locating the cloned repo (``$DIFFDOCK_HOME`` or an explicit
    ``repo_dir``);
  - materializing the receptor to a PDB file and accepting the ligand
    as a SMILES string or a path to an SDF/MOL2 file;
  - assembling and running the ``inference`` command line;
  - parsing the ranked ``rank{N}_confidence{C}.sdf`` outputs into a
    :class:`DockingResult`, best (highest-confidence) pose first.

What it does **not** do:
  - install DiffDock or its weights — that is a manual, ~5 GB setup;
  - generate 3D conformers itself — DiffDock takes a SMILES or a 2D/3D
    ligand and handles embedding internally.

DiffDock's score is a *confidence* (higher = better), the opposite of
Vina's affinity convention (lower = better). To keep
:attr:`DockingResult.poses` uniformly sorted best-first, the wrapper
stores the raw confidence in ``Pose.metadata["confidence"]`` and sets
``Pose.score`` to the negated confidence — so ``score`` ascending is
best-first for every engine.

Installation::

    git clone https://github.com/gcorso/DiffDock
    # then download the model weights per the repo's README
    export DIFFDOCK_HOME=/path/to/DiffDock

For working docking without that setup, use
:class:`molforge.wrappers.docking.Vina`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from molforge.core import Protein
from molforge.core import metadata_keys as mk
from molforge.core.provenance import Provenance
from molforge.cache import get_default_cache
from molforge.docking import (
    DockingEngine,
    DockingEngineNotInstalledError,
    DockingResult,
    Pose,
)
from molforge.io.sdf import read_sdf_string

if TYPE_CHECKING:
    from os import PathLike


class DiffDock(DockingEngine):
    """Wrapper around DiffDock (diffusion-based blind docking).

    Args:
        repo_dir: Path to a cloned ``gcorso/DiffDock`` repository. If
            ``None``, the wrapper reads ``$DIFFDOCK_HOME``. Resolution
            is lazy — construction never touches the filesystem, so a
            ``DiffDock()`` instance is cheap to create in code paths
            that may never call :meth:`dock`.
        python_executable: Python interpreter used to run DiffDock's
            ``inference`` module. Defaults to ``sys.executable``;
            override when DiffDock's dependencies live in a separate
            environment.
        samples_per_complex: Number of poses DiffDock samples (and the
            cap on how many ranked poses are returned). DiffDock's
            default is 10.
        inference_steps: Reverse-diffusion steps. DiffDock's default
            is 20; more steps cost runtime roughly linearly.
        batch_size: Sampling batch size passed through to DiffDock.

    Example:
        >>> from molforge.wrappers.docking import DiffDock
        >>> import molforge as mf
        >>>
        >>> receptor = mf.load("receptor.pdb")
        >>> result = DiffDock().dock(
        ...     receptor=receptor,
        ...     ligand="CC(=O)Oc1ccccc1C(=O)O",  # aspirin SMILES
        ... )
        >>> best = result.best
        >>> best.metadata["confidence"]   # DiffDock confidence score
        0.74
    """

    name = "DiffDock"

    def __init__(
        self,
        *,
        repo_dir: str | PathLike[str] | None = None,
        python_executable: str | None = None,
        samples_per_complex: int = 10,
        inference_steps: int = 20,
        batch_size: int = 10,
    ) -> None:
        if samples_per_complex < 1:
            raise ValueError(f"samples_per_complex must be >= 1, got {samples_per_complex}")
        if inference_steps < 1:
            raise ValueError(f"inference_steps must be >= 1, got {inference_steps}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.repo_dir = Path(repo_dir) if repo_dir is not None else None
        self.python_executable = python_executable
        self.samples_per_complex = samples_per_complex
        self.inference_steps = inference_steps
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def dock(  # type: ignore[override]  # engine-specific kwargs refine the **kwargs ABC contract
        self,
        receptor: Protein | str | PathLike[str],
        ligand: str | PathLike[str],
        *,
        timeout: float | None = None,
        **_kwargs: object,
    ) -> DockingResult:
        """Dock ``ligand`` against ``receptor`` with DiffDock.

        Args:
            receptor: A :class:`Protein` or a path to a PDB file.
            ligand: Either a SMILES string or a path to an
                ``.sdf`` / ``.mol`` / ``.mol2`` file. A value that
                exists as a file is treated as a path; otherwise it is
                treated as SMILES.
            timeout: Optional subprocess timeout in seconds.
            **_kwargs: Reserved for future DiffDock options.

        Returns:
            A :class:`DockingResult` whose poses are sorted best-first.
            DiffDock reports a *confidence* (higher = better); each
            :class:`Pose` keeps the raw value in
            ``metadata["confidence"]`` and sets ``score`` to its
            negation, so ``score`` ascending is best-first — matching
            the convention used by :class:`Vina`.

        Raises:
            DockingEngineNotInstalledError: If the DiffDock repo cannot
                be located.
            RuntimeError: If the DiffDock subprocess fails or produces
                no output.
        """
        # Build Provenance upfront and consult the cache before
        # resolving the install or spawning the DiffDock subprocess —
        # an identical dock returns instantly, even on a machine
        # without DiffDock set up.
        provenance = self._build_provenance(receptor, ligand)
        cache = get_default_cache()
        cached: DockingResult | None = cache.get(provenance, "docking_result")
        if cached is not None:
            return cached
        repo = self._resolve_repo()
        result = self._run_cli(
            receptor=receptor,
            ligand=ligand,
            repo=repo,
            timeout=timeout,
            provenance=provenance,
        )
        cache.put(provenance, result, "docking_result")
        return result

    def _build_provenance(
        self,
        receptor: Protein | str | PathLike[str],
        ligand: str | PathLike[str],
    ) -> Provenance:
        """Construct the Provenance for a :meth:`dock` call.

        Pure function of inputs + constructor parameters, built upfront
        so the cache key and the stored result share one instance. The
        input refs match what :meth:`_parse_outputs` would otherwise
        derive (receptor name / path; ligand SMILES or path string).
        """
        receptor_ref = (
            (receptor.name or "<Protein>") if isinstance(receptor, Protein) else str(receptor)
        )
        parent = (
            receptor.metadata.get(mk.PROVENANCE) if isinstance(receptor, Protein) else None
        )
        return Provenance.from_engine(
            engine="DiffDock",
            parameters={
                "samples_per_complex": self.samples_per_complex,
                "inference_steps": self.inference_steps,
                "batch_size": self.batch_size,
                "repo_dir": str(self.repo_dir) if self.repo_dir else None,
                "python_executable": self.python_executable,
            },
            inputs={
                "receptor": receptor_ref,
                "ligand": str(ligand),
            },
            parent=parent if isinstance(parent, Provenance) else None,
        )

    # ------------------------------------------------------------------
    # Installation resolution
    # ------------------------------------------------------------------
    def _resolve_repo(self) -> Path:
        """Locate the cloned DiffDock repository or raise a clean error."""
        repo_dir = self.repo_dir or os.environ.get("DIFFDOCK_HOME")
        if not repo_dir:
            raise DockingEngineNotInstalledError(
                "DiffDock requires a cloned gcorso/DiffDock repository.\n"
                "Set $DIFFDOCK_HOME or pass repo_dir=... to the constructor.\n"
                "    git clone https://github.com/gcorso/DiffDock\n"
                "then download the model weights per the repo README.\n"
                "For working docking without that setup, use "
                "molforge.wrappers.docking.Vina."
            )
        repo = Path(repo_dir)
        if not repo.is_dir():
            raise DockingEngineNotInstalledError(
                f"DiffDock repo_dir={repo!s} is not a directory. "
                "Clone https://github.com/gcorso/DiffDock."
            )
        if not (repo / "inference.py").is_file():
            raise DockingEngineNotInstalledError(
                f"DiffDock repo_dir={repo!s} does not contain 'inference.py'. "
                "Make sure repo_dir points at the cloned DiffDock repository "
                "root."
            )
        return repo

    # ------------------------------------------------------------------
    # CLI invocation (testable seam)
    # ------------------------------------------------------------------
    def _run_cli(
        self,
        *,
        receptor: Protein | str | PathLike[str],
        ligand: str | PathLike[str],
        repo: Path,
        timeout: float | None,
        provenance: Provenance | None = None,
    ) -> DockingResult:
        """Run the DiffDock ``inference`` module and parse its output.

        Separated from :meth:`dock` so tests can mock the subprocess
        without repeating receptor/ligand materialization. A prebuilt
        ``provenance`` (passed by :meth:`dock`) is threaded into the
        parser; direct callers may omit it and one is built from the
        receptor/ligand refs.
        """
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            out_dir = tmpdir / "out"
            out_dir.mkdir()

            receptor_pdb = self._materialize_receptor(receptor, tmpdir)
            ligand_arg = self._ligand_argument(ligand)

            python = self.python_executable or sys.executable
            cmd = [
                python,
                "-m",
                "inference",
                "--protein_path",
                str(receptor_pdb),
                "--ligand_description",
                ligand_arg,
                "--out_dir",
                str(out_dir),
                "--samples_per_complex",
                str(self.samples_per_complex),
                "--inference_steps",
                str(self.inference_steps),
                "--batch_size",
                str(self.batch_size),
            ]

            try:
                subprocess.run(
                    cmd,
                    check=True,
                    cwd=str(repo),
                    timeout=timeout,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"DiffDock failed (exit {e.returncode}).\n"
                    f"stderr:\n{e.stderr}\n"
                    f"stdout:\n{e.stdout}"
                ) from e

            return self._parse_outputs(
                out_dir,
                receptor=receptor if isinstance(receptor, Protein) else None,
                receptor_ref=(
                    receptor.name or "<Protein>" if isinstance(receptor, Protein) else str(receptor)
                ),
                ligand_ref=str(ligand),
                provenance_parent=(
                    receptor.metadata.get(mk.PROVENANCE) if isinstance(receptor, Protein) else None
                ),
                provenance=provenance,
            )

    # ------------------------------------------------------------------
    # Receptor / ligand materialization
    # ------------------------------------------------------------------
    @staticmethod
    def _materialize_receptor(
        receptor: Protein | str | PathLike[str],
        tmpdir: Path,
    ) -> Path:
        """Return a filesystem path to the receptor PDB.

        A :class:`Protein` is written to a temp PDB; a path is used
        as-is.
        """
        if isinstance(receptor, Protein):
            from molforge.io import write_pdb

            pdb_path = tmpdir / "receptor.pdb"
            write_pdb(receptor, pdb_path)
            return pdb_path
        return Path(receptor).resolve()

    @staticmethod
    def _ligand_argument(ligand: str | PathLike[str]) -> str:
        """Resolve the ``--ligand_description`` argument.

        DiffDock accepts either a SMILES string or a path to a ligand
        file in this one argument. A value that exists as a file on
        disk is passed as an absolute path; anything else is assumed
        to be SMILES and passed through verbatim.
        """
        candidate = Path(ligand)
        if candidate.exists():
            return str(candidate.resolve())
        return str(ligand)

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------
    def _parse_outputs(
        self,
        out_dir: Path,
        *,
        receptor: Protein | None,
        receptor_ref: str | None = None,
        ligand_ref: str | None = None,
        provenance_parent: Provenance | None = None,
        provenance: Provenance | None = None,
    ) -> DockingResult:
        """Parse DiffDock's ranked SDF poses into a DockingResult.

        DiffDock writes one subdirectory per docked complex, each
        containing ``rank{N}_confidence{C}.sdf`` files (and a
        ``rank{N}.sdf`` for the top pose in some versions). The rank
        index and confidence are encoded in the filename.
        """
        sdfs = sorted(out_dir.rglob("rank*.sdf"))
        if not sdfs:
            raise RuntimeError(
                f"DiffDock produced no rank*.sdf output in {out_dir}. "
                "Check the DiffDock logs for the actual error."
            )

        poses: list[Pose] = []
        for sdf in sdfs:
            confidence = _confidence_from_filename(sdf.name)
            mols = read_sdf_string(sdf.read_text(encoding="utf-8"))
            if not mols:
                continue
            ligand = mols[0]
            # DiffDock confidence is higher = better; negate it so
            # `score` ascending is best-first like every other engine.
            score = -confidence if confidence is not None else 0.0
            poses.append(
                Pose(
                    ligand=ligand,
                    score=score,
                    metadata={
                        "engine": "DiffDock",
                        "confidence": confidence,
                        "source_file": sdf.name,
                    },
                )
            )

        poses.sort(key=lambda p: p.score)
        for i, p in enumerate(poses):
            p.rank = i

        # Build result-level metadata, with Provenance layered on top
        # of the ad-hoc config-echo keys for backwards compatibility.
        # A prebuilt ``provenance`` (passed by dock() so the cache key
        # and the stored result share one instance) wins; otherwise the
        # call-site refs (receptor_ref / ligand_ref) drive a fresh one.
        # Tests that call _parse_outputs directly without either still
        # produce a result, just without provenance attached.
        result_metadata: dict[str, object] = {
            "samples_per_complex": self.samples_per_complex,
            "inference_steps": self.inference_steps,
        }
        if provenance is not None:
            result_metadata[mk.PROVENANCE] = provenance
        elif receptor_ref is not None or ligand_ref is not None:
            result_metadata[mk.PROVENANCE] = Provenance.from_engine(
                engine="DiffDock",
                parameters={
                    "samples_per_complex": self.samples_per_complex,
                    "inference_steps": self.inference_steps,
                    "batch_size": self.batch_size,
                    "repo_dir": str(self.repo_dir) if self.repo_dir else None,
                    "python_executable": self.python_executable,
                },
                inputs={
                    "receptor": receptor_ref,
                    "ligand": ligand_ref,
                },
                parent=provenance_parent,
            )

        return DockingResult(
            poses=poses,
            receptor=receptor,
            engine="DiffDock",
            metadata=result_metadata,
        )


# ----------------------------------------------------------------------
# Module-level parsing helpers
# ----------------------------------------------------------------------
def _confidence_from_filename(name: str) -> float | None:
    """Extract the confidence value from a DiffDock SDF filename.

    DiffDock names ranked poses ``rank{N}_confidence{C}.sdf`` where
    ``C`` is a (possibly negative) float, e.g.
    ``rank1_confidence-0.42.sdf``. The top pose is sometimes written as
    a bare ``rank1.sdf`` with no confidence — that yields ``None``.
    """
    stem = name.rsplit(".", 1)[0]
    marker = "confidence"
    idx = stem.find(marker)
    if idx == -1:
        return None
    value = stem[idx + len(marker) :]
    try:
        return float(value)
    except ValueError:
        return None
