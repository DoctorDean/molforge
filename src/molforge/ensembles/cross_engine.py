"""Cross-engine folding ensembles: fold one sequence with several engines.

The single most useful thing you can do to trust a predicted structure is
to fold the sequence with more than one method and look at where they
*agree*. AlphaFold, ESMFold, Boltz and RoseTTAFold are trained on
overlapping but not identical data with different inductive biases; a
region all four place in the same spot is a region you can believe, and a
region where they scatter is one to treat with suspicion — independent of
any single model's self-reported confidence.

:func:`cross_engine_fold` is that workflow, once:

    from molforge.ensembles import cross_engine_fold

    ens = cross_engine_fold(sequence, engines=[esmfold, alphafold, boltz])
    ens.spread()          # pairwise TM / RMSD summary across the engines
    consensus = ens.consensus()   # the engine model most central to the rest
    ens.disagreement()    # per-residue CA spread — where the engines diverge

It is the transpose of :func:`molforge.parallel.fold_many` (one engine,
many sequences): here it's **one sequence, many engines**. The engines run
through :func:`~molforge.parallel.map_parallel`, defaulting to the
``"serial"`` backend — the safe choice when the members are GPU engines
that would otherwise contend for one device.

What comes back is a :class:`CrossEngineEnsemble`: the member structures
all superposed into a common frame, the pairwise TM-score and CA-RMSD
matrices, and the per-residue positional spread that is the real payload —
a direct, model-agnostic map of where the engines agree and where they
don't.

Scope (v1): single-chain (monomer) sequences. The signature already admits
a list of chain sequences for the complex case (cross-engine complex
ensembles, with per-interface agreement) — that path currently raises
:class:`NotImplementedError` so the surface is stable for when it lands.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from molforge.metrics import tm_score
from molforge.parallel import map_parallel
from molforge.structure import rmsd, superpose

if TYPE_CHECKING:
    from collections.abc import Sequence

    from molforge.core import Protein
    from molforge.parallel import Backend, OnError
    from molforge.wrappers.folding import FoldingEngine

__all__ = ["CrossEngineEnsemble", "cross_engine_fold"]

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CrossEngineEnsemble:
    """The result of folding one sequence with several engines.

    Members are all superposed into a common frame (the ``reference``
    chosen at fold time), so their coordinates are directly comparable and
    the ensemble can be written out or visualized as a single overlay.

    Attributes:
        sequence: The one-letter sequence every member folded.
        members: The predicted structures, superposed onto the reference,
            in the order the engines were supplied (failed / skipped
            engines removed). Each member carries
            ``metadata["cross_engine_rmsf"]`` (the per-residue spread) and
            ``metadata["cross_engine_ensemble"]`` (engine list + reference).
        engine_names: Names parallel to ``members``. Duplicates are
            disambiguated with a ``#2`` / ``#3`` suffix so every label is
            unique.
        reference_index: Index into ``members`` of the structure the others
            were superposed onto.
        tm_matrix: ``(N, N)`` pairwise TM-score matrix (1.0 on the
            diagonal). Higher = more similar fold.
        rmsd_matrix: ``(N, N)`` pairwise CA-RMSD matrix in Å (0.0 on the
            diagonal). Lower = more similar.
        per_residue_rmsf: ``(L,)`` float32 array of per-residue CA
            positional spread across the engines, after superposition — the
            root-mean-square fluctuation of each CA about its cross-engine
            mean position. This is the direct "where do the engines
            disagree" signal; also returned by :meth:`disagreement`.
    """

    sequence: str
    members: list[Protein]
    engine_names: list[str]
    reference_index: int
    tm_matrix: NDArray[np.float64]
    rmsd_matrix: NDArray[np.float64]
    per_residue_rmsf: NDArray[np.float32]

    @property
    def n_members(self) -> int:
        """Number of engine models that make up the ensemble."""
        return len(self.members)

    def consensus(self) -> Protein:
        """Return the medoid member — the model most central to the rest.

        The medoid is the member with the greatest summed TM-score to every
        other member: the single structure that best represents where the
        engines agree. It's a real, geometrically-valid model (not an
        averaged one), returned by reference from :attr:`members`.
        """
        idx = int(self.tm_matrix.sum(axis=1).argmax())
        return self.members[idx]

    def spread(self) -> dict[str, float]:
        """Summary statistics of the pairwise agreement across engines.

        A tight ensemble (mean pairwise TM near 1, RMSD near 0) means the
        engines converged; a loose one flags a sequence where methods
        disagree about the fold.

        Returns:
            A dict with ``tm_{min,max,mean,median,std}``,
            ``rmsd_{min,max,mean,median,std}`` over the off-diagonal
            (unordered pairs), and ``n_members``.
        """
        n = self.n_members
        if n < 2:
            keys = ("tm", "rmsd")
            stats = ("min", "max", "mean", "median", "std")
            out = {f"{k}_{s}": 0.0 for k in keys for s in stats}
            out["tm_min"] = out["tm_max"] = out["tm_mean"] = out["tm_median"] = 1.0
            out["n_members"] = float(n)
            return out
        iu = np.triu_indices(n, k=1)
        tm = self.tm_matrix[iu]
        rm = self.rmsd_matrix[iu]
        return {
            "tm_min": float(tm.min()),
            "tm_max": float(tm.max()),
            "tm_mean": float(tm.mean()),
            "tm_median": float(np.median(tm)),
            "tm_std": float(tm.std()),
            "rmsd_min": float(rm.min()),
            "rmsd_max": float(rm.max()),
            "rmsd_mean": float(rm.mean()),
            "rmsd_median": float(np.median(rm)),
            "rmsd_std": float(rm.std()),
            "n_members": float(n),
        }

    def disagreement(self) -> NDArray[np.float32]:
        """Per-residue CA spread across the engines (== :attr:`per_residue_rmsf`).

        Named for how it reads at the call site: high values are residues
        where the engines place the backbone differently (mobile loops,
        low-confidence termini); near-zero values are consensus regions.
        """
        return self.per_residue_rmsf

    def __repr__(self) -> str:
        return (
            f"CrossEngineEnsemble(engines={self.engine_names}, "
            f"L={len(self.sequence)}, reference={self.engine_names[self.reference_index]!r})"
        )


def cross_engine_fold(
    sequence: str | Sequence[str],
    engines: Sequence[FoldingEngine],
    *,
    reference: str = "medoid",
    workers: int | None = None,
    backend: Backend | None = None,
    on_error: OnError = "skip",
    **predict_kwargs: object,
) -> CrossEngineEnsemble:
    """Fold one sequence with several engines and compare the results.

    Runs ``engine.predict(sequence, **predict_kwargs)`` for each engine,
    superposes every model into one frame, and returns a
    :class:`CrossEngineEnsemble` with the pairwise TM / RMSD spread, a
    medoid consensus, and the per-residue cross-engine disagreement.

    Args:
        sequence: The one-letter amino-acid sequence to fold. A list of
            sequences (one per chain) is reserved for the complex case and
            currently raises :class:`NotImplementedError`.
        engines: Two or more folding engines. Any object with a
            ``predict(sequence, **kwargs) -> Protein`` method and a ``name``
            works; the concrete wrappers (ESMFold, AlphaFold, Boltz,
            RoseTTAFold) all qualify.
        reference: Which member defines the common superposition frame:

            - ``"medoid"`` (default) — the model most central to the rest
              (max summed TM-score). Frame-independent and robust.
            - ``"first"`` — the first engine supplied.
            - ``"most_confident"`` — the member with the highest
              ``metadata["mean_confidence"]``.
            - an engine name (as it appears in ``engine_names``) — anchor
              on that specific engine.
        workers: Passed to :func:`~molforge.parallel.map_parallel`.
        backend: Parallel backend. ``None`` (default) resolves to
            ``"process"`` only if *every* engine hints ``parallelism ==
            "process"``, otherwise ``"serial"`` — the safe default for GPU
            engines contending over one device.
        on_error: ``"skip"`` (default) drops engines that fail (logging a
            warning) and forms the ensemble from the rest; ``"raise"``
            propagates the first failure. An ensemble needs at least two
            surviving members either way.
        **predict_kwargs: Forwarded to every engine's ``predict``.

    Returns:
        A :class:`CrossEngineEnsemble`.

    Raises:
        NotImplementedError: If a multi-chain (complex) sequence is passed.
        ValueError: If fewer than two engines are supplied, fewer than two
            produce a structure, the members disagree on residue count, or
            ``reference`` is unrecognized.
    """
    chains = [sequence] if isinstance(sequence, str) else list(sequence)
    if len(chains) != 1:
        raise NotImplementedError(
            "cross_engine_fold v1 supports single-chain (monomer) folding; "
            "cross-engine complex ensembles are planned. Pass one sequence string."
        )
    seq = chains[0]

    engine_list = list(engines)
    if len(engine_list) < 2:
        raise ValueError(
            f"cross_engine_fold needs at least 2 engines to form an ensemble, "
            f"got {len(engine_list)}."
        )

    resolved_backend = _resolve_backend(engine_list, backend)
    # A module-level worker bound with partial (not a closure) so the
    # "process" backend can pickle it. The engine is the mapped item.
    fold = partial(_fold_one, seq, predict_kwargs, on_error)
    labeled: list[tuple[str, Protein | None]] = map_parallel(
        fold,
        engine_list,
        workers=workers,
        backend=resolved_backend,
        on_error="raise",  # policy is applied inside _fold_one
    )

    survivors = [(name, p) for name, p in labeled if p is not None]
    if len(survivors) < 2:
        succeeded = [name for name, p in labeled if p is not None]
        failed = [name for name, p in labeled if p is None]
        raise ValueError(
            f"cross_engine_fold: fewer than 2 engines produced a structure "
            f"(succeeded: {succeeded}, failed: {failed}). "
            "A cross-engine spread needs at least two members."
        )

    names = _dedupe_names([name for name, _ in survivors])
    members_raw = [p for _, p in survivors]

    ca_coords = [_ca_coords(p) for p in members_raw]
    length = ca_coords[0].shape[0]
    for name, ca in zip(names, ca_coords, strict=True):
        if ca.shape[0] != length:
            raise ValueError(
                f"engine {name!r} produced {ca.shape[0]} CA atoms but the first "
                f"member has {length}; all engines fold the same sequence, so CA "
                "counts must match. This usually means an engine returned a "
                "truncated, padded, or multi-chain model."
            )
    if length < 3:
        raise ValueError(f"cross-engine ensemble needs at least 3 residues, got {length}.")

    tm_matrix, rmsd_matrix = _pairwise_matrices(members_raw)
    ref_idx = _resolve_reference(reference, names, members_raw, tm_matrix)

    members, aligned_ca = _superpose_all(members_raw, ca_coords, ref_idx)
    rmsf = _per_residue_rmsf(aligned_ca)

    ensemble_tag = {"engines": list(names), "reference": names[ref_idx]}
    for member in members:
        member.metadata["cross_engine_ensemble"] = ensemble_tag
        member.metadata["cross_engine_rmsf"] = rmsf

    return CrossEngineEnsemble(
        sequence=seq,
        members=members,
        engine_names=names,
        reference_index=ref_idx,
        tm_matrix=tm_matrix,
        rmsd_matrix=rmsd_matrix,
        per_residue_rmsf=rmsf,
    )


# ---------- internals ----------


def _fold_one(
    sequence: str,
    predict_kwargs: dict[str, object],
    on_error: OnError,
    engine: FoldingEngine,
) -> tuple[str, Protein | None]:
    """Fold ``sequence`` with one engine, returning ``(name, protein|None)``.

    Keeps the engine name attached to the result so provenance survives even
    when ``on_error="skip"`` drops a failing engine (which would otherwise
    misalign a plain result list against ``engines``).
    """
    name = getattr(engine, "name", type(engine).__name__)
    try:
        protein = engine.predict(sequence, **predict_kwargs)
    except Exception:
        if on_error == "raise":
            raise
        _logger.warning("cross_engine_fold: engine %s failed; skipping", name, exc_info=True)
        return (name, None)
    return (name, protein)


def _resolve_backend(engines: list[FoldingEngine], backend: Backend | None) -> Backend:
    """An explicit backend wins; else ``"process"`` only if every engine opts
    in via its ``parallelism`` hint, otherwise ``"serial"`` (GPU-safe)."""
    if backend is not None:
        return backend
    hints = {getattr(e, "parallelism", "serial") for e in engines}
    return "process" if hints == {"process"} else "serial"


def _dedupe_names(names: list[str]) -> list[str]:
    """Make labels unique: the first ``ESMFold`` stays, later ones become
    ``ESMFold#2``, ``ESMFold#3`` — so every column of the matrices is
    addressable."""
    counts: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        counts[name] = counts.get(name, 0) + 1
        out.append(name if counts[name] == 1 else f"{name}#{counts[name]}")
    return out


def _ca_coords(protein: Protein) -> NDArray[np.float64]:
    """``(n_residues, 3)`` CA coordinates, one per protein residue.

    Mirrors the selection used by the TM-score / RMSD metrics so the counts
    validated here match what those functions will see.
    """
    arr = protein.atom_array
    out: list[NDArray[np.float64]] = []
    for sl in arr.iter_residue_slices():
        if str(arr.entity_type[sl.start]) != "protein":
            continue
        names = arr.atom_name[sl]
        ca_idx = np.where(names == "CA")[0]
        if not ca_idx.size:
            continue
        out.append(arr.coords[sl][ca_idx[0]])
    return np.asarray(out, dtype=np.float64)


def _pairwise_matrices(
    members: list[Protein],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Full pairwise TM-score and CA-RMSD matrices.

    Each entry uses the metrics' own optimal superposition, so the matrices
    don't depend on the ensemble's chosen reference frame. Members fold the
    same sequence (equal CA counts), so TM normalization length is identical
    both ways and the matrices are symmetrized from the upper triangle.
    """
    n = len(members)
    tm = np.eye(n, dtype=np.float64)
    rm = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            tm[i, j] = tm[j, i] = tm_score(members[i], members[j])
            rm[i, j] = rm[j, i] = rmsd(members[i], members[j], subset="ca")
    return tm, rm


def _resolve_reference(
    reference: str,
    names: list[str],
    members: list[Protein],
    tm_matrix: NDArray[np.float64],
) -> int:
    """Resolve the ``reference`` selector to an index into ``members``."""
    if reference == "medoid":
        return int(tm_matrix.sum(axis=1).argmax())
    if reference == "first":
        return 0
    if reference == "most_confident":
        confs = np.array(
            [float(p.metadata.get("mean_confidence", np.nan)) for p in members],
            dtype=np.float64,
        )
        if np.isnan(confs).all():
            raise ValueError(
                "reference='most_confident' but no member has a "
                "metadata['mean_confidence'] score to rank by."
            )
        return int(np.nanargmax(confs))
    if reference in names:
        return names.index(reference)
    raise ValueError(
        f"unknown reference {reference!r}; expected 'medoid', 'first', "
        f"'most_confident', or an engine name in {names}."
    )


def _superpose_all(
    members: list[Protein],
    ca_coords: list[NDArray[np.float64]],
    ref_idx: int,
) -> tuple[list[Protein], NDArray[np.float64]]:
    """Superpose every member onto the reference on CA atoms.

    Returns deep-copied members with *all* atoms moved by the CA-derived
    transform (so the full structures overlay), plus the ``(N, L, 3)`` stack
    of aligned CA coordinates for the RMSF computation.
    """
    ref_ca = ca_coords[ref_idx]
    n = len(members)
    length = ref_ca.shape[0]
    aligned_ca = np.empty((n, length, 3), dtype=np.float64)
    out: list[Protein] = []
    for i, member in enumerate(members):
        sp = superpose(ca_coords[i], ref_ca)
        moved = deepcopy(member)
        full = moved.atom_array.coords.astype(np.float64)
        moved.atom_array.coords[:] = ((sp.rotation @ full.T).T + sp.translation).astype(np.float32)
        aligned_ca[i] = (sp.rotation @ ca_coords[i].T).T + sp.translation
        out.append(moved)
    return out, aligned_ca


def _per_residue_rmsf(aligned_ca: NDArray[np.float64]) -> NDArray[np.float32]:
    """Root-mean-square fluctuation of each CA about its cross-engine mean.

    ``aligned_ca`` is ``(N_engines, L, 3)``; returns ``(L,)``.
    """
    mean_pos = aligned_ca.mean(axis=0)
    dev = aligned_ca - mean_pos
    rmsf: NDArray[np.float32] = np.sqrt((dev * dev).sum(axis=2).mean(axis=0)).astype(np.float32)
    return rmsf
