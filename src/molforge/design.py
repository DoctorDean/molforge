"""The protein-design loop: generate → fold → (dock) → score → iterate.

Real protein engineering is a loop. You propose sequences for a scaffold,
predict what they fold to, check whether the prediction actually matches
the scaffold you designed for, keep the winners, and design again from
them. Every piece already exists as a molforge wrapper — a sequence
designer (ProteinMPNN, ESM-IF1), a folding engine (ESMFold, AlphaFold,
Boltz), optionally a docking engine — but nothing glues them into the
loop. :class:`DesignLoop` is that glue:

    from molforge.design import DesignLoop
    from molforge.wrappers.generative import ProteinMPNN
    from molforge.wrappers.folding import ESMFold

    loop = DesignLoop(designer=ProteinMPNN(), folder=ESMFold(), n_rounds=3)
    table = loop.run(backbone)          # a Protein or a PDB path
    best = table.best                   # highest self-consistency design
    rows = table.to_records()           # flat dicts → drop into a DataFrame

The default objective is **self-consistency**: fold each designed
sequence and measure how well the prediction superposes on the backbone
it was designed for (scTM / scRMSD). This is the metric the
RFdiffusion / ProteinMPNN / AlphaFold design pipelines are graded on, and
it falls straight out of the corrected :func:`~molforge.metrics.tm_score`
and :func:`~molforge.structure.rmsd`. Other built-in objectives score by
folding confidence (``"plddt"``) or docked affinity (``"affinity"``), and
a custom callable covers everything else.

Folding can be a *single* engine or a *list* — pass a list and each
candidate is folded with :func:`~molforge.ensembles.cross_engine_fold`,
scored against the cross-engine consensus, with the per-residue
cross-engine disagreement recorded as an extra confidence signal.

Iteration is genuine: round *r+1* re-designs onto the folded structures
of the top ``select_top`` candidates from round *r*, so the scaffold
refines as the loop runs. The :class:`DesignTable` accumulates every
candidate across every round, ranked best-first.

Scope (v1): the ``designer`` produces sequences for a provided backbone.
The ``generator`` slot (round-0 backbone generation, e.g. RFdiffusion) is
part of the signature but raises :class:`NotImplementedError` — its
configuration surface (contigs, targets, symmetry) is too engine-specific
to wire generically yet.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from molforge.core import metadata_keys as mk
from molforge.ensembles import cross_engine_fold
from molforge.metrics import tm_score
from molforge.parallel import fold_many
from molforge.scoring import Scorer
from molforge.structure import rmsd

if TYPE_CHECKING:
    import os
    from collections.abc import Callable, Sequence

    from molforge.core import Protein
    from molforge.docking import DockingEngine, DockingResult
    from molforge.generative import DesignedSequence, GenerativeEngine
    from molforge.parallel import Backend
    from molforge.wrappers.folding import FoldingEngine

__all__ = ["DesignCandidate", "DesignLoop", "DesignObjective", "DesignTable"]

_logger = logging.getLogger(__name__)

#: Built-in objective presets (all "higher is better"). A custom
#: ``Callable[[DesignCandidate], float]`` may be used instead.
DesignObjective = Literal["self_consistency", "plddt", "affinity"]


@dataclass
class DesignCandidate:
    """One design as it flows through the loop, accumulating results.

    Attributes:
        sequence: The designed one-letter sequence.
        round: 0-indexed loop round that produced it.
        backbone: The scaffold this sequence was designed onto — the
            user's input backbone in round 0, a previous round's folded
            winner thereafter. The reference for self-consistency.
        structure: The folded (predicted) structure. ``None`` until the
            fold stage runs. For a list ``folder`` this is the cross-engine
            consensus.
        docking: The docking result against the receptor, if a ``docker``
            was configured; otherwise ``None``.
        metrics: Named scalar measurements — ``sc_tm``, ``sc_rmsd``,
            ``plddt``, ``mpnn_score``, ``affinity``, and (for list folders)
            ``cross_engine_tm_mean`` / ``cross_engine_rmsf_mean``.
        score: The objective value; higher is better. ``nan`` until scored.
        metadata: Free-form extras.
    """

    sequence: str
    round: int
    backbone: Protein | None = None
    structure: Protein | None = None
    docking: DockingResult | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    score: float = math.nan
    metadata: dict[str, object] = field(default_factory=dict)

    def __repr__(self) -> str:
        seq = self.sequence if len(self.sequence) <= 24 else self.sequence[:21] + "..."
        return f"DesignCandidate(round={self.round}, seq={seq!r}, score={self.score:.3f})"


@dataclass
class DesignTable:
    """The ranked output of a :meth:`DesignLoop.run`.

    Candidates from every round, sorted best-first by objective score
    (``nan`` scores sort last).

    Attributes:
        candidates: All designs, best-first.
        rounds: Number of rounds that ran.
        objective: The objective name (or ``"custom"`` for a callable).
    """

    candidates: list[DesignCandidate]
    rounds: int
    objective: str

    def __len__(self) -> int:
        return len(self.candidates)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.candidates)

    @property
    def best(self) -> DesignCandidate:
        """The top-scoring candidate. Raises ``IndexError`` if empty."""
        return self.candidates[0]

    def top_n(self, n: int) -> list[DesignCandidate]:
        """The ``n`` best-scoring candidates."""
        return self.candidates[:n]

    def to_records(self) -> list[dict[str, object]]:
        """Flatten to a list of row dicts — ``pandas.DataFrame(...)``-ready.

        Each row has ``round``, ``score``, ``sequence``, and one column per
        metric key present across the candidates (missing values filled
        with ``nan``).
        """
        metric_keys: list[str] = []
        for c in self.candidates:
            for k in c.metrics:
                if k not in metric_keys:
                    metric_keys.append(k)
        rows: list[dict[str, object]] = []
        for c in self.candidates:
            row: dict[str, object] = {
                "round": c.round,
                "score": c.score,
                "sequence": c.sequence,
            }
            for k in metric_keys:
                row[k] = c.metrics.get(k, math.nan)
            rows.append(row)
        return rows


class DesignLoop:
    """Orchestrate generate → fold → (dock) → score → iterate.

    Args:
        designer: A sequence-design engine — any object with
            ``generate(backbone, **kwargs) -> list[DesignedSequence]``
            (ProteinMPNN, ESM-IF1).
        folder: A folding engine, or a list of them. A single engine folds
            each sequence with ``predict``; a list folds via
            :func:`~molforge.ensembles.cross_engine_fold` and scores against
            the consensus.
        docker: Optional docking engine. When set, each folded structure is
            docked against ``receptor`` (supplied to :meth:`run`) and the
            best pose's score is recorded as ``affinity``.
        generator: Reserved for round-0 backbone generation (RFdiffusion).
            Supplying it raises :class:`NotImplementedError` in v1.
        objective: ``"self_consistency"`` (default), ``"plddt"``,
            ``"affinity"``, or a custom ``Callable[[DesignCandidate],
            float]``. Higher is better.
        n_designs: Max designed sequences to carry forward per backbone per
            round (the designer may propose more; the top ``n_designs`` by
            the designer's own score are kept).
        n_rounds: Number of design rounds.
        select_top: Winners carried as next-round seeds (their folded
            structures become the scaffolds redesigned onto).
        workers: Parallel-fold worker count (see :func:`molforge.parallel`).
        backend: Parallel-fold backend override; defaults to the folding
            engine's ``parallelism`` hint.

    Raises:
        NotImplementedError: If ``generator`` is supplied.
        ValueError: If ``objective="affinity"`` without a ``docker``, or the
            numeric parameters are non-positive.
    """

    def __init__(
        self,
        *,
        designer: GenerativeEngine,
        folder: FoldingEngine | Sequence[FoldingEngine],
        docker: DockingEngine | None = None,
        generator: GenerativeEngine | None = None,
        objective: DesignObjective | Callable[[DesignCandidate], float] | Scorer = "self_consistency",
        n_designs: int = 8,
        n_rounds: int = 1,
        select_top: int = 4,
        workers: int | None = None,
        backend: Backend | None = None,
    ) -> None:
        if generator is not None:
            raise NotImplementedError(
                "DesignLoop v1 designs onto a provided backbone; round-0 "
                "backbone generation (the `generator` slot, e.g. RFdiffusion) "
                "is planned. Pass a backbone to run() instead."
            )
        if objective == "affinity" and docker is None:
            raise ValueError("objective='affinity' requires a docker engine.")
        for name, val in (
            ("n_designs", n_designs),
            ("n_rounds", n_rounds),
            ("select_top", select_top),
        ):
            if val < 1:
                raise ValueError(f"{name} must be >= 1, got {val}.")

        self.designer = designer
        self.folder = folder
        self.docker = docker
        self.objective = objective
        self.n_designs = n_designs
        self.n_rounds = n_rounds
        self.select_top = select_top
        self.workers = workers
        self.backend = backend

        self._folders: list[FoldingEngine] = (
            list(folder) if isinstance(folder, (list, tuple)) else [cast("FoldingEngine", folder)]
        )
        self._cross_engine = len(self._folders) > 1
        self._objective_fn = _resolve_objective(objective)

    def run(
        self,
        backbone: Protein | str | os.PathLike[str],
        *,
        receptor: Protein | None = None,
        designer_kwargs: dict[str, Any] | None = None,
        fold_kwargs: dict[str, Any] | None = None,
        dock_kwargs: dict[str, Any] | None = None,
    ) -> DesignTable:
        """Run the loop and return a ranked :class:`DesignTable`.

        Args:
            backbone: The scaffold to design onto — a :class:`Protein` or a
                path to a PDB file. A path is loaded once for scoring.
            receptor: The docking receptor; required if a ``docker`` was
                configured.
            designer_kwargs: Extra kwargs forwarded to ``designer.generate``.
            fold_kwargs: Extra kwargs forwarded to the folding call(s).
            dock_kwargs: Extra kwargs forwarded to ``docker.dock``.

        Returns:
            A :class:`DesignTable` of every candidate across all rounds,
            best-first.

        Raises:
            ValueError: If a ``docker`` is configured but ``receptor`` is
                ``None``, or no candidate survives folding.
        """
        if self.docker is not None and receptor is None:
            raise ValueError("a docker is configured; run() needs a receptor to dock against.")

        designer_kwargs = designer_kwargs or {}
        fold_kwargs = fold_kwargs or {}
        dock_kwargs = dock_kwargs or {}

        seeds: list[Protein] = [_as_protein(backbone)]
        all_candidates: list[DesignCandidate] = []

        for rnd in range(self.n_rounds):
            candidates = self._design_round(rnd, seeds, designer_kwargs)
            if not candidates:
                _logger.warning("DesignLoop round %d: designer proposed nothing; stopping.", rnd)
                break
            self._fold_round(candidates, fold_kwargs)
            if self.docker is not None:
                assert receptor is not None  # guaranteed by the run() guard above
                self._dock_round(candidates, receptor, dock_kwargs)
            for cand in candidates:
                cand.score = self._score(cand)
            all_candidates.extend(candidates)

            ranked = _rank(candidates)
            best = ranked[0].score if ranked else math.nan
            _logger.info(
                "DesignLoop round %d: %d designs, best score %.3f", rnd, len(candidates), best
            )
            # Refine: the next round designs onto the winners' folded structures.
            seeds = [c.structure for c in ranked[: self.select_top] if c.structure is not None]
            if not seeds:
                break

        return DesignTable(
            candidates=_rank(all_candidates),
            rounds=self.n_rounds,
            objective=_objective_name(self.objective),
        )

    # ---------- stages ----------

    def _design_round(
        self, rnd: int, seeds: list[Protein], designer_kwargs: dict[str, Any]
    ) -> list[DesignCandidate]:
        """Design sequences onto each seed backbone."""
        out: list[DesignCandidate] = []
        for seed in seeds:
            designs = cast(
                "list[DesignedSequence]", self.designer.generate(seed, **designer_kwargs)
            )
            for design in designs[: self.n_designs]:
                out.append(
                    DesignCandidate(
                        sequence=design.sequence,
                        round=rnd,
                        backbone=seed,
                        metrics={"mpnn_score": float(design.score)},
                    )
                )
        return out

    def _fold_round(self, candidates: list[DesignCandidate], fold_kwargs: dict[str, Any]) -> None:
        """Fold every candidate; attach the structure and confidence metrics."""
        if self._cross_engine:
            for cand in candidates:
                ens = cross_engine_fold(
                    cand.sequence,
                    self._folders,
                    workers=self.workers,
                    backend=self.backend,
                    **fold_kwargs,
                )
                cand.structure = ens.consensus()
                spread = ens.spread()
                cand.metrics["cross_engine_tm_mean"] = spread["tm_mean"]
                cand.metrics["cross_engine_rmsf_mean"] = float(ens.disagreement().mean())
                cand.metrics["plddt"] = _mean_confidence(cand.structure)
            return
        structures = fold_many(
            self._folders[0],
            [c.sequence for c in candidates],
            workers=self.workers,
            backend=self.backend,
            **fold_kwargs,
        )
        for cand, structure in zip(candidates, structures, strict=True):
            cand.structure = structure
            cand.metrics["plddt"] = _mean_confidence(structure)

    def _dock_round(
        self,
        candidates: list[DesignCandidate],
        receptor: Protein,
        dock_kwargs: dict[str, Any],
    ) -> None:
        """Dock each folded structure against the receptor; record affinity."""
        assert self.docker is not None  # guarded in run()
        for cand in candidates:
            if cand.structure is None:
                continue
            result = self.docker.dock(receptor, cand.structure, **dock_kwargs)
            cand.docking = result
            if result.poses:
                cand.metrics["affinity"] = float(result.best.score)

    def _score(self, cand: DesignCandidate) -> float:
        """Compute self-consistency metrics, then evaluate the objective."""
        if cand.structure is not None and cand.backbone is not None:
            sc_tm, sc_rmsd = _self_consistency(cand.structure, cand.backbone)
            if sc_tm is not None:
                cand.metrics["sc_tm"] = sc_tm
                cand.metrics["sc_rmsd"] = sc_rmsd  # type: ignore[assignment]
        return float(self._objective_fn(cand))


# ---------- internals ----------


def _as_protein(backbone: Protein | str | os.PathLike[str]) -> Protein:
    """Return a :class:`Protein`, loading from a PDB path if needed."""
    from molforge.core import Protein  # local import avoids a heavy import at module load

    if isinstance(backbone, Protein):
        return backbone
    from molforge.io import read_pdb

    return read_pdb(backbone)


def _mean_confidence(structure: Protein) -> float:
    """Mean per-residue confidence (pLDDT-style), or ``nan`` if unavailable."""
    val = structure.metadata.get(mk.MEAN_CONFIDENCE)
    if val is not None:
        return float(val)
    per_res = structure.metadata.get(mk.CONFIDENCE_PER_RESIDUE)
    if per_res is not None and len(per_res) > 0:
        import numpy as np

        return float(np.mean(per_res))
    return math.nan


def _self_consistency(structure: Protein, backbone: Protein) -> tuple[float | None, float | None]:
    """scTM and scRMSD between a folded structure and its design backbone.

    Returns ``(None, None)`` when the CA counts don't correspond (which can
    happen if a folding engine returns a differently-resolved model), rather
    than raising — a single unscoreable design shouldn't sink the run.
    """
    try:
        sc_tm = tm_score(structure, backbone)
        sc_rmsd = rmsd(structure, backbone, subset="ca")
    except ValueError:
        _logger.warning("DesignLoop: could not score self-consistency; skipping.", exc_info=True)
        return None, None
    return sc_tm, sc_rmsd


def _resolve_objective(
    objective: DesignObjective | Callable[[DesignCandidate], float] | Scorer,
) -> Callable[[DesignCandidate], float]:
    """Turn a preset name, callable, or :class:`~molforge.scoring.Scorer` into
    an ``(candidate) -> float`` scorer (always higher-is-better)."""
    if isinstance(objective, Scorer):
        # A Scorer grades the folded structure; its ranking_key is
        # higher-is-better regardless of the scorer's native direction.
        return lambda c: (
            objective.score(c.structure).ranking_key if c.structure is not None else math.nan
        )
    if callable(objective):
        return objective
    if objective == "self_consistency":
        return lambda c: c.metrics.get("sc_tm", math.nan)
    if objective == "plddt":
        return lambda c: c.metrics.get("plddt", math.nan)
    if objective == "affinity":
        # Docking scores are lower-is-better (kcal/mol); negate so the
        # objective stays higher-is-better. Missing affinity ranks last.
        return lambda c: -c.metrics.get("affinity", math.inf)
    raise ValueError(
        f"unknown objective {objective!r}; expected 'self_consistency', "
        "'plddt', 'affinity', or a callable."
    )


def _objective_name(objective: DesignObjective | Callable[[DesignCandidate], float] | Scorer) -> str:
    """Human-readable objective label for the DesignTable."""
    if isinstance(objective, str):
        return objective
    if isinstance(objective, Scorer):
        return objective.name
    return "custom"


def _rank(candidates: list[DesignCandidate]) -> list[DesignCandidate]:
    """Sort candidates best-first (higher score), with ``nan`` scores last."""
    return sorted(
        candidates,
        key=lambda c: (math.isnan(c.score), -c.score if not math.isnan(c.score) else 0.0),
    )
