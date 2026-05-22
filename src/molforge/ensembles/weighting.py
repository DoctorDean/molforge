"""Boltzmann weighting and resampling for pose ensembles.

The fundamental operation: turn a vector of scores into a probability
distribution. For docking scores in kcal/mol, the natural choice is
the Boltzmann weights

    w_i = exp(-E_i / kT) / Z

where Z is the partition function (so weights sum to 1) and ``kT`` is
the thermal energy. At room temperature (298 K), ``kT ≈ 0.593`` kcal/mol.

For ML-derived scores (DiffDock confidence, EquiDock scores, etc.) the
``temperature`` parameter is just a softness control: higher T → more
uniform weights, lower T → winner-takes-all behavior. The convention
``lower_is_better=True`` (Vina, MM-GBSA, free-energy estimates) is the
default; pass ``False`` for ML confidence-style scores where larger
is better.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Sequence

    from molforge.docking import Pose


# Thermal energy at 298 K, in kcal/mol. This is what most docking
# scores are expressed in, so it's the natural default temperature.
KT_298K_KCAL_PER_MOL = 0.5924847


def boltzmann_weights(
    scores: Sequence[float] | NDArray[np.floating] | Sequence[Pose],
    *,
    temperature: float = KT_298K_KCAL_PER_MOL,
    lower_is_better: bool = True,
) -> NDArray[np.float64]:
    """Compute Boltzmann weights from a vector of scores.

    The returned weights satisfy ``sum(w) == 1.0`` and are computed in
    a numerically stable way (subtracting the min/max score before
    exponentiating).

    Args:
        scores: Either a sequence of scalar scores, or a sequence of
            :class:`molforge.docking.Pose` objects (in which case the
            ``score`` attribute is used). NumPy arrays work directly.
        temperature: The thermal energy ``kT`` in the same units as
            ``scores``. Defaults to ``kT`` at 298 K in kcal/mol
            (``0.593``), appropriate for Vina-style docking scores.
            Larger values → softer (more uniform) weights; smaller
            values → winner-takes-all. Must be strictly positive.
        lower_is_better: If ``True`` (default), scores are treated as
            energies — lower (more negative) means better, gets higher
            weight. If ``False``, larger means better (suits ML
            confidence scores). The flag flips the sign internally.

    Returns:
        A ``(n,)`` float64 array of weights summing to 1.

    Raises:
        ValueError: If ``temperature`` is non-positive, ``scores`` is
            empty, or any score is non-finite.

    Example:
        >>> from molforge.ensembles import boltzmann_weights
        >>> # Three docking scores in kcal/mol (Vina convention).
        >>> weights = boltzmann_weights([-9.5, -8.2, -7.1])
        >>> weights.sum()
        1.0
        >>> weights[0] > weights[2]  # best score gets highest weight
        True
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    arr = _coerce_to_score_array(scores)

    if arr.size == 0:
        raise ValueError("scores is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("scores contains non-finite values (nan or inf)")

    # For "lower is better" (energy convention), w_i = exp(-E_i / kT).
    # For "higher is better" (confidence convention), w_i = exp(+S_i / kT).
    # Equivalent formulation: pre-multiply by ±1 to canonicalize, then
    # always do exp(-x / kT). The numerical-stability trick (subtracting
    # the min) means the largest weight always exponentiates to 1.0.
    sign = 1.0 if lower_is_better else -1.0
    canon = sign * arr  # now "lower is better" in canon-space
    canon = canon - canon.min()  # most-favorable point → 0 → exp(0) = 1
    unnormalized = np.exp(-canon / temperature)
    return unnormalized / unnormalized.sum()


def resample(
    poses: Sequence[Pose],
    n_samples: int,
    *,
    weights: NDArray[np.floating] | None = None,
    rng: np.random.Generator | None = None,
    replace: bool = True,
) -> list[Pose]:
    """Draw ``n_samples`` poses with replacement, optionally weighted.

    Useful for downstream uncertainty estimation (bootstrap any pose-
    derived metric over the resampled population) or for converting a
    Boltzmann-weighted ensemble into an unweighted population suitable
    for tools that don't accept weights.

    Args:
        poses: The source ensemble. Order is preserved across calls
            with the same ``rng`` seed.
        n_samples: How many draws to return. Must be ≥ 1.
        weights: A ``(len(poses),)`` array of weights summing to 1
            (e.g. from :func:`boltzmann_weights`). If ``None``,
            uniform weights are used.
        rng: A NumPy ``Generator``. If ``None``, a fresh one is
            created with default seeding (non-reproducible). Pass an
            explicit ``np.random.default_rng(seed)`` for reproducibility.
        replace: Whether to sample with replacement. Default ``True``
            (which is what Boltzmann resampling means). Setting
            ``False`` requires ``n_samples <= len(poses)``.

    Returns:
        A list of ``n_samples`` :class:`Pose` objects drawn from
        ``poses`` according to ``weights``. Poses may be repeated when
        sampling with replacement; objects are not copied (each
        returned reference points at the original pose).

    Raises:
        ValueError: If ``poses`` is empty, ``n_samples < 1``, or
            ``weights`` has the wrong length / doesn't sum to ~1.
    """
    if not poses:
        raise ValueError("poses is empty")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    n = len(poses)
    if weights is None:
        probs = np.full(n, 1.0 / n)
    else:
        probs = np.asarray(weights, dtype=np.float64)
        if probs.shape != (n,):
            raise ValueError(
                f"weights has shape {probs.shape}, expected ({n},)"
            )
        total = probs.sum()
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(
                f"weights must sum to 1.0, got {total:.6f}. Did you forget "
                "to call boltzmann_weights() on the raw scores?"
            )

    if not replace and n_samples > n:
        raise ValueError(
            f"n_samples={n_samples} > n_poses={n} but replace=False"
        )

    rng = rng if rng is not None else np.random.default_rng()
    indices = rng.choice(n, size=n_samples, replace=replace, p=probs)
    return [poses[i] for i in indices]


# ---------- internals ----------


def _coerce_to_score_array(
    scores: Sequence[float] | NDArray[np.floating] | Sequence[Pose],
) -> NDArray[np.float64]:
    """Accept either a numeric sequence or a sequence of Pose objects.

    Returns a 1-D float64 array of scores.
    """
    if isinstance(scores, np.ndarray):
        return np.asarray(scores, dtype=np.float64).ravel()

    # Sequence-of-something. Peek at the first element to dispatch.
    seq = list(scores)
    if not seq:
        return np.array([], dtype=np.float64)

    first = seq[0]
    if hasattr(first, "score"):
        # Looks like a Pose (or anything with a .score attribute).
        return np.array([p.score for p in seq], dtype=np.float64)
    return np.asarray(seq, dtype=np.float64).ravel()
