"""Sequence-level featurizers for ML models.

What's here:
    - :func:`one_hot` — classic 20-dim one-hot encoding.
    - :func:`blosum_embed` — BLOSUM62 row as a 20-dim embedding per residue.
    - :func:`positional_encoding` — sinusoidal positional encodings
      (transformer-style) for absolute or relative position.
    - :func:`compose_features` — concatenate multiple featurizers along
      the last axis for a single combined embedding.

All featurizers return float32 NumPy arrays so they're cheap to convert
to PyTorch / JAX downstream. The convention is ``(L, D)`` where L is
sequence length and D is feature dimensionality.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Standard 20 amino acid alphabet in the order most papers use.
_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
_AA_TO_IDX = {aa: i for i, aa in enumerate(_AA_ALPHABET)}
_UNK_IDX = 20  # 21st row reserved for non-standard / unknown


def _normalize_sequence(sequence: str) -> str:
    """Strip whitespace and uppercase; preserve non-standard letters."""
    return "".join(c for c in sequence.upper() if not c.isspace())


def one_hot(sequence: str, *, include_unk: bool = True) -> NDArray[np.float32]:
    """One-hot encode a protein sequence.

    Args:
        sequence: one-letter amino-acid sequence.
        include_unk: if True (default), use a 21-dimensional encoding
            with column 20 set to 1 for any non-standard residue. If
            False, use 20 dims and silently drop the bit for non-standard
            residues (their rows will be all zero).

    Returns:
        ``(L, 21)`` or ``(L, 20)`` float32 array.
    """
    seq = _normalize_sequence(sequence)
    n = len(seq)
    dim = 21 if include_unk else 20
    out = np.zeros((n, dim), dtype=np.float32)
    for i, aa in enumerate(seq):
        idx = _AA_TO_IDX.get(aa, _UNK_IDX if include_unk else -1)
        if 0 <= idx < dim:
            out[i, idx] = 1.0
    return out


def blosum_embed(
    sequence: str,
    *,
    matrix: str = "BLOSUM62",
) -> NDArray[np.float32]:
    """Use rows of a substitution matrix as per-residue embeddings.

    This is a classic trick from the pre-deep-learning era: every
    residue gets a 20-dim embedding that captures its substitution
    profile against the standard amino acids. Surprisingly strong as
    a baseline for many downstream tasks.

    Args:
        sequence: one-letter amino-acid sequence.
        matrix: substitution matrix name (``"BLOSUM62"`` or ``"PAM250"``).

    Returns:
        ``(L, 20)`` float32 array. Non-standard residues get the
        ``"X"`` row (which exists in both BLOSUM62 and PAM250).
    """
    from molforge.sequence.matrices import get_matrix

    mat, idx_map = get_matrix(matrix)
    seq = _normalize_sequence(sequence)
    # Pull rows for the 20 standard AAs from the full matrix.
    standard_cols = [idx_map[c] for c in _AA_ALPHABET]
    rows = []
    for aa in seq:
        row_idx = idx_map.get(aa, idx_map["X"])
        rows.append(mat[row_idx, standard_cols])
    return np.asarray(rows, dtype=np.float32)


def positional_encoding(
    length: int,
    dim: int = 64,
    *,
    base: float = 10000.0,
) -> NDArray[np.float32]:
    """Sinusoidal absolute positional encoding (Vaswani et al. 2017).

    Identical formulation to the original Transformer paper:
    ``PE[pos, 2i]   = sin(pos / base^(2i/dim))``
    ``PE[pos, 2i+1] = cos(pos / base^(2i/dim))``

    Args:
        length: sequence length.
        dim: embedding dimensionality (must be even).
        base: wavelength base. The Vaswani default is 10000.

    Returns:
        ``(length, dim)`` float32 array.
    """
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}")
    pos = np.arange(length, dtype=np.float64)[:, None]
    div = np.exp(np.arange(0, dim, 2, dtype=np.float64) * (-np.log(base) / dim))
    angles = pos * div[None, :]
    out = np.zeros((length, dim), dtype=np.float32)
    out[:, 0::2] = np.sin(angles).astype(np.float32)
    out[:, 1::2] = np.cos(angles).astype(np.float32)
    return out


def compose_features(*features: NDArray[np.float32]) -> NDArray[np.float32]:
    """Concatenate multiple per-residue feature arrays along the last axis.

    All inputs must have the same leading dimension (sequence length).

    Args:
        *features: any number of ``(L, D_i)`` arrays.

    Returns:
        ``(L, sum(D_i))`` float32 array.

    Raises:
        ValueError: if shapes are incompatible.
    """
    if not features:
        raise ValueError("compose_features requires at least one input array")
    lengths = {f.shape[0] for f in features}
    if len(lengths) != 1:
        raise ValueError(f"all feature arrays must share leading dimension; got {lengths}")
    return np.concatenate(features, axis=-1).astype(np.float32)
