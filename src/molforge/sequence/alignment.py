"""Pairwise sequence alignment — Needleman-Wunsch (global) and Smith-Waterman (local).

Both algorithms are implemented in pure NumPy. They're not the fastest
(BioPython's C-backed aligner is faster on long sequences) but they're
deterministic, dependency-free, and fast enough for everyday protein
work (~10-100ms on 500-residue pairs).

For very long sequences (>5kb) or whole-genome work, route through a
specialized tool. For the antibody/nanobody/protein-design workflows
molforge targets, this is the right level of capability.

References:
    - Needleman & Wunsch 1970, J. Mol. Biol. 48: 443-453
    - Smith & Waterman 1981, J. Mol. Biol. 147: 195-197
    - BLOSUM62: Henikoff & Henikoff 1992, PNAS 89: 10915-10919
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from molforge.sequence.matrices import get_matrix


@dataclass(frozen=True)
class Alignment:
    """Result of a pairwise sequence alignment.

    Attributes:
        aligned_a: Top sequence with gaps inserted (``-`` for gap).
        aligned_b: Bottom sequence with gaps inserted.
        score: Total alignment score.
        identity: Fraction of aligned positions where residues match
            (excluding gap-vs-gap and gap-vs-residue positions).
        coverage_a: Fraction of sequence A covered by the alignment.
        coverage_b: Fraction of sequence B covered by the alignment.
        start_a: Start index in original sequence A (inclusive).
        end_a: End index in original sequence A (exclusive).
        start_b: Start index in original sequence B (inclusive).
        end_b: End index in original sequence B (exclusive).
    """

    aligned_a: str
    aligned_b: str
    score: float
    identity: float
    coverage_a: float
    coverage_b: float
    start_a: int
    end_a: int
    start_b: int
    end_b: int

    @property
    def length(self) -> int:
        """Alignment length including gaps."""
        return len(self.aligned_a)

    def format(self, width: int = 60) -> str:
        """Format the alignment as a human-readable block.

        Args:
            width: Characters per row.

        Returns:
            Multi-line string showing aligned_a / matches / aligned_b
            in stacked blocks of ``width`` columns.
        """
        lines: list[str] = []
        matches = "".join(
            "|" if a == b and a != "-" else (":" if a != "-" and b != "-" else " ")
            for a, b in zip(self.aligned_a, self.aligned_b, strict=True)
        )
        for i in range(0, len(self.aligned_a), width):
            lines.append(self.aligned_a[i : i + width])
            lines.append(matches[i : i + width])
            lines.append(self.aligned_b[i : i + width])
            lines.append("")
        return "\n".join(lines).rstrip()


def _validate_sequences(a: str, b: str) -> tuple[str, str]:
    """Clean and validate sequences before alignment."""
    a_clean = "".join(c for c in a.upper() if not c.isspace())
    b_clean = "".join(c for c in b.upper() if not c.isspace())
    if not a_clean:
        raise ValueError("sequence a is empty after stripping whitespace")
    if not b_clean:
        raise ValueError("sequence b is empty after stripping whitespace")
    return a_clean, b_clean


def _score(
    a: str,
    b: str,
    matrix: NDArray[np.int_] | None,
    matrix_index: dict[str, int] | None,
    match: int,
    mismatch: int,
) -> NDArray[np.int_]:
    """Build the (len(a)+1, len(b)+1) substitution-score lookup for two strings.

    Returns a 2D array `s[i, j]` = score of pairing a[i-1] with b[j-1].
    """
    n, m = len(a), len(b)
    s = np.zeros((n + 1, m + 1), dtype=np.int32)
    if matrix is not None and matrix_index is not None:
        # Substitution matrix path. Map every residue to its row/col index.
        a_idx = np.array([matrix_index.get(c, matrix_index.get("X", 0)) for c in a], dtype=np.int32)
        b_idx = np.array([matrix_index.get(c, matrix_index.get("X", 0)) for c in b], dtype=np.int32)
        s[1:, 1:] = matrix[a_idx[:, None], b_idx[None, :]]
    else:
        # Constant match/mismatch path.
        a_arr = np.array(list(a))
        b_arr = np.array(list(b))
        s[1:, 1:] = np.where(a_arr[:, None] == b_arr[None, :], match, mismatch)
    return s


def needleman_wunsch(
    a: str,
    b: str,
    *,
    matrix: str | None = "BLOSUM62",
    match: int = 2,
    mismatch: int = -1,
    gap_open: int = -10,
    gap_extend: int = -1,
) -> Alignment:
    """Global pairwise alignment (Needleman-Wunsch with affine gaps).

    Args:
        a, b: Sequences to align.
        matrix: Substitution matrix name (e.g. ``"BLOSUM62"``, ``"PAM250"``)
            or ``None`` to use ``match`` / ``mismatch`` instead.
        match, mismatch: Per-position scores when ``matrix=None``.
        gap_open: Penalty for opening a new gap (added at gap start).
        gap_extend: Penalty for each additional gap position.

    Returns:
        An :class:`Alignment` covering the full length of both sequences.

    Notes:
        Implements affine gap penalties via the standard three-matrix
        formulation (M for match/mismatch, X for gap in A, Y for gap in B).
        Penalties are *added* to the score, so they should be negative.
    """
    a, b = _validate_sequences(a, b)
    n, m = len(a), len(b)
    sub_matrix, idx = (None, None)
    if matrix is not None:
        sub_matrix, idx = get_matrix(matrix)
    s = _score(a, b, sub_matrix, idx, match, mismatch)

    NEG = -(10**9)
    M = np.full((n + 1, m + 1), NEG, dtype=np.int64)
    X = np.full((n + 1, m + 1), NEG, dtype=np.int64)  # gap in b (consume a)
    Y = np.full((n + 1, m + 1), NEG, dtype=np.int64)  # gap in a (consume b)
    M[0, 0] = 0
    # First row / column: only gaps possible.
    for i in range(1, n + 1):
        X[i, 0] = gap_open + gap_extend * (i - 1)
    for j in range(1, m + 1):
        Y[0, j] = gap_open + gap_extend * (j - 1)

    # Traceback pointers — 0=M-diag, 1=X-up (gap in b), 2=Y-left (gap in a).
    tb_M = np.zeros((n + 1, m + 1), dtype=np.int8)
    tb_X = np.zeros((n + 1, m + 1), dtype=np.int8)
    tb_Y = np.zeros((n + 1, m + 1), dtype=np.int8)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # M: came from a match/mismatch.
            options = (M[i - 1, j - 1], X[i - 1, j - 1], Y[i - 1, j - 1])
            best = max(options)
            M[i, j] = best + s[i, j]
            tb_M[i, j] = int(np.argmax(options))
            # X: gap in b (advance in a only).
            open_x = M[i - 1, j] + gap_open
            extend_x = X[i - 1, j] + gap_extend
            if open_x >= extend_x:
                X[i, j] = open_x
                tb_X[i, j] = 0  # came from M
            else:
                X[i, j] = extend_x
                tb_X[i, j] = 1  # extended X
            # Y: gap in a (advance in b only).
            open_y = M[i, j - 1] + gap_open
            extend_y = Y[i, j - 1] + gap_extend
            if open_y >= extend_y:
                Y[i, j] = open_y
                tb_Y[i, j] = 0
            else:
                Y[i, j] = extend_y
                tb_Y[i, j] = 2

    # Pick the best end state at (n, m).
    end_options = [M[n, m], X[n, m], Y[n, m]]
    state = int(np.argmax(end_options))
    score = int(end_options[state])

    out_a: list[str] = []
    out_b: list[str] = []
    i, j = n, m
    while i > 0 or j > 0:
        if state == 0:  # M
            out_a.append(a[i - 1])
            out_b.append(b[j - 1])
            prev = int(tb_M[i, j])
            i -= 1
            j -= 1
            state = prev
        elif state == 1:  # X — gap in b
            out_a.append(a[i - 1])
            out_b.append("-")
            prev = int(tb_X[i, j])
            i -= 1
            state = prev
        else:  # Y — gap in a
            out_a.append("-")
            out_b.append(b[j - 1])
            prev = int(tb_Y[i, j])
            j -= 1
            state = prev

    aligned_a = "".join(reversed(out_a))
    aligned_b = "".join(reversed(out_b))
    matches = sum(1 for x, y in zip(aligned_a, aligned_b, strict=True) if x == y and x != "-")
    aligned_positions = sum(
        1 for x, y in zip(aligned_a, aligned_b, strict=True) if x != "-" and y != "-"
    )
    identity = matches / aligned_positions if aligned_positions else 0.0
    return Alignment(
        aligned_a=aligned_a,
        aligned_b=aligned_b,
        score=float(score),
        identity=identity,
        coverage_a=1.0,
        coverage_b=1.0,
        start_a=0,
        end_a=n,
        start_b=0,
        end_b=m,
    )


def smith_waterman(
    a: str,
    b: str,
    *,
    matrix: str | None = "BLOSUM62",
    match: int = 2,
    mismatch: int = -1,
    gap_open: int = -10,
    gap_extend: int = -1,
) -> Alignment:
    """Local pairwise alignment (Smith-Waterman with affine gaps).

    Finds the highest-scoring local subsequence pair.

    Args:
        a, b: Sequences to align.
        matrix, match, mismatch, gap_open, gap_extend: see :func:`needleman_wunsch`.

    Returns:
        An :class:`Alignment` covering only the best-scoring local region;
        ``start_*`` / ``end_*`` give the bounds in the original sequences.
    """
    a, b = _validate_sequences(a, b)
    n, m = len(a), len(b)
    sub_matrix, idx = (None, None)
    if matrix is not None:
        sub_matrix, idx = get_matrix(matrix)
    s = _score(a, b, sub_matrix, idx, match, mismatch)

    M = np.zeros((n + 1, m + 1), dtype=np.int64)
    X = np.zeros((n + 1, m + 1), dtype=np.int64)
    Y = np.zeros((n + 1, m + 1), dtype=np.int64)
    tb = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0 none, 1 diag, 2 up, 3 left

    best = 0
    best_ij = (0, 0)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = max(M[i - 1, j - 1], X[i - 1, j - 1], Y[i - 1, j - 1]) + s[i, j]
            X[i, j] = max(M[i - 1, j] + gap_open, X[i - 1, j] + gap_extend, 0)
            Y[i, j] = max(M[i, j - 1] + gap_open, Y[i, j - 1] + gap_extend, 0)
            candidates = (0, diag, X[i, j], Y[i, j])
            choice = int(np.argmax(candidates))
            M[i, j] = candidates[choice]
            tb[i, j] = choice
            if M[i, j] > best:
                best = int(M[i, j])
                best_ij = (i, j)

    out_a: list[str] = []
    out_b: list[str] = []
    i, j = best_ij
    end_a, end_b = i, j
    while i > 0 and j > 0 and M[i, j] > 0:
        move = int(tb[i, j])
        if move == 1:  # diag
            out_a.append(a[i - 1])
            out_b.append(b[j - 1])
            i -= 1
            j -= 1
        elif move == 2:  # up
            out_a.append(a[i - 1])
            out_b.append("-")
            i -= 1
        elif move == 3:  # left
            out_a.append("-")
            out_b.append(b[j - 1])
            j -= 1
        else:
            break
    start_a, start_b = i, j
    aligned_a = "".join(reversed(out_a))
    aligned_b = "".join(reversed(out_b))
    matches = sum(1 for x, y in zip(aligned_a, aligned_b, strict=True) if x == y and x != "-")
    aligned_positions = sum(
        1 for x, y in zip(aligned_a, aligned_b, strict=True) if x != "-" and y != "-"
    )
    identity = matches / aligned_positions if aligned_positions else 0.0
    return Alignment(
        aligned_a=aligned_a,
        aligned_b=aligned_b,
        score=float(best),
        identity=identity,
        coverage_a=(end_a - start_a) / n if n else 0.0,
        coverage_b=(end_b - start_b) / m if m else 0.0,
        start_a=start_a,
        end_a=end_a,
        start_b=start_b,
        end_b=end_b,
    )


def align(
    a: str,
    b: str,
    *,
    mode: str = "global",
    matrix: str | None = "BLOSUM62",
    match: int = 2,
    mismatch: int = -1,
    gap_open: int = -10,
    gap_extend: int = -1,
) -> Alignment:
    """Pairwise alignment entry point.

    Args:
        mode: ``"global"`` (Needleman-Wunsch) or ``"local"`` (Smith-Waterman).

    See :func:`needleman_wunsch` / :func:`smith_waterman` for the rest.
    """
    if mode == "global":
        return needleman_wunsch(
            a,
            b,
            matrix=matrix,
            match=match,
            mismatch=mismatch,
            gap_open=gap_open,
            gap_extend=gap_extend,
        )
    if mode == "local":
        return smith_waterman(
            a,
            b,
            matrix=matrix,
            match=match,
            mismatch=mismatch,
            gap_open=gap_open,
            gap_extend=gap_extend,
        )
    raise ValueError(f"unknown alignment mode {mode!r}; expected 'global' or 'local'")


def identity(a: str, b: str, *, mode: str = "global") -> float:
    """Convenience: align and return only the identity score."""
    return align(a, b, mode=mode).identity
