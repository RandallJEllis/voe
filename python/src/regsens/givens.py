"""Givens rotation primitives for O(mp) QR column addition and removal."""

import numpy as np


def givens_params(a: float, b: float) -> tuple[float, float, float]:
    """
    Compute (c, s, r) such that [[c, s], [-s, c]] @ [a, b] = [r, 0].

    Uses np.hypot for overflow-safe norm. r >= 0 always.
    """
    r = np.hypot(a, b)
    if r < 1e-300:
        return 1.0, 0.0, 0.0
    return a / r, b / r, r


def apply_givens_rows(M: np.ndarray, j: int, c: float, s: float, col_start: int = 0) -> None:
    """
    In-place left-multiply rows j and j+1 of M by [[c, s], [-s, c]] from col_start onward.

    After this call, M[j+1, j] == 0 (when used to restore upper-triangular form).
    """
    row_j = M[j, col_start:].copy()
    row_j1 = M[j + 1, col_start:].copy()
    M[j, col_start:] = c * row_j + s * row_j1
    M[j + 1, col_start:] = -s * row_j + c * row_j1


def apply_givens_cols(Q: np.ndarray, j: int, c: float, s: float) -> None:
    """
    In-place right-multiply columns j and j+1 of Q by G^T = [[c, -s], [s, c]].

    Mirrors apply_givens_rows so that Q @ R is preserved after rotating rows of R.
    """
    col_j = Q[:, j].copy()
    col_j1 = Q[:, j + 1].copy()
    Q[:, j] = c * col_j + s * col_j1
    Q[:, j + 1] = -s * col_j + c * col_j1
