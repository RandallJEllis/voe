"""
Core QR factorization state and O(mp) update operations.

Column addition  — Gram-Schmidt extension:      O(mp)
Column removal   — Givens rotation chain:        O(mp)
New outcome solve — Q'y + back-substitution:    O(mp + p²)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular

from ._exceptions import RankDeficiencyWarning
from .givens import givens_params, apply_givens_rows, apply_givens_cols


@dataclass
class QRState:
    """
    Mutable container for a thin QR factorization X_sub = Q R.

    Invariants:
      Q  (m × p)  has orthonormal columns: Q.T @ Q == I_p
      R  (p × p)  is upper triangular with non-negative diagonal
      X[:, col_indices] == Q @ R  (up to floating point)
      len(col_indices) == p == Q.shape[1] == R.shape[0]
    """

    Q: np.ndarray           # (m, p)
    R: np.ndarray           # (p, p)
    col_indices: list[int]  # active original column indices, in insertion order
    m: int
    p: int
    tol: float = 1e-10

    def copy(self) -> QRState:
        return QRState(
            Q=self.Q.copy(),
            R=self.R.copy(),
            col_indices=list(self.col_indices),
            m=self.m,
            p=self.p,
            tol=self.tol,
        )


def qr_state_from_cols(
    X: np.ndarray,
    col_indices: list[int],
    tol: float = 1e-10,
) -> QRState:
    """
    Build a QRState from design matrix X and a list of 0-based column indices.

    Empty col_indices → p=0 empty state.
    Sign convention: R[j, j] >= 0 for all j.
    """
    m = X.shape[0]
    p = len(col_indices)

    if p == 0:
        return QRState(
            Q=np.zeros((m, 0), dtype=float),
            R=np.zeros((0, 0), dtype=float),
            col_indices=[],
            m=m, p=0, tol=tol,
        )

    X_sub = X[:, col_indices]
    Q, R = np.linalg.qr(X_sub, mode="reduced")

    # Enforce non-negative diagonal
    for j in range(p):
        if R[j, j] < 0:
            Q[:, j] = -Q[:, j]
            R[j, :] = -R[j, :]

    return QRState(Q=Q, R=R, col_indices=list(col_indices), m=m, p=p, tol=tol)


def qr_add_column(
    state: QRState,
    z: np.ndarray,
    col_idx: int,
    inplace: bool = True,
) -> QRState:
    """
    Append column z (0-based col_idx in original X) to the right of the factorisation.

    Gram-Schmidt extension:
        p_vec    = Q^T z              — projection onto column space
        residual = z - Q p_vec        — orthogonal complement
        q_new    = residual / ||residual||

    If ||residual|| < tol the column is nearly rank-deficient: a warning is issued
    and norm_r is clamped to tol so the factorisation remains numerically defined.
    """
    if not inplace:
        state = state.copy()

    p = state.p
    Q = state.Q
    R = state.R

    p_vec = Q.T @ z
    residual = z - Q @ p_vec
    norm_r = float(np.linalg.norm(residual))

    if norm_r < state.tol:
        warnings.warn(
            f"Column {col_idx} is nearly in the span of existing columns "
            f"(residual norm {norm_r:.2e} < tol {state.tol:.2e}). "
            "Clamping to tol.",
            RankDeficiencyWarning,
            stacklevel=2,
        )
        norm_r = state.tol

    q_new = residual / norm_r

    state.Q = np.concatenate([Q, q_new[:, np.newaxis]], axis=1)

    new_R = np.zeros((p + 1, p + 1), dtype=float)
    new_R[:p, :p] = R
    new_R[:p, p] = p_vec
    new_R[p, p] = norm_r
    state.R = new_R

    state.col_indices.append(col_idx)
    state.p += 1
    return state


def qr_remove_column(
    state: QRState,
    pos: int,
    inplace: bool = True,
) -> QRState:
    """
    Remove the column at 0-based position pos from the factorisation.

    Deleting column pos from R creates a (p × p-1) upper Hessenberg matrix.
    A chain of Givens rotations (rows j, j+1 for j = pos..p-2) restores upper
    triangularity. The same rotations are applied to Q (right-multiply by G^T)
    to maintain Q R = X. The final stub column/row is discarded.
    """
    if not inplace:
        state = state.copy()

    p = state.p
    R_work = np.delete(state.R, pos, axis=1)   # (p, p-1)
    Q_work = state.Q                            # mutated in-place

    for j in range(pos, p - 1):
        a = R_work[j, j]
        b = R_work[j + 1, j]
        if abs(b) < 1e-15:
            continue
        c, s, _ = givens_params(a, b)
        apply_givens_rows(R_work, j, c, s, col_start=j)
        apply_givens_cols(Q_work, j, c, s)

    state.Q = Q_work[:, :-1]
    state.R = R_work[:-1, :]
    del state.col_indices[pos]
    state.p -= 1
    return state


def qr_new_outcome(state: QRState, y: np.ndarray) -> dict:
    """
    Solve OLS for new outcome y using existing Q, R.  Cost: O(mp + p²).

    Returns: beta, fitted, residuals, rss, Qty.
    """
    Qty = state.Q.T @ y
    beta = solve_triangular(state.R, Qty)
    fitted = state.Q @ Qty
    residuals = y - fitted
    rss = float(np.dot(residuals, residuals))
    return {"beta": beta, "fitted": fitted, "residuals": residuals,
            "rss": rss, "Qty": Qty}


def qr_batch_outcomes(state: QRState, Y: np.ndarray) -> dict:
    """
    Solve OLS for all q outcome columns in Y simultaneously.

    Y: (m, q). Returns: beta (p,q), fitted (m,q), residuals (m,q), rss (q,), QtY (p,q).
    """
    QtY = state.Q.T @ Y
    Beta = solve_triangular(state.R, QtY)
    Fitted = state.Q @ QtY
    Residuals = Y - Fitted
    RSS = np.einsum("ij,ij->j", Residuals, Residuals)
    return {"beta": Beta, "fitted": Fitted, "residuals": Residuals,
            "rss": RSS, "QtY": QtY}


def compute_se(R: np.ndarray, sigma) -> np.ndarray:
    """
    Compute OLS standard errors from R factor and residual SD sigma.

    SE_j = sigma * sqrt(diag((X^T X)^{-1}))_j  = sigma * ||R^{-1}[:,j]||

    sigma may be scalar → returns (p,)
    sigma may be array (q,) → returns (p, q)  [multi-outcome]
    """
    p = R.shape[0]
    Ri = solve_triangular(R, np.eye(p))          # R^{-1} (p, p)
    col_norms = np.sqrt(np.sum(Ri ** 2, axis=1)) # (p,) — row norms of R^{-1}

    if np.ndim(sigma) == 0 or np.isscalar(sigma):
        return float(sigma) * col_norms           # (p,)
    return np.outer(col_norms, np.asarray(sigma)) # (p, q)
