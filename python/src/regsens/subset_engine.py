"""
Parallel all-subsets enumeration via hierarchical Gray code traversal.

Strategy
--------
Split k variable columns into k_outer "scheduling" bits and k_inner bits.

  2^k_outer independent chunks → each initialises its QR from scratch (O(mp²))
                                  then runs a sequential Gray code loop (O(mp) per step)

Chunks execute in parallel via ProcessPoolExecutor (bypasses the GIL; work is
CPU-bound NumPy/SciPy). Falls back to sequential on failure.

k_outer heuristic: min(ceil(log2(n_workers)) + 2, k−1, 8)
This gives ~4× oversubscription for good load balancing with at most 256 chunks.
"""

from __future__ import annotations

import multiprocessing
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np
from scipy import stats
from scipy.linalg import solve_triangular

from ._exceptions import ConvergenceWarning, RankDeficiencyWarning
from .gray_code import gray, gray_diff, gray_active_positions, popcount
from .qr_core import (
    QRState, compute_se,
    qr_add_column, qr_batch_outcomes, qr_new_outcome,
    qr_remove_column, qr_state_from_cols,
)


# ---------------------------------------------------------------------------
# IRLS
# ---------------------------------------------------------------------------

def _irls_fit(
    X_active: np.ndarray,
    y: np.ndarray,
    family,
    max_iter: int = 100,
    tol: float = 1e-8,
    start_eta: Optional[np.ndarray] = None,
) -> tuple:
    """
    Fit GLM by IRLS.  Returns (coef, eta, mu, deviance, R_last, n_iter, converged).

    R_last is the R matrix from the final weighted QR — used for SE computation.
    warm_start: pass start_eta from a nearby model's converged linear predictor.
    """
    m, p = X_active.shape

    if start_eta is not None:
        eta = start_eta.copy()
        mu = family.inv_link(eta)
    else:
        mu = family.init_mu(y)
        eta = family.link(mu)

    deviance_prev = np.inf
    coef = np.zeros(p)
    R_last = None

    for it in range(max_iter):
        w = family.irls_weights(mu)
        z = family.working_response(y, mu, eta)

        sqrt_w = np.sqrt(np.maximum(w, 0.0))
        Xw = X_active * sqrt_w[:, np.newaxis]   # W^{1/2} X
        zw = z * sqrt_w                         # W^{1/2} z

        Q, R = np.linalg.qr(Xw, mode="reduced")
        Qtzw = Q.T @ zw
        try:
            coef = solve_triangular(R, Qtzw)
        except Exception:
            break
        R_last = R

        eta = X_active @ coef
        mu = family.inv_link(eta)
        deviance = family.deviance(y, mu)

        if abs(deviance - deviance_prev) / (abs(deviance_prev) + 0.1) < tol:
            return coef, eta, mu, deviance, R_last, it + 1, True

        deviance_prev = deviance

    return coef, eta, mu, deviance_prev, R_last, max_iter, False


# ---------------------------------------------------------------------------
# Per-model fit helper (module-level for pickle compatibility)
# ---------------------------------------------------------------------------

def _fit_model(
    state: QRState,
    X_data: np.ndarray,
    y: Optional[np.ndarray],
    Y: Optional[np.ndarray],
    model_type: str,
    family_dict: Optional[dict],
    all_active_cols: list[int],
    col_mask: list[bool],
    n_var: int,
    subset_id: int,
    warm_eta: Optional[np.ndarray],
    return_coef: bool,
    return_se: bool,
    return_pvalues: bool,
    tss: Optional[float],
    tss_per_outcome: Optional[np.ndarray],
    null_deviance: Optional[float],
    max_iter_glm: int,
    tol_irls: float,
) -> dict:
    n = state.m
    p = state.p
    df_res = max(n - p, 1)

    base = {
        "subset_id": subset_id,
        "col_mask": col_mask,
        "active_col_indices": list(all_active_cols),
        "n_var": n_var,
        "n_obs": n,
        "n_params": p,
        "rank_deficient": False,
    }

    if model_type == "linear":
        if Y is not None:
            q = Y.shape[1]
            batch = qr_batch_outcomes(state, Y)
            sigma = np.sqrt(batch["rss"] / df_res)
            result = {**base, "n_outcomes": q, "rss": batch["rss"], "sigma": sigma}
            if tss_per_outcome is not None:
                tss_safe = np.where(tss_per_outcome > 0, tss_per_outcome, np.inf)
                result["r2"] = 1.0 - batch["rss"] / tss_safe
            aic_arr = n * np.log(np.maximum(batch["rss"], 1e-300) / n) + 2 * (p + 1)
            bic_arr = n * np.log(np.maximum(batch["rss"], 1e-300) / n) + np.log(n) * (p + 1)
            result["aic"] = aic_arr
            result["bic"] = bic_arr
            if return_coef:
                result["coef"] = batch["beta"]
            if return_se and p > 0:
                result["se"] = compute_se(state.R, sigma)
                if return_pvalues:
                    se = result["se"]
                    t_vals = batch["beta"] / np.where(se == 0, np.inf, se)
                    result["t_values"] = t_vals
                    result["pvalues"] = 2 * stats.t.sf(np.abs(t_vals), df=df_res)
            return result

        # Single outcome
        fit = qr_new_outcome(state, y)
        rss = fit["rss"]
        sigma = float(np.sqrt(rss / df_res))
        aic = n * np.log(max(rss, 1e-300) / n) + 2 * (p + 1)
        bic = n * np.log(max(rss, 1e-300) / n) + np.log(n) * (p + 1)
        result = {**base, "rss": rss, "sigma": sigma, "aic": aic, "bic": bic}
        if tss is not None and tss > 0:
            r2 = 1.0 - rss / tss
            result["r2"] = r2
            result["adj_r2"] = 1.0 - (1.0 - r2) * (n - 1) / df_res
        if return_coef:
            result["coef"] = fit["beta"]
        if return_se and p > 0:
            se = compute_se(state.R, sigma)
            result["se"] = se
            if return_pvalues:
                t_vals = fit["beta"] / np.where(se == 0, np.inf, se)
                result["t_values"] = t_vals
                result["pvalues"] = 2 * stats.t.sf(np.abs(t_vals), df=df_res)
        return result

    # GLM
    from .families import Family
    family = Family.from_dict(family_dict)
    X_sub = X_data[:, all_active_cols]
    coef, eta, mu, deviance, R_last, n_iter, converged = _irls_fit(
        X_sub, y, family, max_iter_glm, tol_irls, start_eta=warm_eta
    )
    if not converged:
        warnings.warn(
            f"IRLS did not converge for subset_id={subset_id} after {n_iter} iterations.",
            ConvergenceWarning, stacklevel=2,
        )
    aic = deviance + 2 * p
    bic = deviance + np.log(n) * p
    result = {
        **base,
        "deviance": deviance,
        "null_deviance": null_deviance,
        "aic": aic,
        "bic": bic,
        "n_iter": n_iter,
        "converged": converged,
        "_eta": eta,   # stripped before returning to caller; used for warm-start
    }
    if null_deviance is not None and null_deviance > 0:
        result["mcfadden_r2"] = 1.0 - deviance / null_deviance
    if return_coef:
        result["coef"] = coef
    if return_se and p > 0 and R_last is not None:
        # Fix R_last sign convention (positive diagonal) before computing SE
        signs = np.sign(np.diag(R_last))
        signs[signs == 0] = 1
        R_last = R_last * signs[:, np.newaxis]
        se = compute_se(R_last, 1.0)
        result["se"] = se
        if return_pvalues:
            z_vals = coef / np.where(se == 0, np.inf, se)
            result["z_values"] = z_vals
            result["pvalues"] = 2 * stats.norm.sf(np.abs(z_vals))
    return result


# ---------------------------------------------------------------------------
# Module-level chunk worker (must be top-level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------

def _chunk_worker(args: tuple) -> list[dict]:
    """
    Process all 2^k_inner inner-variable subsets for a fixed outer configuration.

    MUST be a top-level function so it is picklable by ProcessPoolExecutor.
    """
    (
        X_data, y_data, Y_data,
        fixed_cols, outer_active_cols, inner_cols,
        variable_cols,
        min_size, max_size,
        model_type, family_dict,
        tol,
        return_coef, return_se, return_pvalues,
        warm_start,
        tss, tss_per_outcome, null_deviance,
        max_iter_glm, tol_irls,
        outer_gray_val,
    ) = args

    k_inner = len(inner_cols)
    k_total = len(variable_cols)
    k_outer = k_total - k_inner
    n_outer_active = len(outer_active_cols)
    outer_active_set = set(outer_active_cols)

    # Initialize QR from fixed + outer columns
    init_cols = fixed_cols + outer_active_cols
    state = qr_state_from_cols(X_data, init_cols, tol=tol)

    current_inner_pos: list[int] = []  # bit positions (0-based) into inner_cols
    warm_eta: Optional[np.ndarray] = None
    results = []

    for i in range(1 << k_inner):
        rank_deficient = False

        if i > 0:
            bit_pos, is_add = gray_diff(i)
            if is_add:
                col = inner_cols[bit_pos]
                try:
                    qr_add_column(state, X_data[:, col], col, inplace=True)
                    current_inner_pos.append(bit_pos)
                except Exception:
                    rank_deficient = True
            else:
                if bit_pos in current_inner_pos:
                    pos_in_list = current_inner_pos.index(bit_pos)
                    qr_pos = len(fixed_cols) + n_outer_active + pos_in_list
                    qr_remove_column(state, qr_pos, inplace=True)
                    current_inner_pos.remove(bit_pos)

        n_var = n_outer_active + len(current_inner_pos)
        if not (min_size <= n_var <= max_size):
            continue

        # Reconstruct full column lists
        active_inner = [inner_cols[j] for j in current_inner_pos]
        all_active = fixed_cols + outer_active_cols + active_inner

        # col_mask over variable_cols (length k_total)
        active_var_set = outer_active_set | set(active_inner)
        col_mask = [(variable_cols[j] in active_var_set) for j in range(k_total)]

        # Unique subset_id: outer Gray code shifted left, OR inner Gray code
        subset_id = (outer_gray_val << k_inner) | gray(i)

        try:
            result = _fit_model(
                state=state,
                X_data=X_data,
                y=y_data,
                Y=Y_data,
                model_type=model_type,
                family_dict=family_dict,
                all_active_cols=all_active,
                col_mask=col_mask,
                n_var=n_var,
                subset_id=subset_id,
                warm_eta=warm_eta if warm_start else None,
                return_coef=return_coef,
                return_se=return_se,
                return_pvalues=return_pvalues,
                tss=tss,
                tss_per_outcome=tss_per_outcome,
                null_deviance=null_deviance,
                max_iter_glm=max_iter_glm,
                tol_irls=tol_irls,
            )
        except Exception as exc:
            warnings.warn(f"Model fit failed (subset_id={subset_id}): {exc}",
                          RuntimeWarning, stacklevel=2)
            continue

        if rank_deficient:
            result["rank_deficient"] = True

        warm_eta = result.pop("_eta", None)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enumerate_subsets(
    X: np.ndarray,
    y: Optional[np.ndarray],
    Y: Optional[np.ndarray],
    fixed_cols: list[int],
    variable_cols: list[int],
    min_size: int,
    max_size: int,
    model_type: str,
    family_dict: Optional[dict],
    tol: float,
    return_coef: bool,
    return_se: bool,
    return_pvalues: bool,
    n_jobs: int,
    warm_start: bool,
    tss: Optional[float],
    tss_per_outcome: Optional[np.ndarray],
    null_deviance: Optional[float],
    max_iter_glm: int = 100,
    tol_irls: float = 1e-8,
) -> list[dict]:
    """
    Enumerate all valid subsets of variable_cols, fitting a model for each.

    Parameters
    ----------
    X             : (m, n) full design matrix
    y / Y         : outcome(s) — exactly one must be non-None
    fixed_cols    : column indices always included (0-based)
    variable_cols : column indices to enumerate over (k columns → 2^k subsets)
    min_size      : min number of variable cols included
    max_size      : max number of variable cols included
    model_type    : 'linear' or 'glm'
    family_dict   : {'name': 'binomial'} etc.  for GLM
    n_jobs        : -1 = all CPUs, 1 = sequential, >1 = explicit count
    warm_start    : for GLM, seed IRLS from previous model's linear predictor
    tss           : total sum of squares of y (for linear R²)
    tss_per_outcome : (q,) total SS per outcome column
    null_deviance : null model deviance (for GLM pseudo-R²)
    """
    k = len(variable_cols)

    # Edge case: no variable columns
    if k == 0:
        state = qr_state_from_cols(X, fixed_cols, tol=tol)
        result = _fit_model(
            state=state, X_data=X, y=y, Y=Y,
            model_type=model_type, family_dict=family_dict,
            all_active_cols=fixed_cols, col_mask=[],
            n_var=0, subset_id=0, warm_eta=None,
            return_coef=return_coef, return_se=return_se, return_pvalues=return_pvalues,
            tss=tss, tss_per_outcome=tss_per_outcome, null_deviance=null_deviance,
            max_iter_glm=max_iter_glm, tol_irls=tol_irls,
        )
        result.pop("_eta", None)
        return [result]

    # Determine worker count
    if n_jobs == -1:
        n_workers = multiprocessing.cpu_count()
    elif n_jobs <= 0:
        n_workers = 1
    else:
        n_workers = n_jobs

    # Split into outer (scheduling) and inner (sequential) bits
    if n_workers <= 1 or k == 1:
        k_outer = 0
    else:
        k_outer = min(int(np.ceil(np.log2(max(n_workers, 2)))) + 2, k - 1, 8)
        k_outer = max(k_outer, 0)

    k_inner = k - k_outer
    outer_cols = variable_cols[:k_outer]
    inner_cols = variable_cols[k_outer:]

    # Build one args tuple per outer Gray-code configuration
    tasks = []
    for outer_i in range(1 << k_outer):
        outer_gray_val = gray(outer_i)
        active_outer = [outer_cols[j] for j in range(k_outer) if (outer_gray_val >> j) & 1]
        tasks.append((
            X, y, Y,
            fixed_cols, active_outer, inner_cols,
            variable_cols,
            min_size, max_size,
            model_type, family_dict,
            tol,
            return_coef, return_se, return_pvalues,
            warm_start,
            tss, tss_per_outcome, null_deviance,
            max_iter_glm, tol_irls,
            outer_gray_val,
        ))

    # Execute: parallel or sequential
    if n_workers <= 1 or k_outer == 0:
        chunk_results = [_chunk_worker(t) for t in tasks]
    else:
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                chunk_results = list(executor.map(_chunk_worker, tasks))
        except Exception as exc:
            warnings.warn(
                f"Parallel execution failed ({exc}), falling back to sequential.",
                RuntimeWarning, stacklevel=2,
            )
            chunk_results = [_chunk_worker(t) for t in tasks]

    return [r for chunk in chunk_results for r in chunk]
