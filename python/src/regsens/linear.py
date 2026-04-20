"""LinearSensitivity: all-subsets OLS via QR reuse and Gray code enumeration."""

from __future__ import annotations

import time
import warnings
from typing import Optional

import numpy as np

from .qr_core import qr_state_from_cols, qr_new_outcome, qr_batch_outcomes, compute_se
from .subset_engine import enumerate_subsets
from .results import SensitivityResult
from ._exceptions import RankDeficiencyWarning


class LinearSensitivity:
    """
    Efficient all-subsets sensitivity analysis for linear regression.

    For each subset of variable predictors:
      - QR is computed once per subset (or updated from previous via O(mp) Givens step)
      - All outcomes in Y are solved simultaneously: O(mp · q) instead of O(mp² · q)
      - Gray code traversal ensures each step changes exactly one predictor

    Parameters
    ----------
    X : (m, k) array
        Design matrix WITHOUT intercept column.
    y : (m,) array, optional
        Single outcome. Mutually exclusive with Y.
    Y : (m, q) array, optional
        Multiple outcomes. QR is reused across all q outcomes for each subset.
    intercept_always : bool
        If True (default), prepend a column of ones and treat as a fixed predictor.
    fixed_cols : list of int, optional
        0-based column indices in X to always include in every model.
    variable_names : list of str, optional
        Names for columns of X (length k). Defaults to 'x0', 'x1', ...
    outcome_names : list of str, optional
        Names for columns of Y (length q).
    min_size : int
        Minimum number of variable predictors per model (default 0).
    max_size : int, optional
        Maximum number of variable predictors. Default: all k.
    tol : float
        Rank-deficiency tolerance (default 1e-10).
    n_jobs : int
        Parallel workers: -1 = all CPUs (default), 1 = sequential, N = N workers.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        Y: Optional[np.ndarray] = None,
        intercept_always: bool = True,
        fixed_cols: Optional[list[int]] = None,
        variable_names: Optional[list[str]] = None,
        outcome_names: Optional[list[str]] = None,
        min_size: int = 0,
        max_size: Optional[int] = None,
        tol: float = 1e-10,
        n_jobs: int = -1,
    ):
        if (y is None) == (Y is None):
            raise ValueError("Exactly one of y or Y must be provided.")

        X = np.asarray(X, dtype=float)
        m, k = X.shape

        # Build augmented X (prepend intercept if requested)
        if intercept_always:
            X_aug = np.column_stack([np.ones(m), X])
            shift = 1
            fixed_names_list = ["(Intercept)"]
        else:
            X_aug = X.copy()
            shift = 0
            fixed_names_list = []

        n_aug = X_aug.shape[1]

        # Fixed column indices (0-based into X_aug)
        user_fixed_aug = sorted({c + shift for c in (fixed_cols or [])})
        intercept_set = {0} if intercept_always else set()
        all_fixed = sorted(intercept_set | set(user_fixed_aug))
        all_variable = [j for j in range(n_aug) if j not in set(all_fixed)]
        k_var = len(all_variable)

        if k_var > 25:
            raise ValueError(
                f"Too many variable columns ({k_var} > 25). "
                "Use max_size to reduce the enumeration space."
            )
        if k_var > 20:
            warnings.warn(
                f"All-subsets enumeration with {k_var} variable columns "
                f"will evaluate 2^{k_var} = {2**k_var:,} models. "
                "This may take a long time.",
                UserWarning, stacklevel=2,
            )

        # Build variable names
        if variable_names is not None:
            raw_names = list(variable_names)
        elif hasattr(X, "columns"):     # pandas DataFrame
            raw_names = list(X.columns)
        else:
            raw_names = [f"x{j}" for j in range(k)]

        if len(raw_names) != k:
            raise ValueError(f"variable_names length {len(raw_names)} != X columns {k}")

        # col_to_name: maps X_aug column index → name
        col_to_name: dict[int, str] = {}
        for j, idx in enumerate(all_fixed):
            if idx == 0 and intercept_always:
                col_to_name[idx] = "(Intercept)"
            else:
                col_to_name[idx] = raw_names[idx - shift]
                fixed_names_list.append(raw_names[idx - shift])

        var_names_filtered = []
        for idx in all_variable:
            name = raw_names[idx - shift]
            col_to_name[idx] = name
            var_names_filtered.append(name)

        # Store outcomes
        if y is not None:
            y = np.asarray(y, dtype=float)
            self.y = y
            self.Y = None
            self.tss = float(np.sum((y - y.mean()) ** 2))
            self.tss_per_outcome = None
        else:
            Y = np.asarray(Y, dtype=float)
            self.y = None
            self.Y = Y
            self.tss = None
            self.tss_per_outcome = np.sum((Y - Y.mean(axis=0)) ** 2, axis=0)

        self.X_aug = X_aug
        self.fixed_cols = all_fixed
        self.variable_cols = all_variable
        self.col_to_name = col_to_name
        self.fixed_names = fixed_names_list
        self.variable_names = var_names_filtered
        self.outcome_names = outcome_names
        self.m = m
        self.k_var = k_var
        self.min_size = min_size
        self.max_size = k_var if max_size is None else max_size
        self.tol = tol
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------

    def fit_all_subsets(
        self,
        return_coef: bool = True,
        return_se: bool = True,
        return_pvalues: bool = True,
        callback=None,
        progress: bool = False,
    ) -> SensitivityResult:
        """
        Fit all valid predictor subsets.

        Returns a SensitivityResult with DataFrame conversion, variable summary,
        and model ranking methods.
        """
        t0 = time.time()

        raw = enumerate_subsets(
            X=self.X_aug,
            y=self.y,
            Y=self.Y,
            fixed_cols=self.fixed_cols,
            variable_cols=self.variable_cols,
            min_size=self.min_size,
            max_size=self.max_size,
            model_type="linear",
            family_dict=None,
            tol=self.tol,
            return_coef=return_coef,
            return_se=return_se,
            return_pvalues=return_pvalues,
            n_jobs=self.n_jobs,
            warm_start=False,
            tss=self.tss,
            tss_per_outcome=self.tss_per_outcome,
            null_deviance=None,
        )

        if callback is not None:
            for r in raw:
                callback(r)

        return SensitivityResult(
            results=raw,
            col_to_name=self.col_to_name,
            variable_names=self.variable_names,
            fixed_names=self.fixed_names,
            outcome_names=self.outcome_names,
            family=None,
            n_obs=self.m,
            n_variable_cols=self.k_var,
            n_subsets=len(raw),
            elapsed_sec=time.time() - t0,
        )

    def fit_subset(self, col_mask) -> dict:
        """
        Fit a specific subset identified by a boolean mask over variable_cols.

        col_mask : list/array of bool, length k_var
        """
        col_mask = list(col_mask)
        active_var = [self.variable_cols[j] for j, v in enumerate(col_mask) if v]
        all_active = self.fixed_cols + active_var
        state = qr_state_from_cols(self.X_aug, all_active, self.tol)
        n, p = self.m, state.p
        df_res = max(n - p, 1)

        if self.Y is not None:
            batch = qr_batch_outcomes(state, self.Y)
            sigma = np.sqrt(batch["rss"] / df_res)
            se = compute_se(state.R, sigma) if p > 0 else None
            return {
                "col_mask": col_mask,
                "active_col_indices": all_active,
                "n_var": int(np.sum(col_mask)),
                "n_obs": n, "n_params": p,
                "coef": batch["beta"],
                "se": se,
                "rss": batch["rss"],
                "sigma": sigma,
                "r2": 1.0 - batch["rss"] / np.where(self.tss_per_outcome > 0, self.tss_per_outcome, np.inf),
                "n_outcomes": self.Y.shape[1],
            }
        else:
            fit = qr_new_outcome(state, self.y)
            sigma = float(np.sqrt(fit["rss"] / df_res))
            se = compute_se(state.R, sigma) if p > 0 else np.array([])
            from scipy import stats
            t_vals = fit["beta"] / np.where(se == 0, np.inf, se) if p > 0 else np.array([])
            return {
                "col_mask": col_mask,
                "active_col_indices": all_active,
                "n_var": int(np.sum(col_mask)),
                "n_obs": n, "n_params": p,
                "coef": fit["beta"],
                "se": se,
                "t_values": t_vals,
                "pvalues": 2 * stats.t.sf(np.abs(t_vals), df=df_res) if p > 0 else np.array([]),
                "rss": fit["rss"],
                "sigma": sigma,
                "r2": 1.0 - fit["rss"] / self.tss if self.tss and self.tss > 0 else np.nan,
                "adj_r2": 1.0 - (1.0 - (1.0 - fit["rss"]/self.tss)) * (n-1)/df_res if self.tss and self.tss > 0 else np.nan,
                "aic": n * np.log(max(fit["rss"], 1e-300) / n) + 2 * (p + 1),
                "bic": n * np.log(max(fit["rss"], 1e-300) / n) + np.log(n) * (p + 1),
            }

    def fit_leave_one_out(self) -> SensitivityResult:
        """Fit all models with exactly one variable removed from the full model."""
        t0 = time.time()
        raw = []
        for j in range(self.k_var):
            mask = [True] * self.k_var
            mask[j] = False
            raw.append(self.fit_subset(mask))
        return SensitivityResult(
            results=raw,
            col_to_name=self.col_to_name,
            variable_names=self.variable_names,
            fixed_names=self.fixed_names,
            outcome_names=self.outcome_names,
            family=None,
            n_obs=self.m,
            n_variable_cols=self.k_var,
            n_subsets=len(raw),
            elapsed_sec=time.time() - t0,
        )

    def fit_add_one(self) -> SensitivityResult:
        """Fit all models with exactly one variable added to the null (intercept-only) model."""
        t0 = time.time()
        raw = []
        for j in range(self.k_var):
            mask = [False] * self.k_var
            mask[j] = True
            raw.append(self.fit_subset(mask))
        return SensitivityResult(
            results=raw,
            col_to_name=self.col_to_name,
            variable_names=self.variable_names,
            fixed_names=self.fixed_names,
            outcome_names=self.outcome_names,
            family=None,
            n_obs=self.m,
            n_variable_cols=self.k_var,
            n_subsets=len(raw),
            elapsed_sec=time.time() - t0,
        )
