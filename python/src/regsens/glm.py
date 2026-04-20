"""GLMSensitivity: all-subsets GLM via IRLS warm-starting and Gray code enumeration."""

from __future__ import annotations

import time
import warnings
from typing import Optional

import numpy as np

from .families import Family, get_family
from .subset_engine import enumerate_subsets, _irls_fit
from .qr_core import qr_state_from_cols, compute_se
from .results import SensitivityResult
from ._exceptions import RankDeficiencyWarning, ConvergenceWarning


class GLMSensitivity:
    """
    Efficient all-subsets sensitivity analysis for GLMs (logistic, Poisson, Gaussian).

    For each subset of variable predictors, fits a GLM via IRLS. Between neighboring
    models (differing by one predictor), the linear predictor from the converged model
    is used to warm-start the next, typically reducing IRLS iterations from 5-15 to 1-3.

    Parameters
    ----------
    X : (m, k) array
        Design matrix WITHOUT intercept column.
    y : (m,) array
        Outcome vector.
    family : str or Family
        'binomial', 'poisson', or 'gaussian', or a Family instance.
    intercept_always : bool
        If True (default), prepend ones and treat as a fixed predictor.
    fixed_cols : list of int, optional
        0-based column indices in X to always include.
    variable_names : list of str, optional
        Names for columns of X.
    min_size, max_size : int
        Bounds on number of variable predictors per model.
    max_iter : int
        Maximum IRLS iterations per model (default 100).
    tol_irls : float
        IRLS convergence tolerance on relative deviance change (default 1e-8).
    tol_qr : float
        Rank-deficiency tolerance for QR operations (default 1e-10).
    warm_start : bool
        If True (default), seed IRLS from the previous model's linear predictor.
    n_jobs : int
        Parallel workers: -1 = all CPUs (default), 1 = sequential.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        family: str | Family = "binomial",
        intercept_always: bool = True,
        fixed_cols: Optional[list[int]] = None,
        variable_names: Optional[list[str]] = None,
        min_size: int = 0,
        max_size: Optional[int] = None,
        max_iter: int = 100,
        tol_irls: float = 1e-8,
        tol_qr: float = 1e-10,
        warm_start: bool = True,
        n_jobs: int = -1,
    ):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        m, k = X.shape

        self.family = get_family(family)

        if intercept_always:
            X_aug = np.column_stack([np.ones(m), X])
            shift = 1
            fixed_names_list = ["(Intercept)"]
        else:
            X_aug = X.copy()
            shift = 0
            fixed_names_list = []

        n_aug = X_aug.shape[1]
        user_fixed_aug = sorted({c + shift for c in (fixed_cols or [])})
        intercept_set = {0} if intercept_always else set()
        all_fixed = sorted(intercept_set | set(user_fixed_aug))
        all_variable = [j for j in range(n_aug) if j not in set(all_fixed)]
        k_var = len(all_variable)

        if k_var > 25:
            raise ValueError(
                f"Too many variable columns ({k_var} > 25). "
                "Use max_size to reduce enumeration space."
            )
        if k_var > 20:
            warnings.warn(
                f"All-subsets GLM enumeration with {k_var} variable columns "
                f"will evaluate 2^{k_var} = {2**k_var:,} models.",
                UserWarning, stacklevel=2,
            )

        if variable_names is not None:
            raw_names = list(variable_names)
        elif hasattr(X, "columns"):
            raw_names = list(X.columns)
        else:
            raw_names = [f"x{j}" for j in range(k)]

        col_to_name: dict[int, str] = {}
        for idx in all_fixed:
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

        self.X_aug = X_aug
        self.y = y
        self.fixed_cols = all_fixed
        self.variable_cols = all_variable
        self.col_to_name = col_to_name
        self.fixed_names = fixed_names_list
        self.variable_names = var_names_filtered
        self.m = m
        self.k_var = k_var
        self.min_size = min_size
        self.max_size = k_var if max_size is None else max_size
        self.max_iter = max_iter
        self.tol_irls = tol_irls
        self.tol_qr = tol_qr
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.null_deviance = self.family.null_deviance(y)

    # ------------------------------------------------------------------

    def fit_all_subsets(
        self,
        return_coef: bool = True,
        return_se: bool = True,
        return_pvalues: bool = True,
        return_deviance: bool = True,
        return_aic: bool = True,
        callback=None,
        progress: bool = False,
    ) -> SensitivityResult:
        """
        Fit all valid predictor subsets. Returns a SensitivityResult.
        """
        t0 = time.time()

        raw = enumerate_subsets(
            X=self.X_aug,
            y=self.y,
            Y=None,
            fixed_cols=self.fixed_cols,
            variable_cols=self.variable_cols,
            min_size=self.min_size,
            max_size=self.max_size,
            model_type="glm",
            family_dict=self.family.to_dict(),
            tol=self.tol_qr,
            return_coef=return_coef,
            return_se=return_se,
            return_pvalues=return_pvalues,
            n_jobs=self.n_jobs,
            warm_start=self.warm_start,
            tss=None,
            tss_per_outcome=None,
            null_deviance=self.null_deviance,
            max_iter_glm=self.max_iter,
            tol_irls=self.tol_irls,
        )

        if callback is not None:
            for r in raw:
                callback(r)

        return SensitivityResult(
            results=raw,
            col_to_name=self.col_to_name,
            variable_names=self.variable_names,
            fixed_names=self.fixed_names,
            outcome_names=None,
            family=self.family.name,
            n_obs=self.m,
            n_variable_cols=self.k_var,
            n_subsets=len(raw),
            elapsed_sec=time.time() - t0,
        )

    def fit_subset(self, col_mask) -> dict:
        """Fit a specific subset identified by a boolean mask over variable_cols."""
        col_mask = list(col_mask)
        active_var = [self.variable_cols[j] for j, v in enumerate(col_mask) if v]
        all_active = self.fixed_cols + active_var
        X_sub = self.X_aug[:, all_active]
        p = len(all_active)

        coef, eta, mu, deviance, R_last, n_iter, converged = _irls_fit(
            X_sub, self.y, self.family, self.max_iter, self.tol_irls
        )
        if not converged:
            warnings.warn(
                f"IRLS did not converge after {n_iter} iterations.",
                ConvergenceWarning, stacklevel=2,
            )

        se = np.full(p, np.nan)
        if R_last is not None and p > 0:
            signs = np.sign(np.diag(R_last))
            signs[signs == 0] = 1
            R_fixed = R_last * signs[:, np.newaxis]
            se = compute_se(R_fixed, 1.0)

        from scipy import stats
        z_vals = coef / np.where(se == 0, np.inf, se)
        aic = deviance + 2 * p
        bic = deviance + np.log(self.m) * p

        return {
            "col_mask": col_mask,
            "active_col_indices": all_active,
            "n_var": int(np.sum(col_mask)),
            "n_obs": self.m, "n_params": p,
            "coef": coef, "se": se,
            "z_values": z_vals,
            "pvalues": 2 * stats.norm.sf(np.abs(z_vals)),
            "deviance": deviance,
            "null_deviance": self.null_deviance,
            "mcfadden_r2": 1.0 - deviance / self.null_deviance if self.null_deviance > 0 else np.nan,
            "aic": aic, "bic": bic,
            "n_iter": n_iter, "converged": converged,
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
            outcome_names=None,
            family=self.family.name,
            n_obs=self.m,
            n_variable_cols=self.k_var,
            n_subsets=len(raw),
            elapsed_sec=time.time() - t0,
        )
