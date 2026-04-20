"""
Result containers for sensitivity analyses.

SensitivityResult wraps the raw list of result dicts from enumerate_subsets and
provides DataFrame conversion, variable summary, and model ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SensitivityResult:
    """
    Container for all subset results returned by fit_all_subsets().

    Attributes
    ----------
    results       : list of raw result dicts from enumerate_subsets
    col_to_name   : mapping from 0-based column index → variable name (for all columns)
    variable_names: names for the k variable (enumerable) columns
    fixed_names   : names for fixed columns (e.g. ['(Intercept)'])
    outcome_names : names for outcome columns (None for single outcome)
    family        : 'gaussian' / 'binomial' / 'poisson' / None for linear
    n_obs         : number of observations
    n_variable_cols: number of variable predictors (k)
    n_subsets     : number of fitted models
    elapsed_sec   : wall-clock time for fit_all_subsets()
    """

    results: list[dict]
    col_to_name: dict[int, str]
    variable_names: list[str]
    fixed_names: list[str]
    outcome_names: Optional[list[str]]
    family: Optional[str]
    n_obs: int
    n_variable_cols: int
    n_subsets: int
    elapsed_sec: float

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _all_names(self) -> list[str]:
        """All coefficient names: fixed first, then variable."""
        return self.fixed_names + self.variable_names

    def _name_to_col(self) -> dict[str, int]:
        return {v: k for k, v in self.col_to_name.items()}

    def _is_linear(self) -> bool:
        return self.family is None or self.family == "gaussian"

    # ------------------------------------------------------------------
    # DataFrame conversion
    # ------------------------------------------------------------------

    def to_dataframe(self, format: str = "wide"):
        """
        Convert results to a pandas DataFrame.

        Parameters
        ----------
        format : 'wide' — one row per subset (default)
                 'long' — one row per (subset, predictor)

        Requires pandas: pip install pandas
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install with: pip install regsens[pandas]"
            )
        if format == "wide":
            return self._to_wide(pd)
        elif format == "long":
            return self._to_long(pd)
        else:
            raise ValueError(f"format must be 'wide' or 'long', got {format!r}")

    def _to_wide(self, pd):
        all_names = self._all_names()
        is_lin = self._is_linear()
        rows = []

        for r in self.results:
            active = r["active_col_indices"]
            idx_to_pos = {idx: pos for pos, idx in enumerate(active)}

            # Variable names included in this model (excluding fixed)
            included = [
                self.col_to_name[i]
                for i in active
                if self.col_to_name.get(i) in self.variable_names
            ]

            row: dict = {
                "subset_id":     r.get("subset_id"),
                "included_vars": ",".join(included) if included else "(none)",
                "n_vars":        r["n_var"],
                "n_params":      r["n_params"],
            }

            if is_lin:
                n_out = r.get("n_outcomes", 1)
                if n_out == 1:
                    row["rss"]     = r.get("rss", np.nan)
                    row["r2"]      = r.get("r2", np.nan)
                    row["adj_r2"]  = r.get("adj_r2", np.nan)
                    row["sigma"]   = r.get("sigma", np.nan)
                else:
                    rss = r.get("rss", np.full(n_out, np.nan))
                    r2  = r.get("r2",  np.full(n_out, np.nan))
                    onames = self.outcome_names or [str(i) for i in range(n_out)]
                    for qi, on in enumerate(onames):
                        row[f"rss_{on}"]  = rss[qi] if hasattr(rss, "__len__") else np.nan
                        row[f"r2_{on}"]   = r2[qi]  if hasattr(r2,  "__len__") else np.nan
            else:
                row["deviance"]   = r.get("deviance", np.nan)
                row["pseudo_r2"]  = r.get("mcfadden_r2", np.nan)
                row["n_iter"]     = r.get("n_iter")
                row["converged"]  = r.get("converged")

            row["aic"]           = r.get("aic", np.nan)
            row["bic"]           = r.get("bic", np.nan)
            row["rank_deficient"]= r.get("rank_deficient", False)

            # Per-variable coefficient columns
            coef = r.get("coef")
            se   = r.get("se")
            pval = r.get("pvalues")
            n_out = r.get("n_outcomes", 1)

            for name in all_names:
                col_idx = next((k for k, v in self.col_to_name.items() if v == name), None)
                if col_idx is not None and col_idx in idx_to_pos:
                    pos = idx_to_pos[col_idx]
                    if coef is not None:
                        if n_out == 1:
                            row[f"coef_{name}"] = float(coef[pos]) if coef.ndim == 1 else float(coef[pos, 0])
                        else:
                            onames = self.outcome_names or [str(i) for i in range(n_out)]
                            for qi, on in enumerate(onames):
                                row[f"coef_{name}_{on}"] = float(coef[pos, qi])
                    if se is not None:
                        row[f"se_{name}"] = float(se[pos]) if se.ndim == 1 else float(se[pos, 0])
                    if pval is not None:
                        row[f"pval_{name}"] = float(pval[pos]) if pval.ndim == 1 else float(pval[pos, 0])
                else:
                    if coef is not None:
                        if n_out == 1:
                            row[f"coef_{name}"] = np.nan
                        else:
                            onames = self.outcome_names or [str(i) for i in range(n_out)]
                            for qi, on in enumerate(onames):
                                row[f"coef_{name}_{on}"] = np.nan
                    if se is not None:
                        row[f"se_{name}"] = np.nan
                    if pval is not None:
                        row[f"pval_{name}"] = np.nan

            rows.append(row)

        return pd.DataFrame(rows)

    def _to_long(self, pd):
        rows = []
        for r in self.results:
            active = r["active_col_indices"]
            idx_to_pos = {idx: pos for pos, idx in enumerate(active)}
            coef = r.get("coef")
            se   = r.get("se")
            pval = r.get("pvalues")
            tv   = r.get("t_values") if r.get("t_values") is not None else r.get("z_values")

            for idx, name in sorted(self.col_to_name.items()):
                included = idx in idx_to_pos
                pos = idx_to_pos.get(idx)
                rows.append({
                    "subset_id":      r.get("subset_id"),
                    "predictor":      name,
                    "included":       included,
                    "n_vars":         r["n_var"],
                    "coef":           float(coef[pos]) if (coef is not None and included and coef.ndim == 1) else np.nan,
                    "se":             float(se[pos])   if (se   is not None and included and se.ndim   == 1) else np.nan,
                    "t_or_z":         float(tv[pos])   if (tv   is not None and included) else np.nan,
                    "pvalue":         float(pval[pos]) if (pval is not None and included and pval.ndim == 1) else np.nan,
                    "rss_or_deviance": r.get("rss") if self._is_linear() else r.get("deviance"),
                    "aic":            r.get("aic"),
                })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Summary methods
    # ------------------------------------------------------------------

    def variable_summary(self):
        """
        Per-variable summary across all models where that variable is included.

        Returns a pandas DataFrame with columns:
          variable, n_models, coef_mean, coef_sd, coef_min, coef_max,
          pct_significant_05, pct_significant_01
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required: pip install regsens[pandas]")

        rows = []
        for name in self.variable_names:
            col_idx = next((k for k, v in self.col_to_name.items() if v == name), None)
            coefs, pvals = [], []
            for r in self.results:
                active = r["active_col_indices"]
                if col_idx in active:
                    pos = active.index(col_idx)
                    c = r.get("coef")
                    p = r.get("pvalues")
                    if c is not None and c.ndim == 1:
                        coefs.append(float(c[pos]))
                    if p is not None and p.ndim == 1:
                        pvals.append(float(p[pos]))

            if not coefs:
                continue
            coefs = np.array(coefs)
            pvals = np.array(pvals) if pvals else np.array([])
            rows.append({
                "variable":          name,
                "n_models":          len(coefs),
                "coef_mean":         float(np.mean(coefs)),
                "coef_sd":           float(np.std(coefs, ddof=min(1, len(coefs) - 1))),
                "coef_min":          float(np.min(coefs)),
                "coef_max":          float(np.max(coefs)),
                "pct_significant_05": float(np.mean(pvals < 0.05)) if len(pvals) else np.nan,
                "pct_significant_01": float(np.mean(pvals < 0.01)) if len(pvals) else np.nan,
            })

        try:
            import pandas as pd
            return pd.DataFrame(rows)
        except ImportError:
            return rows

    def best_subsets(self, criterion: str = "aic", n: int = 10):
        """
        Return the top-n models sorted by criterion.

        criterion : 'aic', 'bic', 'r2', 'rss', 'deviance'
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required: pip install regsens[pandas]")

        reverse = criterion in ("r2",)

        def key(r):
            v = r.get(criterion, np.nan)
            if isinstance(v, np.ndarray):
                v = float(v.flat[0])
            return (v is None or np.isnan(v), v if v is not None else np.inf)

        sorted_results = sorted(self.results, key=key, reverse=reverse)
        top = sorted_results[:n]
        tmp = SensitivityResult(
            results=top,
            col_to_name=self.col_to_name,
            variable_names=self.variable_names,
            fixed_names=self.fixed_names,
            outcome_names=self.outcome_names,
            family=self.family,
            n_obs=self.n_obs,
            n_variable_cols=self.n_variable_cols,
            n_subsets=len(top),
            elapsed_sec=0.0,
        )
        return tmp.to_dataframe("wide")

    def coefficient_stability(
        self,
        variable: str | int,
        conditioning_on: Optional[list[str | int]] = None,
    ):
        """
        Coefficient of `variable` across all models where it is included.

        conditioning_on: if given, restrict to models that also include these variables.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required: pip install regsens[pandas]")

        n2col = self._name_to_col()
        if isinstance(variable, str):
            target_idx = n2col.get(variable)
        else:
            target_idx = variable

        cond_idxs = set()
        if conditioning_on:
            for v in conditioning_on:
                idx = n2col.get(v) if isinstance(v, str) else v
                if idx is not None:
                    cond_idxs.add(idx)

        rows = []
        for r in self.results:
            active = r["active_col_indices"]
            if target_idx not in active:
                continue
            if cond_idxs and not cond_idxs.issubset(set(active)):
                continue
            pos = active.index(target_idx)
            c    = r.get("coef")
            se   = r.get("se")
            pval = r.get("pvalues")
            included = [
                self.col_to_name.get(i, str(i))
                for i in active
                if self.col_to_name.get(i) in self.variable_names
            ]
            rows.append({
                "subset_id":    r.get("subset_id"),
                "included_vars": ",".join(included),
                "n_vars":       r["n_var"],
                "coef":         float(c[pos])    if c    is not None and c.ndim == 1    else np.nan,
                "se":           float(se[pos])   if se   is not None and se.ndim == 1   else np.nan,
                "pvalue":       float(pval[pos]) if pval is not None and pval.ndim == 1 else np.nan,
                "aic":          r.get("aic", np.nan),
            })

        return pd.DataFrame(rows)
