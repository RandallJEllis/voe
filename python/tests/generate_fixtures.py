"""
Generate cross-language test fixtures for the Python QR core.

Produces:
  tests/fixtures/X.csv          — design matrix (n x k)
  tests/fixtures/y_linear.csv   — continuous outcome
  tests/fixtures/y_binary.csv   — binary outcome
  tests/fixtures/y_count.csv    — count outcome
  tests/fixtures/refs_linear.csv     — OLS reference (all 2^k subsets)
  tests/fixtures/refs_binomial.csv   — logistic GLM reference
  tests/fixtures/refs_poisson.csv    — Poisson GLM reference

Reference values are computed with numpy (OLS) and statsmodels (GLM),
which are independent of regsens. Both the Python and R packages are
expected to match these references to within a tight tolerance.

Usage:
    cd /path/to/voe/python
    python tests/generate_fixtures.py
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

FIXTURES = Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# Reproducible data
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(2024_03_20)

N = 60   # observations
K = 4    # variable predictors (2^4 = 16 subsets)

# Moderately correlated design
Sigma = 0.3 * np.ones((K, K)) + 0.7 * np.eye(K)
L = np.linalg.cholesky(Sigma)
X = RNG.standard_normal((N, K)) @ L.T

# True signals
BETA_LINEAR  = np.array([2.0, 0.0, -1.5, 0.0])  # intercept added separately
BETA_LOGIT   = np.array([0.8, 0.0, -0.6, 0.0])
BETA_POISSON = np.array([0.5, 0.0, -0.3, 0.0])
INTERCEPT    = 0.5

y_linear = INTERCEPT + X @ BETA_LINEAR  + RNG.standard_normal(N)
y_binary = RNG.binomial(1, 1 / (1 + np.exp(-(INTERCEPT + X @ BETA_LOGIT)))).astype(float)
y_count  = RNG.poisson(np.exp(INTERCEPT + X @ BETA_POISSON)).astype(float)

# ---------------------------------------------------------------------------
# Save raw data
# ---------------------------------------------------------------------------
pd.DataFrame(X, columns=[f"x{j+1}" for j in range(K)]).to_csv(
    FIXTURES / "X.csv", index=False
)
pd.Series(y_linear, name="y").to_csv(FIXTURES / "y_linear.csv", index=False)
pd.Series(y_binary, name="y").to_csv(FIXTURES / "y_binary.csv", index=False)
pd.Series(y_count,  name="y").to_csv(FIXTURES / "y_count.csv",  index=False)
print("Saved X, y_linear, y_binary, y_count.")

# ---------------------------------------------------------------------------
# Helper: iterate all 2^K subsets in Gray-code order
# ---------------------------------------------------------------------------
def gray(i):
    return i ^ (i >> 1)

def gray_diff(i):
    """Return (bit_pos 0-based, is_add) for transition i-1 -> i."""
    lsb = (i & -i)
    bit_pos = lsb.bit_length() - 1
    g = gray(i)
    is_add = bool((g >> bit_pos) & 1)
    return bit_pos, is_add

# ---------------------------------------------------------------------------
# Linear reference: OLS via numpy lstsq
# ---------------------------------------------------------------------------
X_aug = np.column_stack([np.ones(N), X])   # (N, K+1)
var_names = [f"x{j+1}" for j in range(K)]
all_names = ["(Intercept)"] + var_names

tss = np.sum((y_linear - y_linear.mean()) ** 2)

rows = []
for i in range(2**K):
    g = gray(i)
    mask = [(g >> j) & 1 for j in range(K)]  # bit j -> variable j
    active_var_idx = [j for j in range(K) if mask[j]]
    active_cols = [0] + [j + 1 for j in active_var_idx]   # 0 = intercept
    X_sub = X_aug[:, active_cols]
    beta, _, _, _ = np.linalg.lstsq(X_sub, y_linear, rcond=None)
    residuals = y_linear - X_sub @ beta
    rss = float(np.dot(residuals, residuals))
    n_params = len(active_cols)
    df_res = N - n_params
    sigma = np.sqrt(rss / df_res)
    r2 = 1.0 - rss / tss
    # AIC convention: count sigma² as an extra parameter, matching both packages.
    aic = N * np.log(rss / N) + 2.0 * (n_params + 1)
    # SE: sigma * sqrt(diag((X'X)^{-1}))
    XtX_inv = np.linalg.inv(X_sub.T @ X_sub)
    se = sigma * np.sqrt(np.diag(XtX_inv))

    row = {
        "subset_id": g,
        "n_active":  len(active_var_idx),
    }
    for j, nm in enumerate(all_names):
        pos_in_active = [idx for idx, col in enumerate(active_cols) if col == (0 if nm == "(Intercept)" else var_names.index(nm) + 1)]
        if pos_in_active:
            row[f"coef_{nm}"] = float(beta[pos_in_active[0]])
            row[f"se_{nm}"]   = float(se[pos_in_active[0]])
        else:
            row[f"coef_{nm}"] = float("nan")
            row[f"se_{nm}"]   = float("nan")
    row["rss"] = rss
    row["r2"]  = float(r2)
    row["aic"] = float(aic)
    rows.append(row)

pd.DataFrame(rows).to_csv(FIXTURES / "refs_linear.csv", index=False)
print(f"Saved refs_linear.csv ({len(rows)} subsets).")

# ---------------------------------------------------------------------------
# GLM reference: statsmodels
# ---------------------------------------------------------------------------
try:
    import statsmodels.api as sm

    for family_name, sm_family, y_glm in [
        ("binomial", sm.families.Binomial(), y_binary),
        ("poisson",  sm.families.Poisson(),  y_count),
    ]:
        rows = []
        for i in range(2**K):
            g = gray(i)
            mask = [(g >> j) & 1 for j in range(K)]
            active_var_idx = [j for j in range(K) if mask[j]]
            active_cols = [0] + [j + 1 for j in active_var_idx]
            X_sub = X_aug[:, active_cols]

            model = sm.GLM(y_glm, X_sub, family=sm_family)
            fit = model.fit(disp=False)
            beta = fit.params
            deviance = float(fit.deviance)
            n_params = len(active_cols)
            # AIC convention: deviance + 2k, matching both packages (not the
            # full statsmodels AIC which includes the saturated log-likelihood).
            aic = deviance + 2.0 * n_params

            row = {
                "subset_id": g,
                "n_active":  len(active_var_idx),
            }
            for j, nm in enumerate(all_names):
                active_col_names = ["(Intercept)"] + [var_names[j] for j in active_var_idx]
                if nm in active_col_names:
                    pos = active_col_names.index(nm)
                    row[f"coef_{nm}"] = float(beta[pos])
                else:
                    row[f"coef_{nm}"] = float("nan")
            row["deviance"] = deviance
            row["aic"] = aic
            rows.append(row)

        out_path = FIXTURES / f"refs_{family_name}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"Saved {out_path.name} ({len(rows)} subsets).")

except ImportError:
    print("statsmodels not installed — GLM reference files not generated.")
    print("Install with: pip install statsmodels")

print("\nDone. Fixture files are in:", FIXTURES)
