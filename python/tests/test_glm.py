"""Tests for GLMSensitivity."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from regsens import GLMSensitivity

RNG = np.random.default_rng(13)

try:
    import statsmodels.api as sm
    HAS_SM = True
except ImportError:
    HAS_SM = False

needs_sm = pytest.mark.skipif(not HAS_SM, reason="statsmodels not installed")


def _make_binary(m=60, k=2, seed=13):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((m, k))
    eta = 0.5 * X[:, 0] - 0.3 * X[:, 1]
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, prob).astype(float)
    return X, y


def _make_counts(m=60, k=2, seed=14):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((m, k))
    mu = np.exp(0.3 + 0.4 * X[:, 0] - 0.2 * X[:, 1])
    y = rng.poisson(mu).astype(float)
    return X, y


# ---------------------------------------------------------------------------
# Count
# ---------------------------------------------------------------------------

def test_all_subsets_count_k3():
    X, y = _make_binary(k=3)
    sens = GLMSensitivity(X, y, family="binomial")
    result = sens.fit_all_subsets()
    assert result.n_subsets == 2 ** 3


# ---------------------------------------------------------------------------
# Correctness vs statsmodels
# ---------------------------------------------------------------------------

@needs_sm
def test_binomial_coef_matches_statsmodels():
    X, y = _make_binary(m=80, k=2)
    sens = GLMSensitivity(X, y, family="binomial")
    result = sens.fit_all_subsets()

    X_aug = sens.X_aug  # with intercept
    for r in result.results:
        active = r["active_col_indices"]
        X_sub = X_aug[:, active]
        sm_model = sm.Logit(y, X_sub).fit(disp=0, maxiter=200)
        assert_allclose(r["coef"], sm_model.params, atol=1e-4,
                        err_msg=f"Binomial coef mismatch for mask {r['col_mask']}")


@needs_sm
def test_poisson_coef_matches_statsmodels():
    X, y = _make_counts(m=80, k=2)
    sens = GLMSensitivity(X, y, family="poisson")
    result = sens.fit_all_subsets()

    X_aug = sens.X_aug
    for r in result.results:
        active = r["active_col_indices"]
        X_sub = X_aug[:, active]
        sm_model = sm.Poisson(y, X_sub).fit(disp=0, maxiter=200)
        assert_allclose(r["coef"], sm_model.params, atol=1e-4,
                        err_msg=f"Poisson coef mismatch for mask {r['col_mask']}")


@needs_sm
def test_aic_matches_statsmodels():
    X, y = _make_binary(m=80, k=2)
    sens = GLMSensitivity(X, y, family="binomial")
    result = sens.fit_all_subsets()

    X_aug = sens.X_aug
    for r in result.results:
        active = r["active_col_indices"]
        X_sub = X_aug[:, active]
        sm_model = sm.Logit(y, X_sub).fit(disp=0, maxiter=200)
        assert r["aic"] == pytest.approx(sm_model.aic, abs=0.5)


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------

def test_deviance_positive():
    X, y = _make_binary(k=3)
    sens = GLMSensitivity(X, y, family="binomial")
    result = sens.fit_all_subsets()
    for r in result.results:
        assert r["deviance"] > 0


def test_pseudo_r2_in_range():
    X, y = _make_binary(k=3)
    sens = GLMSensitivity(X, y, family="binomial")
    result = sens.fit_all_subsets()
    for r in result.results:
        if "mcfadden_r2" in r:
            assert -0.1 <= r["mcfadden_r2"] <= 1.1


def test_warm_start_converges():
    X, y = _make_binary(k=3)
    sens = GLMSensitivity(X, y, family="binomial", warm_start=True)
    result = sens.fit_all_subsets()
    n_converged = sum(1 for r in result.results if r.get("converged", False))
    # Most models should converge
    assert n_converged >= len(result.results) * 0.9


def test_fit_subset_matches_all_subsets():
    X, y = _make_binary(k=3)
    sens = GLMSensitivity(X, y, family="binomial")
    result = sens.fit_all_subsets()

    mask = [True, False, True]
    r_all = next(r for r in result.results if r["col_mask"] == mask)
    r_sub = sens.fit_subset(mask)
    assert_allclose(r_sub["coef"], r_all["coef"], atol=1e-4)


# ---------------------------------------------------------------------------
# DataFrame
# ---------------------------------------------------------------------------

def test_to_dataframe_wide():
    X, y = _make_binary(k=3)
    sens = GLMSensitivity(X, y, family="binomial")
    result = sens.fit_all_subsets()
    try:
        df = result.to_dataframe("wide")
        assert len(df) == 2 ** 3
        assert "deviance" in df.columns
        assert "pseudo_r2" in df.columns
    except ImportError:
        pytest.skip("pandas not installed")
