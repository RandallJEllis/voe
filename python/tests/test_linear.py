"""Tests for LinearSensitivity."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from regsens import LinearSensitivity

RNG = np.random.default_rng(7)


def _ols_ref(X_aug, y):
    """Reference OLS via numpy lstsq."""
    beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    return beta


def _make_data(m=50, k=4, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((m, k))
    y = rng.standard_normal(m)
    return X, y


# ---------------------------------------------------------------------------
# Count
# ---------------------------------------------------------------------------

def test_all_subsets_count_k4():
    X, y = _make_data(k=4)
    sens = LinearSensitivity(X, y=y)
    result = sens.fit_all_subsets()
    assert result.n_subsets == 2 ** 4


def test_all_subsets_count_k3():
    X, y = _make_data(k=3)
    sens = LinearSensitivity(X, y=y)
    result = sens.fit_all_subsets()
    assert result.n_subsets == 2 ** 3


# ---------------------------------------------------------------------------
# Correctness vs lstsq
# ---------------------------------------------------------------------------

def test_coefficients_match_lstsq():
    X, y = _make_data(k=4)
    sens = LinearSensitivity(X, y=y, variable_names=["a","b","c","d"])
    result = sens.fit_all_subsets()

    # Check a few specific subsets
    for r in result.results:
        active = r["active_col_indices"]
        X_sub = sens.X_aug[:, active]
        ref_beta = _ols_ref(X_sub, y)
        assert_allclose(r["coef"], ref_beta, atol=1e-8,
                        err_msg=f"Coef mismatch for subset {r['col_mask']}")


def test_rss_correct():
    X, y = _make_data(k=3)
    sens = LinearSensitivity(X, y=y)
    result = sens.fit_all_subsets()
    for r in result.results:
        active = r["active_col_indices"]
        X_sub = sens.X_aug[:, active]
        residuals = y - X_sub @ r["coef"]
        expected_rss = float(np.sum(residuals ** 2))
        assert r["rss"] == pytest.approx(expected_rss, rel=1e-6)


def test_r2_in_bounds():
    X, y = _make_data(k=4)
    sens = LinearSensitivity(X, y=y)
    result = sens.fit_all_subsets()
    for r in result.results:
        if r["n_var"] > 0:
            assert 0.0 <= r.get("r2", 0.5) <= 1.0 + 1e-10


# ---------------------------------------------------------------------------
# Multi-outcome
# ---------------------------------------------------------------------------

def test_multi_outcome_count():
    X, _ = _make_data(k=3)
    Y = RNG.standard_normal((50, 3))
    sens = LinearSensitivity(X, Y=Y)
    result = sens.fit_all_subsets()
    assert result.n_subsets == 2 ** 3


def test_multi_outcome_coef_shape():
    X, _ = _make_data(k=3)
    Y = RNG.standard_normal((50, 4))
    sens = LinearSensitivity(X, Y=Y)
    result = sens.fit_all_subsets()
    for r in result.results:
        p = r["n_params"]
        assert r["coef"].shape == (p, 4)


def test_multi_outcome_matches_individual():
    X, _ = _make_data(k=3)
    m = X.shape[0]
    Y = RNG.standard_normal((m, 3))
    sens_multi = LinearSensitivity(X, Y=Y)
    result_multi = sens_multi.fit_all_subsets()

    for qi in range(3):
        sens_single = LinearSensitivity(X, y=Y[:, qi])
        result_single = sens_single.fit_all_subsets()

        # match by same col_mask
        mask_to_single = {tuple(r["col_mask"]): r for r in result_single.results}
        for r_m in result_multi.results:
            key = tuple(r_m["col_mask"])
            r_s = mask_to_single[key]
            assert_allclose(r_m["coef"][:, qi], r_s["coef"], atol=1e-8)


# ---------------------------------------------------------------------------
# fit_subset consistency
# ---------------------------------------------------------------------------

def test_fit_subset_matches_all_subsets():
    X, y = _make_data(k=4)
    sens = LinearSensitivity(X, y=y)
    result = sens.fit_all_subsets()

    # Pick a specific mask
    test_mask = [True, False, True, False]
    r_all = next(r for r in result.results if r["col_mask"] == test_mask)
    r_single = sens.fit_subset(test_mask)
    assert_allclose(r_single["coef"], r_all["coef"], atol=1e-8)


# ---------------------------------------------------------------------------
# fit_leave_one_out / fit_add_one
# ---------------------------------------------------------------------------

def test_leave_one_out_count():
    X, y = _make_data(k=5)
    sens = LinearSensitivity(X, y=y)
    result = sens.fit_leave_one_out()
    assert result.n_subsets == 5


def test_add_one_count():
    X, y = _make_data(k=5)
    sens = LinearSensitivity(X, y=y)
    result = sens.fit_add_one()
    assert result.n_subsets == 5


# ---------------------------------------------------------------------------
# SensitivityResult methods
# ---------------------------------------------------------------------------

def test_to_dataframe_wide_shape():
    X, y = _make_data(k=4)
    sens = LinearSensitivity(X, y=y)
    result = sens.fit_all_subsets()
    try:
        df = result.to_dataframe("wide")
        assert len(df) == 2 ** 4
    except ImportError:
        pytest.skip("pandas not installed")


def test_to_dataframe_long_shape():
    X, y = _make_data(k=3)
    sens = LinearSensitivity(X, y=y)
    result = sens.fit_all_subsets()
    try:
        df = result.to_dataframe("long")
        assert len(df) > 0
    except ImportError:
        pytest.skip("pandas not installed")


def test_variable_summary_n_models():
    """Each variable should appear in exactly 2^(k-1) models."""
    k = 4
    X, y = _make_data(k=k)
    sens = LinearSensitivity(X, y=y)
    result = sens.fit_all_subsets()
    try:
        vs = result.variable_summary()
        assert all(vs["n_models"] == 2 ** (k - 1))
    except ImportError:
        pytest.skip("pandas not installed")


def test_best_subsets_aic_ordering():
    X, y = _make_data(k=4)
    sens = LinearSensitivity(X, y=y)
    result = sens.fit_all_subsets()
    try:
        df = result.best_subsets("aic", n=5)
        aics = df["aic"].values
        assert all(aics[i] <= aics[i + 1] for i in range(len(aics) - 1))
    except ImportError:
        pytest.skip("pandas not installed")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_no_intercept_mode():
    X, y = _make_data(k=3)
    sens = LinearSensitivity(X, y=y, intercept_always=False)
    result = sens.fit_all_subsets()
    assert result.n_subsets == 2 ** 3
    # Coefficients of full model must match lstsq without intercept
    full_mask = [True, True, True]
    r_full = next(r for r in result.results if r["col_mask"] == full_mask)
    ref = _ols_ref(X, y)
    assert_allclose(r_full["coef"], ref, atol=1e-8)


def test_k_exceeds_25_raises():
    X = np.random.default_rng(0).standard_normal((100, 26))
    y = np.random.default_rng(0).standard_normal(100)
    with pytest.raises(ValueError, match="25"):
        LinearSensitivity(X, y=y)


def test_sequential_equals_parallel():
    X, y = _make_data(k=4)
    result_seq = LinearSensitivity(X, y=y, n_jobs=1).fit_all_subsets()
    result_par = LinearSensitivity(X, y=y, n_jobs=2).fit_all_subsets()

    seq_by_mask = {tuple(r["col_mask"]): r for r in result_seq.results}
    for r_p in result_par.results:
        r_s = seq_by_mask[tuple(r_p["col_mask"])]
        assert_allclose(r_p.get("rss", 0), r_s.get("rss", 0), rtol=1e-6)
