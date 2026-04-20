"""Tests for the Python VoE wrapper."""

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from scipy import stats

from voe import conduct_vibration, conduct_vibration_for_k


def _make_data(seed=101, n=80):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    age = rng.standard_normal(n)
    z_num = rng.standard_normal(n)
    z_cat = pd.Categorical(rng.choice(["a", "b", "c"], size=n))
    cat_effect = pd.Series(z_cat).map({"a": -0.3, "b": 0.1, "c": 0.4}).to_numpy(dtype=float)

    y = 0.5 + 0.8 * x - 0.4 * age + 0.6 * z_num + cat_effect + rng.standard_normal(n) * 0.3
    y2 = -0.2 + 0.3 * x + 0.2 * age - 0.5 * z_num + 0.5 * cat_effect + rng.standard_normal(n) * 0.4

    return pd.DataFrame(
        {
            "y": y,
            "y2": y2,
            "x": x,
            "age": age,
            "z_num": z_num,
            "z_cat": z_cat,
        }
    )


def _encode_group(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    series = frame[column]
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return pd.DataFrame({column: series.astype(float)}, index=frame.index)
    return pd.get_dummies(series, prefix=column, drop_first=True, dtype=float)


def _ols_reference(frame: pd.DataFrame, include_adjust: str):
    blocks = [
        pd.DataFrame({"(Intercept)": np.ones(len(frame), dtype=float)}, index=frame.index),
        _encode_group(frame, "x"),
        _encode_group(frame, "age"),
        _encode_group(frame, include_adjust),
    ]
    X = pd.concat(blocks, axis=1).to_numpy(dtype=float)
    y = frame["y"].to_numpy(dtype=float)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    rss = float(np.sum(residuals ** 2))
    n, p = X.shape
    sigma = np.sqrt(rss / (n - p))
    XtX_inv = np.linalg.inv(X.T @ X)
    se = sigma * np.sqrt(np.diag(XtX_inv))
    t_values = beta / se
    pvalues = 2 * stats.t.sf(np.abs(t_values), df=n - p)
    bic = n * np.log(rss / n) + np.log(n) * p
    return {
        "beta_x": beta[1],
        "se_x": se[1],
        "pvalue_x": pvalues[1],
        "bic": bic,
    }


def test_conduct_vibration_for_k_matches_groupwise_ols():
    frame = _make_data()
    result = conduct_vibration_for_k(
        frame,
        outcome="y",
        exposure="x",
        base_covariates=["age"],
        adjust_by=["z_num", "z_cat"],
        k=1,
    )

    assert len(result.combinations) == 2
    assert set(result.vib_frame.columns) == {
        "estimate",
        "se",
        "z",
        "pvalue",
        "combination_index",
        "factor_level",
    }

    refs = {
        ("z_num",): _ols_reference(frame, "z_num"),
        ("z_cat",): _ols_reference(frame, "z_cat"),
    }
    for combo_idx, combo in enumerate(result.combinations, start=1):
        vib_row = result.vib_frame[result.vib_frame["combination_index"] == combo_idx].iloc[0]
        bic_row = result.bic_frame[result.bic_frame["combination_index"] == combo_idx].iloc[0]
        ref = refs[combo]
        assert_allclose(vib_row["estimate"], ref["beta_x"], atol=1e-8)
        assert_allclose(vib_row["se"], ref["se_x"], atol=1e-8)
        assert_allclose(vib_row["pvalue"], ref["pvalue_x"], atol=1e-8)
        assert_allclose(bic_row["bic"], ref["bic"], atol=1e-8)


def test_conduct_vibration_multi_outcome_matches_separate_runs():
    frame = _make_data(seed=202)
    multi = conduct_vibration(
        frame,
        outcomes=["y", "y2"],
        exposure="x",
        base_covariates=["age"],
        adjust_by=["z_num", "z_cat"],
    )
    single_y = conduct_vibration(
        frame,
        outcome="y",
        exposure="x",
        base_covariates=["age"],
        adjust_by=["z_num", "z_cat"],
    )
    single_y2 = conduct_vibration(
        frame,
        outcome="y2",
        exposure="x",
        base_covariates=["age"],
        adjust_by=["z_num", "z_cat"],
    )

    expected_vib = pd.concat(
        [single_y.vib_frame.assign(outcome="y"), single_y2.vib_frame.assign(outcome="y2")],
        ignore_index=True,
    )
    expected_bic = pd.concat(
        [single_y.bic_frame.assign(outcome="y"), single_y2.bic_frame.assign(outcome="y2")],
        ignore_index=True,
    )

    sort_vib_cols = list(multi.vib_frame.columns)
    sort_bic_cols = list(multi.bic_frame.columns)

    multi_vib = multi.vib_frame.sort_values(
        ["outcome", "k", "combination_index", "factor_level"]
    ).reset_index(drop=True)
    expected_vib = expected_vib[sort_vib_cols].sort_values(
        ["outcome", "k", "combination_index", "factor_level"]
    ).reset_index(drop=True)
    multi_bic = multi.bic_frame.sort_values(
        ["outcome", "k", "combination_index"]
    ).reset_index(drop=True)
    expected_bic = expected_bic[sort_bic_cols].sort_values(
        ["outcome", "k", "combination_index"]
    ).reset_index(drop=True)

    assert_allclose(multi_vib["estimate"], expected_vib["estimate"], atol=1e-8)
    assert_allclose(multi_vib["se"], expected_vib["se"], atol=1e-8)
    assert_allclose(multi_vib["pvalue"], expected_vib["pvalue"], atol=1e-8)
    assert_allclose(multi_bic["bic"], expected_bic["bic"], atol=1e-8)
    assert len(multi.bic_frame) == 2 * (2 + 1)
