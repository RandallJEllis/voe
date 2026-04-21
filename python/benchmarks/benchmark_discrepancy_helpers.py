#!/usr/bin/env python3
"""Shared helpers for cross-language discrepancy benchmarks."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_PATH = Path(__file__).resolve()
PYTHON_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = PYTHON_ROOT.parent
SRC_ROOT = PYTHON_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from regsens.qr_core import compute_se, qr_batch_outcomes, qr_new_outcome, qr_state_from_cols
from voe import conduct_vibration, conduct_vibration_for_k
from voe.core import VibrationResult, _combination_tuples, _gaussian_bic, _prepare_design


def time_many(fun, reps: int) -> np.ndarray:
    times = np.empty(reps, dtype=float)
    for idx in range(reps):
        start = time.perf_counter()
        fun()
        times[idx] = time.perf_counter() - start
    return times


def _t_pvalues(t_values: np.ndarray, df_res: int) -> np.ndarray:
    return 2 * stats.t.sf(np.abs(t_values), df=df_res)


def make_many_outcomes_data(n_obs: int, n_covariates: int, n_outcomes: int, seed: int) -> dict:
    if n_obs < 2:
        raise ValueError("`n_obs` must be at least 2.")
    if n_covariates < 1:
        raise ValueError("`n_covariates` must be at least 1.")
    if n_outcomes < 1:
        raise ValueError("`n_outcomes` must be at least 1.")

    rng = np.random.default_rng(seed)
    columns = {"x": rng.standard_normal(n_obs)}
    cov_names = [f"z{idx}" for idx in range(1, n_covariates + 1)]
    for name in cov_names:
        columns[name] = rng.standard_normal(n_obs)

    frame = pd.DataFrame(columns)

    base_effect = np.linspace(0.25, 0.05, num=min(8, n_covariates), dtype=float)
    outcome_names = [f"y{idx}" for idx in range(1, n_outcomes + 1)]
    cov_matrix = frame[cov_names].to_numpy(dtype=float)
    outcome_columns = {}
    for idx, outcome_name in enumerate(outcome_names, start=1):
        weights = np.zeros(n_covariates, dtype=float)
        if base_effect.size:
            sign_scale = -1.0 if idx % 2 == 0 else 1.0
            weights[: base_effect.size] = sign_scale * base_effect * (1.0 + idx / (4.0 * n_outcomes))
        y_signal = (0.6 + 0.15 * math.sin(idx / 5.0)) * frame["x"].to_numpy(dtype=float)
        y_signal = y_signal + cov_matrix @ weights
        outcome_columns[outcome_name] = y_signal + rng.standard_normal(n_obs) * 0.8

    frame = pd.concat([frame, pd.DataFrame(outcome_columns)], axis=1)

    return {
        "frame": frame,
        "predictor_names": ["x", *cov_names],
        "outcome_names": outcome_names,
    }


def run_many_outcomes_qr(frame: pd.DataFrame, predictor_names: list[str], outcome_names: list[str]) -> pd.DataFrame:
    X = np.column_stack(
        [
            np.ones(len(frame), dtype=float),
            frame[predictor_names].to_numpy(dtype=float),
        ]
    )
    Y = frame[outcome_names].to_numpy(dtype=float)
    terms = ["(Intercept)", *predictor_names]

    state = qr_state_from_cols(X, list(range(X.shape[1])), tol=1e-10)
    batch = qr_batch_outcomes(state, Y)
    df_res = state.m - state.p
    sigma = np.sqrt(batch["rss"] / df_res)
    se = compute_se(state.R, sigma)
    t_values = batch["beta"] / np.where(se == 0, np.inf, se)
    pvalues = _t_pvalues(t_values, df_res)

    rows = []
    for outcome_idx, outcome_name in enumerate(outcome_names):
        for term_idx, term_name in enumerate(terms):
            rows.append(
                {
                    "outcome": outcome_name,
                    "term": term_name,
                    "Estimate": float(batch["beta"][term_idx, outcome_idx]),
                    "Std.Error": float(se[term_idx, outcome_idx]),
                    "t.value": float(t_values[term_idx, outcome_idx]),
                    "Pr": float(pvalues[term_idx, outcome_idx]),
                }
            )
    return pd.DataFrame(rows)


def run_many_outcomes_native(
    frame: pd.DataFrame, predictor_names: list[str], outcome_names: list[str]
) -> pd.DataFrame:
    rows = []
    for outcome_name in outcome_names:
        X = np.column_stack(
            [
                np.ones(len(frame), dtype=float),
                frame[predictor_names].to_numpy(dtype=float),
            ]
        )
        y = frame[outcome_name].to_numpy(dtype=float)
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ beta
        rss = float(np.dot(residuals, residuals))
        n_obs, n_params = X.shape
        df_res = n_obs - n_params
        sigma2 = rss / df_res
        xtx_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(xtx_inv) * sigma2)
        t_values = beta / np.where(se == 0, np.inf, se)
        pvalues = _t_pvalues(t_values, df_res)

        for term_name, estimate, se_val, t_val, p_val in zip(
            ["(Intercept)", *predictor_names], beta, se, t_values, pvalues
        ):
            rows.append(
                {
                    "outcome": outcome_name,
                    "term": term_name,
                    "Estimate": float(estimate),
                    "Std.Error": float(se_val),
                    "t.value": float(t_val),
                    "Pr": float(p_val),
                }
            )
    return pd.DataFrame(rows)


def validate_many_outcomes(qr_result: pd.DataFrame, native_result: pd.DataFrame, tol: float = 1e-8) -> dict:
    order_cols = ["outcome", "term"]
    qr_sorted = qr_result.sort_values(order_cols).reset_index(drop=True)
    native_sorted = native_result.sort_values(order_cols).reset_index(drop=True)

    if not qr_sorted[order_cols].equals(native_sorted[order_cols]):
        raise ValueError("QR and native many-outcome outputs are not aligned.")

    numeric_cols = ["Estimate", "Std.Error", "t.value", "Pr"]
    max_diff = float(np.max(np.abs(qr_sorted[numeric_cols].to_numpy() - native_sorted[numeric_cols].to_numpy())))
    if not np.isfinite(max_diff) or max_diff > tol:
        raise ValueError(
            f"Validation failed: max many-outcome diff {max_diff:.3e} exceeds tolerance {tol:.3e}"
        )
    return {"max_diff": max_diff}


def print_many_outcomes_summary(config: dict, qr_times: np.ndarray, native_times: np.ndarray, validation: dict) -> None:
    qr_median = float(np.median(qr_times))
    native_median = float(np.median(native_times))
    print(f"Rows                  : {config['n_obs']}")
    print(f"Covariates            : {config['n_covariates']}")
    print(f"Design columns        : {config['n_covariates'] + 2}")
    print(f"Outcomes              : {config['n_outcomes']}")
    print(f"QR median seconds     : {qr_median:.6f}")
    print(f"Native median seconds : {native_median:.6f}")
    print(f"Speedup               : {native_median / qr_median:.2f}x")
    print(f"Max diff              : {validation['max_diff']:.3e}")


def make_voe_multioutcome_data(n_obs: int, n_adjust: int, n_outcomes: int, seed: int) -> dict:
    if n_obs < 2:
        raise ValueError("`n_obs` must be at least 2.")
    if n_adjust < 1:
        raise ValueError("`n_adjust` must be at least 1.")
    if n_outcomes < 1:
        raise ValueError("`n_outcomes` must be at least 1.")

    rng = np.random.default_rng(seed)
    columns = {
        "x": rng.standard_normal(n_obs),
        "age": rng.standard_normal(n_obs),
    }

    adjust_names = [f"z{idx}" for idx in range(1, n_adjust + 1)]
    for idx, name in enumerate(adjust_names, start=1):
        if idx % 4 == 0:
            columns[name] = pd.Categorical(rng.choice(["a", "b", "c", "d"], size=n_obs))
        else:
            columns[name] = rng.standard_normal(n_obs)

    frame = pd.DataFrame(columns)

    outcome_names = [f"y{idx}" for idx in range(1, n_outcomes + 1)]
    outcome_columns = {}
    for idx, outcome_name in enumerate(outcome_names, start=1):
        y_signal = (0.7 + 0.1 * math.cos(idx / 4.0)) * frame["x"].to_numpy(dtype=float)
        y_signal = y_signal - (0.35 + 0.05 * math.sin(idx / 6.0)) * frame["age"].to_numpy(dtype=float)
        for adjust_idx, name in enumerate(adjust_names[: min(4, n_adjust)], start=1):
            column = frame[name]
            if isinstance(column.dtype, pd.CategoricalDtype):
                effect_map = {"a": -0.3, "b": 0.0, "c": 0.25, "d": 0.5}
                effect = pd.Series(column).map(effect_map).to_numpy(dtype=float)
                y_signal = y_signal + (0.8 + idx / (3.0 * n_outcomes)) * effect
            else:
                y_signal = y_signal + (0.15 / adjust_idx) * (1.0 + idx / (5.0 * n_outcomes)) * column.to_numpy(dtype=float)
        outcome_columns[outcome_name] = y_signal + rng.standard_normal(n_obs) * 0.6

    frame = pd.concat([frame, pd.DataFrame(outcome_columns)], axis=1)

    return {
        "frame": frame,
        "outcome_names": outcome_names,
        "base_covariates": ["age"],
        "adjust_names": adjust_names,
        "exposure": "x",
    }


def run_voe_multioutcome_qr(config: dict) -> VibrationResult:
    kwargs = dict(
        data=config["frame"],
        outcomes=config["outcome_names"],
        exposure=config["exposure"],
        adjust_by=config["adjust_names"],
        base_covariates=config["base_covariates"],
        family="gaussian",
    )
    if config["is_fixed_k"]:
        return conduct_vibration_for_k(k=config["k_min"], **kwargs)
    return conduct_vibration(k_min=config["k_min"], k_max=config["k_max"], **kwargs)


def _native_vibration_for_subset(
    context, active_cols: list[int], exposure_positions: list[int], combo_idx: int, outcome_name: str, y: np.ndarray
) -> tuple[list[dict], dict]:
    X = context.X[:, active_cols]
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    rss = float(np.dot(residuals, residuals))
    n_obs, n_params = X.shape
    df_res = n_obs - n_params

    if n_params > 0 and df_res > 0:
        sigma2 = rss / df_res
        xtx_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(xtx_inv) * sigma2)
        t_values = beta / np.where(se == 0, np.inf, se)
        pvalues = _t_pvalues(t_values, df_res)
    else:
        se = np.full(n_params, np.nan)
        t_values = np.full(n_params, np.nan)
        pvalues = np.full(n_params, np.nan)

    vib_rows = []
    for factor_level, row_idx in enumerate(exposure_positions, start=1):
        vib_rows.append(
            {
                "estimate": float(beta[row_idx]),
                "se": float(se[row_idx]) if n_params > 0 else math.nan,
                "z": float(t_values[row_idx]) if n_params > 0 else math.nan,
                "pvalue": float(pvalues[row_idx]) if n_params > 0 else math.nan,
                "combination_index": combo_idx,
                "factor_level": factor_level,
                "outcome": outcome_name,
            }
        )

    bic_row = {
        "edf": n_params,
        "bic": _gaussian_bic(rss, n_obs, n_params),
        "combination_index": combo_idx,
        "outcome": outcome_name,
    }
    return vib_rows, bic_row


def _run_voe_multioutcome_native_for_k_prepared(context, k: int) -> VibrationResult:
    combos = _combination_tuples(len(context.adjust_groups), k)
    named_combos = [tuple(context.adjust_names[idx] for idx in combo) for combo in combos]
    vib_rows: list[dict] = []
    bic_rows: list[dict] = []

    base_active = list(context.fixed_cols)
    exposure_positions = [base_active.index(col_idx) for col_idx in context.exposure_col_indices]

    for combo_idx, current_terms in enumerate(combos, start=1):
        active_cols = list(base_active)
        for term_idx in current_terms:
            active_cols.extend(context.adjust_groups[term_idx]["col_indices"])
        active_exposure_positions = [active_cols.index(col_idx) for col_idx in context.exposure_col_indices]

        for outcome_idx, outcome_name in enumerate(context.outcome_names):
            y = context.Y[:, outcome_idx]
            subset_vib_rows, bic_row = _native_vibration_for_subset(
                context,
                active_cols,
                active_exposure_positions,
                combo_idx,
                outcome_name,
                y,
            )
            vib_rows.extend(subset_vib_rows)
            bic_rows.append(bic_row)

    return VibrationResult(
        vib_frame=pd.DataFrame(vib_rows),
        bic_frame=pd.DataFrame(bic_rows),
        combinations=named_combos,
        adjust=list(context.adjust_names),
        family="gaussian",
        exposure=context.exposure_name,
        outcomes=list(context.outcome_names),
        k=k,
    )


def run_voe_multioutcome_native(config: dict) -> VibrationResult:
    context = _prepare_design(
        config["frame"],
        outcome=None,
        outcomes=config["outcome_names"],
        exposure=config["exposure"],
        adjust_by=config["adjust_names"],
        base_covariates=config["base_covariates"],
        drop_first=True,
        family="gaussian",
    )

    if config["is_fixed_k"]:
        return _run_voe_multioutcome_native_for_k_prepared(context, config["k_min"])

    fixed_results = [
        _run_voe_multioutcome_native_for_k_prepared(context, k)
        for k in range(config["k_min"], config["k_max"] + 1)
    ]
    vib_frames = []
    bic_frames = []
    for result in fixed_results:
        vib = result.vib_frame.copy()
        vib["k"] = result.k
        bic = result.bic_frame.copy()
        bic["k"] = result.k
        vib_frames.append(vib)
        bic_frames.append(bic)

    return VibrationResult(
        vib_frame=pd.concat(vib_frames, ignore_index=True) if vib_frames else pd.DataFrame(),
        bic_frame=pd.concat(bic_frames, ignore_index=True) if bic_frames else pd.DataFrame(),
        combinations=[result.combinations for result in fixed_results],
        adjust=list(context.adjust_names),
        family="gaussian",
        exposure=context.exposure_name,
        outcomes=list(context.outcome_names),
        k=None,
    )


def _normalize_voe_frames(result: VibrationResult, is_fixed_k: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    vib = result.vib_frame.copy()
    bic = result.bic_frame.copy()

    if is_fixed_k:
        vib_cols = ["estimate", "se", "z", "pvalue", "combination_index", "factor_level", "outcome"]
        bic_cols = ["edf", "bic", "combination_index", "outcome"]
        sort_vib = ["outcome", "combination_index", "factor_level"]
        sort_bic = ["outcome", "combination_index"]
    else:
        vib_cols = ["estimate", "se", "z", "pvalue", "combination_index", "factor_level", "k", "outcome"]
        bic_cols = ["edf", "bic", "combination_index", "k", "outcome"]
        sort_vib = ["outcome", "k", "combination_index", "factor_level"]
        sort_bic = ["outcome", "k", "combination_index"]

    vib = vib[vib_cols].sort_values(sort_vib).reset_index(drop=True)
    bic = bic[bic_cols].sort_values(sort_bic).reset_index(drop=True)
    return vib, bic


def validate_voe_multioutcome(
    qr_result: VibrationResult,
    native_result: VibrationResult,
    is_fixed_k: bool,
    tol_estimate: float = 1e-8,
    tol_bic: float = 1e-8,
) -> dict:
    qr_vib, qr_bic = _normalize_voe_frames(qr_result, is_fixed_k)
    native_vib, native_bic = _normalize_voe_frames(native_result, is_fixed_k)

    vib_id_cols = [name for name in ["outcome", "k", "combination_index", "factor_level"] if name in qr_vib.columns]
    bic_id_cols = [name for name in ["outcome", "k", "combination_index"] if name in qr_bic.columns]

    for name in vib_id_cols:
        qr_col = qr_vib[name].to_numpy()
        native_col = native_vib[name].to_numpy()
        if qr_col.dtype.kind in {"i", "f"} or native_col.dtype.kind in {"i", "f"}:
            same = np.array_equal(qr_col.astype(float), native_col.astype(float))
        else:
            same = np.array_equal(qr_col.astype(str), native_col.astype(str))
        if not same:
            raise ValueError(f"QR and native multi-outcome vibration identifier {name!r} differs.")

    for name in bic_id_cols:
        qr_col = qr_bic[name].to_numpy()
        native_col = native_bic[name].to_numpy()
        if qr_col.dtype.kind in {"i", "f"} or native_col.dtype.kind in {"i", "f"}:
            same = np.array_equal(qr_col.astype(float), native_col.astype(float))
        else:
            same = np.array_equal(qr_col.astype(str), native_col.astype(str))
        if not same:
            raise ValueError(f"QR and native multi-outcome BIC identifier {name!r} differs.")

    estimate_diff = float(np.max(np.abs(qr_vib["estimate"].to_numpy() - native_vib["estimate"].to_numpy())))
    bic_diff = float(np.max(np.abs(qr_bic["bic"].to_numpy() - native_bic["bic"].to_numpy())))
    if not np.isfinite(estimate_diff) or estimate_diff > tol_estimate:
        raise ValueError(
            f"Validation failed: max multi-outcome estimate diff {estimate_diff:.3e} exceeds tolerance {tol_estimate:.3e}"
        )
    if not np.isfinite(bic_diff) or bic_diff > tol_bic:
        raise ValueError(
            f"Validation failed: max multi-outcome BIC diff {bic_diff:.3e} exceeds tolerance {tol_bic:.3e}"
        )

    return {
        "max_estimate_diff": estimate_diff,
        "max_bic_diff": bic_diff,
    }


def print_voe_multioutcome_summary(
    config: dict, qr_times: np.ndarray, native_times: np.ndarray, validation: dict
) -> None:
    qr_median = float(np.median(qr_times))
    native_median = float(np.median(native_times))
    if config["is_fixed_k"]:
        k_label = str(config["k_min"])
    else:
        k_label = f"{config['k_min']}:{config['k_max']}"

    print(f"Rows                  : {len(config['frame'])}")
    print(f"Adjustors             : {len(config['adjust_names'])}")
    print(f"Outcomes              : {len(config['outcome_names'])}")
    print(f"Model count           : {config['n_models']}")
    print(f"k                     : {k_label}")
    print(f"QR median seconds     : {qr_median:.6f}")
    print(f"Native median seconds : {native_median:.6f}")
    print(f"Speedup               : {native_median / qr_median:.2f}x")
    print(f"Max estimate diff     : {validation['max_estimate_diff']:.3e}")
    print(f"Max BIC diff          : {validation['max_bic_diff']:.3e}")
