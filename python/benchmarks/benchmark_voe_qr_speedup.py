#!/usr/bin/env python3
"""Benchmark QR-backed Gaussian VoE against a fresh full-QR baseline."""

from __future__ import annotations

import argparse
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

from regsens.qr_core import compute_se, qr_new_outcome, qr_state_from_cols
from voe import conduct_vibration, conduct_vibration_for_k
from voe.core import VibrationResult, _combination_tuples, _gaussian_bic, _prepare_design

NHANES_FIXTURE = REPO_ROOT / "benchmarks" / "data" / "nhanes_voe_gaussian.csv"
NHANES_ADJUST = [
    "SES_LEVEL",
    "current_past_smoking",
    "education",
    "RIDRETH1",
    "any_cad",
    "any_ht",
    "any_diabetes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("synthetic", "nhanes"), default="synthetic")
    parser.add_argument("--k-min", type=int, default=None)
    parser.add_argument("--k-max", type=int, default=None)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-obs", type=int, default=8000)
    parser.add_argument("--n-adjust", type=int, default=12)
    return parser.parse_args()


def make_synthetic_data(n_obs: int, n_adjust: int, seed: int) -> dict:
    if n_adjust < 1:
        raise ValueError("Synthetic benchmarks require at least one adjustor.")

    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "y": rng.standard_normal(n_obs),
            "x": rng.standard_normal(n_obs),
            "age": rng.standard_normal(n_obs),
        }
    )

    adjust_names = [f"z{i}" for i in range(1, n_adjust + 1)]
    for idx, name in enumerate(adjust_names, start=1):
        if idx % 4 == 0:
            frame[name] = pd.Categorical(rng.choice(["a", "b", "c", "d"], size=n_obs))
        else:
            frame[name] = rng.standard_normal(n_obs)

    signal = 0.7 * frame["x"] - 0.4 * frame["age"]
    for idx, name in enumerate(adjust_names[: min(4, n_adjust)], start=1):
        if isinstance(frame[name].dtype, pd.CategoricalDtype):
            effect_map = {"a": -0.3, "b": 0.0, "c": 0.3, "d": 0.5}
            signal = signal + pd.Series(frame[name]).map(effect_map).to_numpy(dtype=float)
        else:
            signal = signal + (0.2 / idx) * frame[name]
    frame["y"] = signal + rng.standard_normal(n_obs) * 0.5

    return {
        "frame": frame,
        "outcome": "y",
        "exposure": "x",
        "base_covariates": ["age"],
        "adjust_by": adjust_names,
        "n_adjust": n_adjust,
    }


def load_nhanes_data() -> dict:
    if not NHANES_FIXTURE.exists():
        raise FileNotFoundError(
            "Missing NHANES benchmark fixture. "
            "Run `Rscript benchmarks/regenerate_nhanes_fixture.R` first."
        )

    frame = pd.read_csv(NHANES_FIXTURE)
    for name in ["SES_LEVEL", "current_past_smoking", "education", "RIDRETH1"]:
        frame[name] = frame[name].astype("category")

    return {
        "frame": frame,
        "outcome": "LBXVID",
        "exposure": "LBXBCD_log_z",
        "base_covariates": ["RIDAGEYR", "male"],
        "adjust_by": list(NHANES_ADJUST),
        "n_adjust": len(NHANES_ADJUST),
    }


def resolve_config(args: argparse.Namespace) -> dict:
    config = load_nhanes_data() if args.dataset == "nhanes" else make_synthetic_data(
        args.n_obs, args.n_adjust, args.seed
    )

    if args.reps < 1:
        raise ValueError("`--reps` must be at least 1.")

    if args.k_min is None:
        args.k_min = 1 if args.dataset == "nhanes" else min(6, config["n_adjust"])
    if args.k_max is None:
        args.k_max = config["n_adjust"] if args.dataset == "nhanes" else min(6, config["n_adjust"])
    if args.k_min > args.k_max:
        raise ValueError("`--k-min` must be less than or equal to `--k-max`.")

    config["dataset"] = args.dataset
    config["k_min"] = args.k_min
    config["k_max"] = args.k_max
    config["reps"] = args.reps
    config["seed"] = args.seed
    config["n_obs"] = len(config["frame"])
    config["n_models"] = sum(math.comb(config["n_adjust"], k) for k in range(args.k_min, args.k_max + 1))
    config["is_fixed_k"] = args.k_min == args.k_max
    return config


def _run_fresh_for_k_prepared(context, k: int) -> VibrationResult:
    combos = _combination_tuples(len(context.adjust_groups), k)
    vib_rows: list[dict] = []
    bic_rows: list[dict] = []
    exposure_positions = list(context.exposure_col_indices)
    if not exposure_positions:
        raise ValueError("Exposure term did not map to any design columns.")

    for combo_idx, current_terms in enumerate(combos, start=1):
        active_cols = list(context.fixed_cols)
        for term_idx in current_terms:
            active_cols.extend(context.adjust_groups[term_idx]["col_indices"])

        state = qr_state_from_cols(context.X, active_cols, tol=1e-10)
        fit = qr_new_outcome(state, context.y)
        n_params = state.p
        df_res = context.n_obs - n_params
        if n_params > 0 and df_res > 0:
            sigma = float(np.sqrt(fit["rss"] / df_res))
            se = compute_se(state.R, sigma)
            t_values = fit["beta"] / np.where(se == 0, np.inf, se)
            pvalues = 2 * stats.t.sf(np.abs(t_values), df=df_res)
        else:
            se = np.full(n_params, np.nan)
            t_values = np.full(n_params, np.nan)
            pvalues = np.full(n_params, np.nan)

        bic_rows.append(
            {
                "edf": n_params,
                "bic": _gaussian_bic(fit["rss"], context.n_obs, n_params),
                "combination_index": combo_idx,
            }
        )
        for factor_level, row_idx in enumerate(exposure_positions, start=1):
            vib_rows.append(
                {
                    "estimate": fit["beta"][row_idx],
                    "se": se[row_idx] if n_params > 0 else np.nan,
                    "z": t_values[row_idx] if n_params > 0 else np.nan,
                    "pvalue": pvalues[row_idx] if n_params > 0 else np.nan,
                    "combination_index": combo_idx,
                    "factor_level": factor_level,
                }
            )

    return VibrationResult(
        vib_frame=pd.DataFrame(vib_rows),
        bic_frame=pd.DataFrame(bic_rows),
        combinations=[tuple(context.adjust_names[idx] for idx in combo) for combo in combos],
        adjust=list(context.adjust_names),
        family="gaussian",
        exposure=context.exposure_name,
        outcomes=list(context.outcome_names),
        k=k,
    )


def run_fresh_once(config: dict) -> VibrationResult:
    context = _prepare_design(
        config["frame"],
        outcome=config["outcome"],
        outcomes=None,
        exposure=config["exposure"],
        adjust_by=config["adjust_by"],
        base_covariates=config["base_covariates"],
        drop_first=True,
        family="gaussian",
    )

    if config["is_fixed_k"]:
        return _run_fresh_for_k_prepared(context, config["k_min"])

    fixed_results = [
        _run_fresh_for_k_prepared(context, k)
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


def run_qr_once(config: dict) -> VibrationResult:
    kwargs = dict(
        data=config["frame"],
        outcome=config["outcome"],
        exposure=config["exposure"],
        adjust_by=config["adjust_by"],
        base_covariates=config["base_covariates"],
        family="gaussian",
    )
    if config["is_fixed_k"]:
        return conduct_vibration_for_k(k=config["k_min"], **kwargs)
    return conduct_vibration(k_min=config["k_min"], k_max=config["k_max"], **kwargs)


def result_frames(result: VibrationResult, is_fixed_k: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    vib = result.vib_frame.copy()
    bic = result.bic_frame.copy()
    if is_fixed_k:
        sort_vib = ["combination_index", "factor_level"]
        sort_bic = ["combination_index"]
    else:
        sort_vib = ["k", "combination_index", "factor_level"]
        sort_bic = ["k", "combination_index"]
    vib = vib.sort_values(sort_vib).reset_index(drop=True)
    bic = bic.sort_values(sort_bic).reset_index(drop=True)
    return vib, bic


def validate_results(
    qr_result: VibrationResult,
    fresh_result: VibrationResult,
    is_fixed_k: bool,
    tol_estimate: float = 1e-8,
    tol_bic: float = 1e-8,
) -> dict:
    qr_vib, qr_bic = result_frames(qr_result, is_fixed_k)
    fresh_vib, fresh_bic = result_frames(fresh_result, is_fixed_k)

    if list(qr_vib.columns) != list(fresh_vib.columns):
        raise ValueError("QR and fresh baseline vibration columns differ.")
    if list(qr_bic.columns) != list(fresh_bic.columns):
        raise ValueError("QR and fresh baseline BIC columns differ.")
    if len(qr_vib) != len(fresh_vib) or len(qr_bic) != len(fresh_bic):
        raise ValueError("QR and fresh baseline returned different row counts.")

    for name in [col for col in ("k", "combination_index", "factor_level") if col in qr_vib.columns]:
        if not qr_vib[name].equals(fresh_vib[name]):
            raise ValueError(f"QR and fresh baseline differ in vibration identifier column `{name}`.")
    for name in [col for col in ("k", "combination_index") if col in qr_bic.columns]:
        if not qr_bic[name].equals(fresh_bic[name]):
            raise ValueError(f"QR and fresh baseline differ in BIC identifier column `{name}`.")

    estimate_diff = float(np.max(np.abs(qr_vib["estimate"].to_numpy() - fresh_vib["estimate"].to_numpy())))
    bic_diff = float(np.max(np.abs(qr_bic["bic"].to_numpy() - fresh_bic["bic"].to_numpy())))

    if not np.isfinite(estimate_diff) or estimate_diff > tol_estimate:
        raise ValueError(
            f"Validation failed: max estimate diff {estimate_diff:.3e} exceeds tolerance {tol_estimate:.3e}"
        )
    if not np.isfinite(bic_diff) or bic_diff > tol_bic:
        raise ValueError(
            f"Validation failed: max BIC diff {bic_diff:.3e} exceeds tolerance {tol_bic:.3e}"
        )

    return {"max_estimate_diff": estimate_diff, "max_bic_diff": bic_diff}


def time_many(fun, reps: int) -> list[float]:
    times = []
    for _ in range(reps):
        started = time.perf_counter()
        fun()
        times.append(time.perf_counter() - started)
    return times


def print_summary(config: dict, qr_times: list[float], fresh_times: list[float], validation: dict) -> None:
    qr_median = float(np.median(qr_times))
    fresh_median = float(np.median(fresh_times))
    k_label = str(config["k_min"]) if config["is_fixed_k"] else f"{config['k_min']}:{config['k_max']}"

    print(f"Dataset           : {config['dataset']}")
    print(f"Rows              : {config['n_obs']}")
    print(f"Adjustors         : {config['n_adjust']}")
    print(f"Model count       : {config['n_models']}")
    print(f"k                 : {k_label}")
    print(f"QR median seconds : {qr_median:.6f}")
    print(f"Fresh QR seconds  : {fresh_median:.6f}")
    print(f"Speedup           : {fresh_median / qr_median:.2f}x")
    print(f"Max estimate diff : {validation['max_estimate_diff']:.3e}")
    print(f"Max BIC diff      : {validation['max_bic_diff']:.3e}")


def main() -> int:
    config = resolve_config(parse_args())
    qr_validation = run_qr_once(config)
    fresh_validation = run_fresh_once(config)
    validation = validate_results(qr_validation, fresh_validation, config["is_fixed_k"])

    run_qr_once(config)
    run_fresh_once(config)

    qr_times = time_many(lambda: run_qr_once(config), config["reps"])
    fresh_times = time_many(lambda: run_fresh_once(config), config["reps"])
    print_summary(config, qr_times, fresh_times, validation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
