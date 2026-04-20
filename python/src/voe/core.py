"""Python VoE API backed by the QR core."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
from scipy import stats

from regsens.qr_core import (
    compute_se,
    qr_add_column,
    qr_batch_outcomes,
    qr_new_outcome,
    qr_remove_column,
    qr_state_from_cols,
)


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised in packaging, not tests
        raise ImportError(
            "The Python VoE API requires pandas. Install with `pip install pandas`."
        ) from exc
    return pd


@dataclass
class VibrationResult:
    """Container for VoE results."""

    vib_frame: object
    bic_frame: object
    combinations: list[list[tuple[str, ...]]] | list[tuple[str, ...]]
    adjust: list[str]
    family: str
    exposure: str
    outcomes: list[str]
    k: int | None = None

    @property
    def vibration(self):
        return self.vib_frame

    @property
    def bic(self):
        return self.bic_frame


@dataclass
class _DesignContext:
    """Shared design-matrix context across VoE calls."""

    frame: object
    exposure_name: str
    X: np.ndarray
    y: np.ndarray | None
    Y: np.ndarray | None
    n_obs: int
    fixed_cols: list[int]
    fixed_coef_names: list[str]
    exposure_col_indices: list[int]
    exposure_coef_names: list[str]
    adjust_groups: list[dict]
    adjust_names: list[str]
    outcome_names: list[str]
    is_multi_outcome: bool


def _coerce_outcomes(outcome: str | None, outcomes: Sequence[str] | None) -> list[str]:
    if (outcome is None) == (outcomes is None):
        raise ValueError("Provide exactly one of `outcome` or `outcomes`.")
    if outcome is not None:
        return [outcome]
    return list(outcomes)


def _encode_group(frame, column: str, drop_first: bool):
    pd = _require_pandas()
    series = frame[column]
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        encoded = pd.DataFrame({column: series.astype(float)}, index=frame.index)
    else:
        encoded = pd.get_dummies(series, prefix=column, drop_first=drop_first, dtype=float)
    if encoded.shape[1] == 0:
        raise ValueError(
            f"Column {column!r} did not produce any design columns after encoding."
        )
    return encoded


def _prepare_design(
    data,
    *,
    outcome: str | None,
    outcomes: Sequence[str] | None,
    exposure: str,
    adjust_by: Sequence[str],
    base_covariates: Sequence[str] | None,
    drop_first: bool,
    family: str,
) -> _DesignContext:
    pd = _require_pandas()
    if family != "gaussian":
        raise NotImplementedError("The Python VoE API currently supports `family='gaussian'` only.")

    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame.")

    outcome_names = _coerce_outcomes(outcome, outcomes)
    adjust_names = list(adjust_by)
    base_names = list(base_covariates or [])

    seen = set()
    for label in [exposure, *base_names, *adjust_names]:
        if label in seen:
            raise ValueError(f"Duplicate predictor specification detected for {label!r}.")
        seen.add(label)

    required = outcome_names + [exposure] + base_names + adjust_names
    missing = [name for name in required if name not in data.columns]
    if missing:
        raise KeyError(f"Missing columns in `data`: {missing}")

    frame = data.loc[:, required].dropna().copy()
    if frame.empty:
        raise ValueError("No complete-case rows remain after dropping missing values.")

    blocks = []
    fixed_cols: list[int] = []
    fixed_coef_names: list[str] = []
    adjust_groups: list[dict] = []
    column_offset = 0

    intercept_block = pd.DataFrame({"(Intercept)": np.ones(len(frame), dtype=float)}, index=frame.index)
    blocks.append(intercept_block)
    fixed_cols.append(column_offset)
    fixed_coef_names.append("(Intercept)")
    column_offset += 1

    exposure_block = _encode_group(frame, exposure, drop_first=drop_first)
    exposure_coef_names = list(exposure_block.columns)
    blocks.append(exposure_block)
    exposure_cols = list(range(column_offset, column_offset + exposure_block.shape[1]))
    fixed_cols.extend(exposure_cols)
    fixed_coef_names.extend(exposure_coef_names)
    column_offset += exposure_block.shape[1]

    for column in base_names:
        block = _encode_group(frame, column, drop_first=drop_first)
        blocks.append(block)
        block_cols = list(range(column_offset, column_offset + block.shape[1]))
        fixed_cols.extend(block_cols)
        fixed_coef_names.extend(list(block.columns))
        column_offset += block.shape[1]

    for column in adjust_names:
        block = _encode_group(frame, column, drop_first=drop_first)
        blocks.append(block)
        block_cols = list(range(column_offset, column_offset + block.shape[1]))
        adjust_groups.append(
            {
                "name": column,
                "col_indices": block_cols,
                "coef_names": list(block.columns),
            }
        )
        column_offset += block.shape[1]

    design = pd.concat(blocks, axis=1)
    X = design.to_numpy(dtype=float)

    if len(outcome_names) == 1:
        y = frame[outcome_names[0]].to_numpy(dtype=float)
        Y = None
    else:
        y = None
        Y = frame[outcome_names].to_numpy(dtype=float)

    return _DesignContext(
        frame=frame,
        exposure_name=exposure,
        X=X,
        y=y,
        Y=Y,
        n_obs=X.shape[0],
        fixed_cols=fixed_cols,
        fixed_coef_names=fixed_coef_names,
        exposure_col_indices=exposure_cols,
        exposure_coef_names=exposure_coef_names,
        adjust_groups=adjust_groups,
        adjust_names=adjust_names,
        outcome_names=outcome_names,
        is_multi_outcome=Y is not None,
    )


def _combination_tuples(n_adjust: int, k: int) -> list[tuple[int, ...]]:
    if k < 0 or k > n_adjust:
        raise ValueError(f"`k` must be between 0 and {n_adjust}.")
    if k == 0:
        return [tuple()]
    return list(combinations(range(n_adjust), k))


def _update_qr_state(context: _DesignContext, state, coef_names, previous_terms, current_terms):
    remove_terms = [term for term in previous_terms if term not in current_terms]
    add_terms = [term for term in current_terms if term not in previous_terms]

    if remove_terms:
        remove_positions = []
        for term_idx in remove_terms:
            for col_idx in context.adjust_groups[term_idx]["col_indices"]:
                remove_positions.append(state.col_indices.index(col_idx))
        for pos in sorted(remove_positions, reverse=True):
            state = qr_remove_column(state, pos, inplace=True)
            del coef_names[pos]

    for term_idx in add_terms:
        term_group = context.adjust_groups[term_idx]
        for col_idx, coef_name in zip(term_group["col_indices"], term_group["coef_names"]):
            state = qr_add_column(state, context.X[:, col_idx], col_idx, inplace=True)
            coef_names.append(coef_name)

    return state, coef_names


def _gaussian_bic(rss: float, n_obs: int, edf: int) -> float:
    rss = max(float(rss), 1e-300)
    return n_obs * np.log(rss / n_obs) + np.log(n_obs) * edf


def _conduct_vibration_for_k_prepared(context: _DesignContext, k: int):
    pd = _require_pandas()
    combos = _combination_tuples(len(context.adjust_groups), k)
    state = qr_state_from_cols(context.X, list(context.fixed_cols))
    coef_names = list(context.fixed_coef_names)
    previous_terms: tuple[int, ...] = tuple()
    vib_rows: list[dict] = []
    bic_rows: list[dict] = []

    exposure_positions = list(context.exposure_col_indices)
    if not exposure_positions:
        raise ValueError("Exposure term did not map to any design columns.")

    for combo_idx, current_terms in enumerate(combos, start=1):
        state, coef_names = _update_qr_state(
            context, state, coef_names, previous_terms, current_terms
        )
        previous_terms = current_terms
        n_params = state.p
        df_res = context.n_obs - n_params

        if context.is_multi_outcome:
            batch = qr_batch_outcomes(state, context.Y)
            if n_params > 0 and df_res > 0:
                sigma = np.sqrt(batch["rss"] / df_res)
                se = compute_se(state.R, sigma)
                t_values = batch["beta"] / np.where(se == 0, np.inf, se)
                pvalues = 2 * stats.t.sf(np.abs(t_values), df=df_res)
            else:
                se = np.full((n_params, len(context.outcome_names)), np.nan)
                t_values = np.full((n_params, len(context.outcome_names)), np.nan)
                pvalues = np.full((n_params, len(context.outcome_names)), np.nan)

            for outcome_idx, outcome_name in enumerate(context.outcome_names):
                bic_rows.append(
                    {
                        "edf": n_params,
                        "bic": _gaussian_bic(batch["rss"][outcome_idx], context.n_obs, n_params),
                        "combination_index": combo_idx,
                        "outcome": outcome_name,
                    }
                )
                for factor_level, row_idx in enumerate(exposure_positions, start=1):
                    vib_rows.append(
                        {
                            "estimate": batch["beta"][row_idx, outcome_idx],
                            "se": se[row_idx, outcome_idx],
                            "z": t_values[row_idx, outcome_idx],
                            "pvalue": pvalues[row_idx, outcome_idx],
                            "combination_index": combo_idx,
                            "factor_level": factor_level,
                            "outcome": outcome_name,
                        }
                    )
        else:
            fit = qr_new_outcome(state, context.y)
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

    vib_frame = pd.DataFrame(vib_rows)
    bic_frame = pd.DataFrame(bic_rows)
    named_combos = [tuple(context.adjust_names[idx] for idx in combo) for combo in combos]
    return VibrationResult(
        vib_frame=vib_frame,
        bic_frame=bic_frame,
        combinations=named_combos,
        adjust=list(context.adjust_names),
        family="gaussian",
        exposure=context.exposure_name,
        outcomes=list(context.outcome_names),
        k=k,
    )


def conduct_vibration_for_k(
    data,
    *,
    outcome: str | None = None,
    outcomes: Sequence[str] | None = None,
    exposure: str,
    adjust_by: Sequence[str],
    base_covariates: Sequence[str] | None = None,
    family: str = "gaussian",
    k: int = 1,
    drop_first: bool = True,
) -> VibrationResult:
    """Run VoE for a fixed number of adjustment groups."""

    context = _prepare_design(
        data,
        outcome=outcome,
        outcomes=outcomes,
        exposure=exposure,
        adjust_by=adjust_by,
        base_covariates=base_covariates,
        drop_first=drop_first,
        family=family,
    )
    return _conduct_vibration_for_k_prepared(context, k)


def conduct_vibration(
    data,
    *,
    outcome: str | None = None,
    outcomes: Sequence[str] | None = None,
    exposure: str,
    adjust_by: Sequence[str],
    base_covariates: Sequence[str] | None = None,
    family: str = "gaussian",
    k_min: int | None = None,
    k_max: int | None = None,
    drop_first: bool = True,
) -> VibrationResult:
    """Run exhaustive VoE across all requested adjustment-set sizes."""

    pd = _require_pandas()
    context = _prepare_design(
        data,
        outcome=outcome,
        outcomes=outcomes,
        exposure=exposure,
        adjust_by=adjust_by,
        base_covariates=base_covariates,
        drop_first=drop_first,
        family=family,
    )

    n_adjust = len(context.adjust_groups)
    if k_min is None:
        k_min = 1
    if k_max is None:
        k_max = n_adjust
    if k_min > k_max:
        raise ValueError("`k_min` must be less than or equal to `k_max`.")

    fixed_results = [
        _conduct_vibration_for_k_prepared(context, k)
        for k in range(k_min, k_max + 1)
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
