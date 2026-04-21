#!/usr/bin/env python3
"""Benchmark fixed-design many-outcome QR reuse against native Python OLS fits."""

from __future__ import annotations

import argparse

from benchmark_discrepancy_helpers import (
    make_many_outcomes_data,
    print_many_outcomes_summary,
    run_many_outcomes_native,
    run_many_outcomes_qr,
    time_many,
    validate_many_outcomes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-obs", type=int, default=23922)
    parser.add_argument("--n-covariates", type=int, default=87)
    parser.add_argument("--n-outcomes", type=int, default=100)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reps < 1:
        raise ValueError("`--reps` must be at least 1.")

    config = {
        "n_obs": args.n_obs,
        "n_covariates": args.n_covariates,
        "n_outcomes": args.n_outcomes,
        "reps": args.reps,
        "seed": args.seed,
    }
    synthetic = make_many_outcomes_data(
        n_obs=args.n_obs,
        n_covariates=args.n_covariates,
        n_outcomes=args.n_outcomes,
        seed=args.seed,
    )

    qr_validation = run_many_outcomes_qr(
        synthetic["frame"], synthetic["predictor_names"], synthetic["outcome_names"]
    )
    native_validation = run_many_outcomes_native(
        synthetic["frame"], synthetic["predictor_names"], synthetic["outcome_names"]
    )
    validation = validate_many_outcomes(qr_validation, native_validation)

    run_many_outcomes_qr(synthetic["frame"], synthetic["predictor_names"], synthetic["outcome_names"])
    run_many_outcomes_native(synthetic["frame"], synthetic["predictor_names"], synthetic["outcome_names"])

    qr_times = time_many(
        lambda: run_many_outcomes_qr(
            synthetic["frame"], synthetic["predictor_names"], synthetic["outcome_names"]
        ),
        args.reps,
    )
    native_times = time_many(
        lambda: run_many_outcomes_native(
            synthetic["frame"], synthetic["predictor_names"], synthetic["outcome_names"]
        ),
        args.reps,
    )

    print_many_outcomes_summary(config, qr_times, native_times, validation)


if __name__ == "__main__":
    main()
