#!/usr/bin/env python3
"""Benchmark multi-outcome Gaussian VoE QR updates against native Python OLS enumeration."""

from __future__ import annotations

import argparse
import math

from benchmark_discrepancy_helpers import (
    make_voe_multioutcome_data,
    print_voe_multioutcome_summary,
    run_voe_multioutcome_native,
    run_voe_multioutcome_qr,
    time_many,
    validate_voe_multioutcome,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-obs", type=int, default=2000)
    parser.add_argument("--n-adjust", type=int, default=12)
    parser.add_argument("--n-outcomes", type=int, default=100)
    parser.add_argument("--k-min", type=int, default=6)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reps < 1:
        raise ValueError("`--reps` must be at least 1.")
    if args.k_min > args.k_max:
        raise ValueError("`--k-min` must be less than or equal to `--k-max`.")

    config = make_voe_multioutcome_data(
        n_obs=args.n_obs,
        n_adjust=args.n_adjust,
        n_outcomes=args.n_outcomes,
        seed=args.seed,
    )
    config["k_min"] = args.k_min
    config["k_max"] = args.k_max
    config["reps"] = args.reps
    config["is_fixed_k"] = args.k_min == args.k_max
    config["n_models"] = sum(math.comb(len(config["adjust_names"]), k) for k in range(args.k_min, args.k_max + 1))

    qr_validation = run_voe_multioutcome_qr(config)
    native_validation = run_voe_multioutcome_native(config)
    validation = validate_voe_multioutcome(qr_validation, native_validation, config["is_fixed_k"])

    run_voe_multioutcome_qr(config)
    run_voe_multioutcome_native(config)

    qr_times = time_many(lambda: run_voe_multioutcome_qr(config), args.reps)
    native_times = time_many(lambda: run_voe_multioutcome_native(config), args.reps)

    print_voe_multioutcome_summary(config, qr_times, native_times, validation)


if __name__ == "__main__":
    main()
