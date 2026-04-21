#!/usr/bin/env python3
"""Sweep multi-outcome Gaussian VoE speedup across outcome count."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_discrepancy_helpers import (
    make_voe_multioutcome_data,
    run_voe_multioutcome_native,
    run_voe_multioutcome_qr,
    time_many,
    validate_voe_multioutcome,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-obs", type=int, default=2000)
    parser.add_argument("--n-adjust", type=int, default=12)
    parser.add_argument("--outcome-counts", type=str, default="5,10,20,40,80")
    parser.add_argument("--k-min", type=int, default=6)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-csv", type=str, default="")
    return parser.parse_args()


def _parse_counts(raw: str) -> list[int]:
    counts = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not counts:
        raise ValueError("`--outcome-counts` must not be empty.")
    return counts


def main() -> None:
    args = parse_args()
    if args.reps < 1:
        raise ValueError("`--reps` must be at least 1.")
    if args.k_min > args.k_max:
        raise ValueError("`--k-min` must be less than or equal to `--k-max`.")

    rows = []
    for n_outcomes in _parse_counts(args.outcome_counts):
        config = make_voe_multioutcome_data(
            n_obs=args.n_obs,
            n_adjust=args.n_adjust,
            n_outcomes=n_outcomes,
            seed=args.seed + n_outcomes,
        )
        config["k_min"] = args.k_min
        config["k_max"] = args.k_max
        config["reps"] = args.reps
        config["is_fixed_k"] = args.k_min == args.k_max
        config["n_models"] = sum(
            math.comb(len(config["adjust_names"]), k) for k in range(args.k_min, args.k_max + 1)
        )

        qr_validation = run_voe_multioutcome_qr(config)
        native_validation = run_voe_multioutcome_native(config)
        validation = validate_voe_multioutcome(qr_validation, native_validation, config["is_fixed_k"])

        run_voe_multioutcome_qr(config)
        run_voe_multioutcome_native(config)

        qr_times = time_many(lambda: run_voe_multioutcome_qr(config), args.reps)
        native_times = time_many(lambda: run_voe_multioutcome_native(config), args.reps)

        qr_seconds = float(np.median(qr_times))
        native_seconds = float(np.median(native_times))
        rows.append(
            {
                "n_obs": args.n_obs,
                "n_adjust": args.n_adjust,
                "n_outcomes": n_outcomes,
                "k_min": args.k_min,
                "k_max": args.k_max,
                "n_models": config["n_models"],
                "qr_seconds": qr_seconds,
                "native_seconds": native_seconds,
                "speedup": native_seconds / qr_seconds,
                "max_estimate_diff": validation["max_estimate_diff"],
                "max_bic_diff": validation["max_bic_diff"],
            }
        )
        print(
            f"done outcomes={n_outcomes} qr={qr_seconds:.3f} "
            f"native={native_seconds:.3f} speedup={native_seconds / qr_seconds:.2f}x"
        )

    results = pd.DataFrame(rows)
    print(results.to_string(index=False))

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
