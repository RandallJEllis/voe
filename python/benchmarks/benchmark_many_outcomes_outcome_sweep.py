#!/usr/bin/env python3
"""Sweep fixed-design many-outcome QR speedup across outcome count."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_discrepancy_helpers import (
    make_many_outcomes_data,
    run_many_outcomes_native,
    run_many_outcomes_qr,
    time_many,
    validate_many_outcomes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-obs", type=int, default=23922)
    parser.add_argument("--n-covariates", type=int, default=87)
    parser.add_argument("--outcome-counts", type=str, default="10,25,50,75,100")
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

    rows = []
    for n_outcomes in _parse_counts(args.outcome_counts):
        synthetic = make_many_outcomes_data(
            n_obs=args.n_obs,
            n_covariates=args.n_covariates,
            n_outcomes=n_outcomes,
            seed=args.seed + n_outcomes,
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

        qr_seconds = float(np.median(qr_times))
        native_seconds = float(np.median(native_times))
        rows.append(
            {
                "n_obs": args.n_obs,
                "n_covariates": args.n_covariates,
                "design_columns": args.n_covariates + 2,
                "n_outcomes": n_outcomes,
                "qr_seconds": qr_seconds,
                "native_seconds": native_seconds,
                "speedup": native_seconds / qr_seconds,
                "max_diff": validation["max_diff"],
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
