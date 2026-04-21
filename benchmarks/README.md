# QR Speedup Benchmarks

These scripts benchmark the Gaussian VoE QR-update path against a fresh
full-QR-per-model baseline inside this `voe` fork.

## Files

- `benchmark_voe_qr_speedup.R`: R benchmark runner
- `regenerate_nhanes_fixture.R`: rebuilds the shared NHANES-derived CSV fixture
- `data/nhanes_voe_gaussian.csv`: committed benchmark fixture shared by R and Python
- `../python/benchmarks/benchmark_voe_qr_speedup.py`: Python benchmark runner

## Regenerate The NHANES Fixture

```bash
cd /Users/randalljellis/Documents/qr_trick/voe
Rscript benchmarks/regenerate_nhanes_fixture.R
```

This writes `benchmarks/data/nhanes_voe_gaussian.csv` with the columns:

- `LBXVID`
- `LBXBCD_log_z`
- `RIDAGEYR`
- `male`
- `SES_LEVEL`
- `current_past_smoking`
- `education`
- `RIDRETH1`
- `any_cad`
- `any_ht`
- `any_diabetes`

## Run The Benchmarks

```bash
cd /Users/randalljellis/Documents/qr_trick/voe
Rscript benchmarks/benchmark_voe_qr_speedup.R --dataset synthetic
Rscript benchmarks/benchmark_voe_qr_speedup.R --dataset nhanes --reps 3

python3 python/benchmarks/benchmark_voe_qr_speedup.py --dataset synthetic
python3 python/benchmarks/benchmark_voe_qr_speedup.py --dataset nhanes --reps 3
```

Both runners support:

- `--dataset synthetic|nhanes`
- `--k-min`
- `--k-max`
- `--reps`
- `--seed`
- `--n-obs` for synthetic data only
- `--n-adjust` for synthetic data only

If `k_min == k_max`, the scripts benchmark the fixed-`k` public API. Otherwise
they benchmark the exhaustive API over the requested `k` range.

## Output Fields

Each script prints:

- dataset preset
- row count
- adjustor count
- model count
- `k` or `k` range
- QR median seconds
- fresh-QR median seconds
- speedup multiplier
- max estimate diff from the validation pass
- max BIC diff from the validation pass

Validation runs once before timing and exits non-zero if the QR-backed result
and fresh baseline disagree beyond tolerance.
