#!/usr/bin/env Rscript

args_full <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args_full, value = TRUE)
if (!length(file_arg)) {
	stop("Unable to determine script path.", call. = FALSE)
}

script_path <- normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = TRUE)
bench_dir <- dirname(script_path)
repo_root <- normalizePath(file.path(bench_dir, ".."), winslash = "/", mustWork = TRUE)

source(file.path(repo_root, "R", "qr_backend.R"))
source(file.path(repo_root, "R", "vibration.R"))
source(file.path(bench_dir, "benchmark_discrepancy_helpers.R"))

print_help <- function() {
	cat(
		paste(
			"Usage:",
			"  Rscript benchmarks/benchmark_voe_multioutcome_lm_baseline.R [options]",
			"",
			"Options:",
			"  --n-obs INT",
			"  --n-adjust INT",
			"  --n-outcomes INT",
			"  --k-min INT",
			"  --k-max INT",
			"  --reps INT",
			"  --seed INT",
			"  --help",
			sep = "\n"
		)
	)
}

opts <- parse_cli_args(
	commandArgs(trailingOnly = TRUE),
	list(
		n_obs = "2000",
		n_adjust = "12",
		n_outcomes = "100",
		k_min = "6",
		k_max = "6",
		reps = "1",
		seed = "123",
		help = FALSE
	)
)

if (isTRUE(opts$help)) {
	print_help()
	quit(save = "no", status = 0L)
}

synthetic <- make_voe_multioutcome_data(
	n_obs = int_arg(opts$n_obs, "--n-obs"),
	n_adjust = int_arg(opts$n_adjust, "--n-adjust"),
	n_outcomes = int_arg(opts$n_outcomes, "--n-outcomes"),
	seed = int_arg(opts$seed, "--seed")
)

config <- synthetic
config$k_min <- int_arg(opts$k_min, "--k-min")
config$k_max <- int_arg(opts$k_max, "--k-max")
config$reps <- int_arg(opts$reps, "--reps")
config$is_fixed_k <- identical(config$k_min, config$k_max)
config$n_models <- sum(vapply(config$k_min:config$k_max, function(k) {
	choose(length(config$adjust_names), k)
}, numeric(1L)))

if (config$reps < 1L) {
	stop("`--reps` must be at least 1.", call. = FALSE)
}
if (config$k_min > config$k_max) {
	stop("`--k-min` must be less than or equal to `--k-max`.", call. = FALSE)
}

qr_validation <- run_voe_multioutcome_qr(config)
lm_validation <- run_voe_multioutcome_lm(config)
validation <- validate_voe_multioutcome(qr_validation, lm_validation, config$is_fixed_k)

invisible(run_voe_multioutcome_qr(config))
invisible(run_voe_multioutcome_lm(config))

qr_times <- time_many(function() run_voe_multioutcome_qr(config), config$reps)
lm_times <- time_many(function() run_voe_multioutcome_lm(config), config$reps)

print_voe_multioutcome_summary(config, qr_times, lm_times, validation)
