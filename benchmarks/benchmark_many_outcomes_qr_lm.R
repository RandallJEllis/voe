#!/usr/bin/env Rscript

args_full <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args_full, value = TRUE)
if (!length(file_arg)) {
	stop("Unable to determine script path.", call. = FALSE)
}

script_path <- normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = TRUE)
bench_dir <- dirname(script_path)
source(file.path(bench_dir, "benchmark_discrepancy_helpers.R"))

print_help <- function() {
	cat(
		paste(
			"Usage:",
			"  Rscript benchmarks/benchmark_many_outcomes_qr_lm.R [options]",
			"",
			"Options:",
			"  --n-obs INT",
			"  --n-covariates INT",
			"  --n-outcomes INT",
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
		n_obs = "23922",
		n_covariates = "87",
		n_outcomes = "100",
		reps = "1",
		seed = "123",
		help = FALSE
	)
)

if (isTRUE(opts$help)) {
	print_help()
	quit(save = "no", status = 0L)
}

config <- list(
	n_obs = int_arg(opts$n_obs, "--n-obs"),
	n_covariates = int_arg(opts$n_covariates, "--n-covariates"),
	n_outcomes = int_arg(opts$n_outcomes, "--n-outcomes"),
	reps = int_arg(opts$reps, "--reps"),
	seed = int_arg(opts$seed, "--seed")
)

if (config$reps < 1L) {
	stop("`--reps` must be at least 1.", call. = FALSE)
}

synthetic <- make_many_outcomes_data(
	n_obs = config$n_obs,
	n_covariates = config$n_covariates,
	n_outcomes = config$n_outcomes,
	seed = config$seed
)

qr_validation <- run_many_outcomes_qr(
	synthetic$data,
	synthetic$predictor_names,
	synthetic$outcome_names
)
lm_validation <- run_many_outcomes_lm(
	synthetic$data,
	synthetic$predictor_names,
	synthetic$outcome_names
)
validation <- validate_many_outcomes(qr_validation, lm_validation)

invisible(run_many_outcomes_qr(synthetic$data, synthetic$predictor_names, synthetic$outcome_names))
invisible(run_many_outcomes_lm(synthetic$data, synthetic$predictor_names, synthetic$outcome_names))

qr_times <- time_many(function() {
	run_many_outcomes_qr(synthetic$data, synthetic$predictor_names, synthetic$outcome_names)
}, config$reps)
lm_times <- time_many(function() {
	run_many_outcomes_lm(synthetic$data, synthetic$predictor_names, synthetic$outcome_names)
}, config$reps)

print_many_outcomes_summary(config, qr_times, lm_times, validation)
