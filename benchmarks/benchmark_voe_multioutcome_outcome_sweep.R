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
			"  Rscript benchmarks/benchmark_voe_multioutcome_outcome_sweep.R [options]",
			"",
			"Options:",
			"  --n-obs INT",
			"  --n-adjust INT",
			"  --outcome-counts INT,INT,...",
			"  --k-min INT",
			"  --k-max INT",
			"  --reps INT",
			"  --seed INT",
			"  --output-csv PATH",
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
		outcome_counts = "5,10,20,40,80",
		k_min = "6",
		k_max = "6",
		reps = "1",
		seed = "123",
		output_csv = "",
		help = FALSE
	)
)

if (isTRUE(opts$help)) {
	print_help()
	quit(save = "no", status = 0L)
}

n_obs <- int_arg(opts$n_obs, "--n-obs")
n_adjust <- int_arg(opts$n_adjust, "--n-adjust")
k_min <- int_arg(opts$k_min, "--k-min")
k_max <- int_arg(opts$k_max, "--k-max")
reps <- int_arg(opts$reps, "--reps")
seed <- int_arg(opts$seed, "--seed")
outcome_counts <- split_ints(opts$outcome_counts, "--outcome-counts")

if (reps < 1L) {
	stop("`--reps` must be at least 1.", call. = FALSE)
}
if (k_min > k_max) {
	stop("`--k-min` must be less than or equal to `--k-max`.", call. = FALSE)
}

rows <- lapply(outcome_counts, function(n_outcomes) {
	config <- make_voe_multioutcome_data(
		n_obs = n_obs,
		n_adjust = n_adjust,
		n_outcomes = n_outcomes,
		seed = seed + n_outcomes
	)
	config$k_min <- k_min
	config$k_max <- k_max
	config$reps <- reps
	config$is_fixed_k <- identical(k_min, k_max)
	config$n_models <- sum(vapply(k_min:k_max, function(k) {
		choose(length(config$adjust_names), k)
	}, numeric(1L)))

	qr_validation <- run_voe_multioutcome_qr(config)
	lm_validation <- run_voe_multioutcome_lm(config)
	validation <- validate_voe_multioutcome(qr_validation, lm_validation, config$is_fixed_k)

	invisible(run_voe_multioutcome_qr(config))
	invisible(run_voe_multioutcome_lm(config))

	qr_times <- time_many(function() run_voe_multioutcome_qr(config), reps)
	lm_times <- time_many(function() run_voe_multioutcome_lm(config), reps)

	record <- data.frame(
		n_obs = n_obs,
		n_adjust = n_adjust,
		n_outcomes = n_outcomes,
		k_min = k_min,
		k_max = k_max,
		n_models = config$n_models,
		qr_seconds = stats::median(qr_times),
		lm_seconds = stats::median(lm_times),
		speedup = stats::median(lm_times) / stats::median(qr_times),
		max_estimate_diff = validation$max_estimate_diff,
		max_bic_diff = validation$max_bic_diff,
		stringsAsFactors = FALSE,
		check.names = FALSE
	)
	cat(sprintf(
		"done outcomes=%i qr=%.3f lm=%.3f speedup=%.2fx\n",
		record$n_outcomes,
		record$qr_seconds,
		record$lm_seconds,
		record$speedup
	))
	record
})

results <- do.call(rbind, rows)
rownames(results) <- NULL
print(results)

if (nzchar(opts$output_csv)) {
	utils::write.csv(results, opts$output_csv, row.names = FALSE)
	cat(sprintf("Wrote %s\n", opts$output_csv))
}
