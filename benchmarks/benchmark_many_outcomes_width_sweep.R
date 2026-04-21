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
			"  Rscript benchmarks/benchmark_many_outcomes_width_sweep.R [options]",
			"",
			"Options:",
			"  --n-obs INT",
			"  --n-outcomes INT",
			"  --widths INT,INT,...",
			"  --reps INT",
			"  --seed INT",
			"  --output-csv PATH",
			"  --output-plot PATH",
			"  --help",
			sep = "\n"
		)
	)
}

opts <- parse_cli_args(
	commandArgs(trailingOnly = TRUE),
	list(
		n_obs = "23922",
		n_outcomes = "100",
		widths = "10,20,40,80,120",
		reps = "1",
		seed = "123",
		output_csv = "",
		output_plot = "",
		help = FALSE
	)
)

if (isTRUE(opts$help)) {
	print_help()
	quit(save = "no", status = 0L)
}

n_obs <- int_arg(opts$n_obs, "--n-obs")
n_outcomes <- int_arg(opts$n_outcomes, "--n-outcomes")
reps <- int_arg(opts$reps, "--reps")
seed <- int_arg(opts$seed, "--seed")
widths <- split_ints(opts$widths, "--widths")

if (reps < 1L) {
	stop("`--reps` must be at least 1.", call. = FALSE)
}

rows <- lapply(widths, function(width) {
	synthetic <- make_many_outcomes_data(
		n_obs = n_obs,
		n_covariates = width,
		n_outcomes = n_outcomes,
		seed = seed + width
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
	}, reps)
	lm_times <- time_many(function() {
		run_many_outcomes_lm(synthetic$data, synthetic$predictor_names, synthetic$outcome_names)
	}, reps)

	record <- data.frame(
		n_covariates = width,
		design_columns = width + 2L,
		n_obs = n_obs,
		n_outcomes = n_outcomes,
		qr_seconds = stats::median(qr_times),
		lm_seconds = stats::median(lm_times),
		speedup = stats::median(lm_times) / stats::median(qr_times),
		max_diff = validation$max_diff,
		stringsAsFactors = FALSE,
		check.names = FALSE
	)
	cat(sprintf(
		"done width=%i design_cols=%i qr=%.3f lm=%.3f speedup=%.2fx\n",
		record$n_covariates,
		record$design_columns,
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

if (nzchar(opts$output_plot)) {
	grDevices::png(opts$output_plot, width = 1000, height = 600, res = 150)
	plot(
		results$design_columns,
		results$speedup,
		type = "o",
		lwd = 2,
		pch = 16,
		col = "#3366AA",
		xlab = "Design columns (intercept + exposure + covariates)",
		ylab = "Speedup (lm / QR)",
		main = "Many-outcome QR speedup vs design width"
	)
	grid()
	invisible(grDevices::dev.off())
	cat(sprintf("Wrote %s\n", opts$output_plot))
}
