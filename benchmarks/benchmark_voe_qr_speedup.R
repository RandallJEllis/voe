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

parse_args <- function(args) {
	opts <- list(
		dataset = "synthetic",
		k_min = NULL,
		k_max = NULL,
		reps = 1L,
		seed = 123L,
		n_obs = 8000L,
		n_adjust = 12L,
		help = FALSE
	)

	i <- 1L
	while (i <= length(args)) {
		token <- args[[i]]
		if (!startsWith(token, "--")) {
			stop(sprintf("Unexpected argument: %s", token), call. = FALSE)
		}

		if (token == "--help") {
			opts$help <- TRUE
			i <- i + 1L
			next
		}

		if (grepl("=", token, fixed = TRUE)) {
			parts <- strsplit(sub("^--", "", token), "=", fixed = TRUE)[[1]]
			key <- gsub("-", "_", parts[[1]])
			value <- parts[[2]]
			i <- i + 1L
		} else {
			key <- gsub("-", "_", sub("^--", "", token))
			i <- i + 1L
			if (i > length(args)) {
				stop(sprintf("Missing value for --%s", gsub("_", "-", key)), call. = FALSE)
			}
			value <- args[[i]]
			i <- i + 1L
		}

		if (!key %in% names(opts)) {
			stop(sprintf("Unknown option: --%s", gsub("_", "-", key)), call. = FALSE)
		}

		if (key %in% c("k_min", "k_max", "reps", "seed", "n_obs", "n_adjust")) {
			opts[[key]] <- as.integer(value)
		} else {
			opts[[key]] <- value
		}
	}

	opts
}

print_help <- function() {
	cat(
		paste(
			"Usage:",
			"  Rscript benchmarks/benchmark_voe_qr_speedup.R [options]",
			"",
			"Options:",
			"  --dataset synthetic|nhanes",
			"  --k-min INT",
			"  --k-max INT",
			"  --reps INT",
			"  --seed INT",
			"  --n-obs INT       Synthetic rows only",
			"  --n-adjust INT    Synthetic adjustors only",
			"  --help",
			sep = "\n"
		)
	)
}

make_synthetic_data <- function(n_obs, n_adjust, seed) {
	if (n_adjust < 1L) {
		stop("Synthetic benchmarks require at least one adjustor.", call. = FALSE)
	}

	set.seed(seed)
	dat <- data.frame(
		y = rnorm(n_obs),
		x = rnorm(n_obs),
		age = rnorm(n_obs)
	)

	adjust_names <- sprintf("z%i", seq_len(n_adjust))
	for (idx in seq_along(adjust_names)) {
		name <- adjust_names[[idx]]
		if (idx %% 4L == 0L) {
			dat[[name]] <- factor(sample(c("a", "b", "c", "d"), n_obs, replace = TRUE))
		} else {
			dat[[name]] <- rnorm(n_obs)
		}
	}

	y_signal <- 0.7 * dat$x - 0.4 * dat$age
	for (idx in seq_len(min(4L, n_adjust))) {
		name <- adjust_names[[idx]]
		if (is.factor(dat[[name]])) {
			effect_map <- c(a = -0.3, b = 0.0, c = 0.3, d = 0.5)
			y_signal <- y_signal + unname(effect_map[as.character(dat[[name]])])
		} else {
			y_signal <- y_signal + (0.2 / idx) * dat[[name]]
		}
	}

	dat$y <- y_signal + rnorm(n_obs, sd = 0.5)

	list(
		data = dat,
		base_formula = y ~ x + age,
		adjustby = stats::as.formula(
			paste("~", paste(adjust_names, collapse = " + "))
		),
		n_adjust = n_adjust
	)
}

load_nhanes_data <- function() {
	fixture_path <- file.path(bench_dir, "data", "nhanes_voe_gaussian.csv")
	if (!file.exists(fixture_path)) {
		stop(
			paste(
				"Missing NHANES benchmark fixture.",
				sprintf("Run `Rscript %s` first.", file.path("benchmarks", "regenerate_nhanes_fixture.R"))
			),
			call. = FALSE
		)
	}

	dat <- utils::read.csv(fixture_path, stringsAsFactors = FALSE)
	for (name in c("SES_LEVEL", "current_past_smoking", "education", "RIDRETH1")) {
		dat[[name]] <- factor(dat[[name]])
	}

	adjust_names <- c(
		"SES_LEVEL",
		"current_past_smoking",
		"education",
		"RIDRETH1",
		"any_cad",
		"any_ht",
		"any_diabetes"
	)

	list(
		data = dat,
		base_formula = LBXVID ~ LBXBCD_log_z + RIDAGEYR + male,
		adjustby = stats::as.formula(
			paste("~", paste(adjust_names, collapse = " + "))
		),
		n_adjust = length(adjust_names)
	)
}

resolve_config <- function(opts) {
	config <- if (identical(opts$dataset, "nhanes")) {
		load_nhanes_data()
	} else if (identical(opts$dataset, "synthetic")) {
		make_synthetic_data(opts$n_obs, opts$n_adjust, opts$seed)
	} else {
		stop("`--dataset` must be one of: synthetic, nhanes", call. = FALSE)
	}

	if (is.null(opts$k_min)) {
		opts$k_min <- if (identical(opts$dataset, "nhanes")) 1L else min(6L, config$n_adjust)
	}
	if (is.null(opts$k_max)) {
		opts$k_max <- if (identical(opts$dataset, "nhanes")) config$n_adjust else min(6L, config$n_adjust)
	}

	if (opts$reps < 1L) {
		stop("`--reps` must be at least 1.", call. = FALSE)
	}
	if (opts$k_min > opts$k_max) {
		stop("`--k-min` must be less than or equal to `--k-max`.", call. = FALSE)
	}

	config$dataset <- opts$dataset
	config$k_min <- opts$k_min
	config$k_max <- opts$k_max
	config$reps <- opts$reps
	config$seed <- opts$seed
	config$n_obs <- nrow(config$data)
	config$n_models <- sum(vapply(opts$k_min:opts$k_max, function(k) {
		choose(config$n_adjust, k)
	}, numeric(1L)))
	config$is_fixed_k <- identical(opts$k_min, opts$k_max)
	config
}

conductVibrationForK_gaussian_fresh_prepared <- function(context, k, print_progress = FALSE) {
	if (isTRUE(context$is_multi_outcome)) {
		stop("Benchmark baseline only supports single-outcome gaussian VoE.", call. = FALSE)
	}

	n_adjust <- length(context$adjust_groups)
	varComb <- combination_matrix(n_adjust, k)
	n_models <- ncol(varComb)
	model_results <- vector("list", n_models)
	coef_colnames <- NULL

	for (ii in seq_len(n_models)) {
		current_terms <- if (k == 0L) integer(0) else varComb[, ii]
		col_indices <- context$fixed_col_indices
		coef_names <- context$fixed_coef_names

		if (length(current_terms)) {
			for (term_idx in current_terms) {
				term_group <- context$adjust_groups[[term_idx]]
				col_indices <- c(col_indices, term_group$col_indices)
				coef_names <- c(coef_names, term_group$coef_names)
			}
		}

		state <- qr_state_from_cols(context$X_weighted, col_indices)
		model_fit <- fit_gaussian_qr_model(context, state, coef_names)
		rowIndex <- find_exposure_rows(
			rownames(model_fit$coef_table),
			context$exposure_term,
			context$exposure_vars
		)
		if (!length(rowIndex)) {
			next
		}

		if (is.null(coef_colnames)) {
			coef_colnames <- colnames(model_fit$coef_table)
		}

		model_results[[ii]] <- list(
			coefs = model_fit$coef_table[rowIndex, , drop = FALSE],
			bic_edf = unname(model_fit$bic["edf"]),
			bic_val = unname(model_fit$bic["bic"]),
			combo_idx = ii,
			n_levels = length(rowIndex)
		)
	}

	model_results <- Filter(Negate(is.null), model_results)
	if (!length(model_results)) {
		return(NULL)
	}

	assemble_fixed_k_result(
		model_results = model_results,
		coef_colnames = coef_colnames,
		k = k,
		varComb = varComb,
		family = "gaussian",
		base_formula = context$base_formula,
		adjust_terms = context$adjust
	)
}

conductVibration_gaussian_fresh_prepared <- function(context, k_min, k_max) {
	returnFrames <- lapply(k_min:k_max, function(k) {
		conductVibrationForK_gaussian_fresh_prepared(context, k, print_progress = FALSE)
	})
	returnFrames <- Filter(Negate(is.null), returnFrames)
	if (!length(returnFrames)) {
		return(NULL)
	}
	gatherFrames(returnFrames)
}

run_qr_once <- function(config) {
	if (config$is_fixed_k) {
		return(conductVibrationForK(
			config$base_formula,
			config$data,
			config$adjustby,
			k = config$k_min,
			family = "gaussian",
			print_progress = FALSE
		))
	}

	conductVibration(
		config$base_formula,
		config$data,
		config$adjustby,
		family = "gaussian",
		kMin = config$k_min,
		kMax = config$k_max,
		print_progress = FALSE
	)
}

run_fresh_once <- function(config) {
	context <- prepare_gaussian_qr_context(
		config$base_formula,
		config$data,
		config$adjustby,
		list()
	)
	if (!isTRUE(context$supported)) {
		stop(sprintf("Fresh baseline could not prepare gaussian QR context: %s", context$reason), call. = FALSE)
	}

	if (config$is_fixed_k) {
		return(conductVibrationForK_gaussian_fresh_prepared(context, config$k_min, print_progress = FALSE))
	}

	conductVibration_gaussian_fresh_prepared(context, config$k_min, config$k_max)
}

result_frames <- function(result, is_fixed_k) {
	if (is_fixed_k) {
		vib <- as.data.frame(result$vibration, check.names = FALSE, stringsAsFactors = FALSE)
		bic <- as.data.frame(result$bic, check.names = FALSE, stringsAsFactors = FALSE)
		estimate_col <- "Estimate"
		order_cols_vib <- c("combination_index", "factor_level")
		order_cols_bic <- c("combination_index")
	} else {
		vib <- as.data.frame(result$vibFrame, check.names = FALSE, stringsAsFactors = FALSE)
		bic <- as.data.frame(result$bicFrame, check.names = FALSE, stringsAsFactors = FALSE)
		estimate_col <- "estimate"
		order_cols_vib <- c("k", "combination_index", "factor_level")
		order_cols_bic <- c("k", "combination_index")
	}

	vib <- vib[do.call(order, vib[order_cols_vib]), , drop = FALSE]
	bic <- bic[do.call(order, bic[order_cols_bic]), , drop = FALSE]
	rownames(vib) <- NULL
	rownames(bic) <- NULL

	list(
		vib = vib,
		bic = bic,
		estimate_col = estimate_col
	)
}

validate_results <- function(qr_result, fresh_result, is_fixed_k, tol_estimate = 1e-8, tol_bic = 1e-8) {
	qr_frames <- result_frames(qr_result, is_fixed_k)
	fresh_frames <- result_frames(fresh_result, is_fixed_k)

	if (!identical(names(qr_frames$vib), names(fresh_frames$vib))) {
		stop("QR and fresh baseline vibration columns differ.", call. = FALSE)
	}
	if (!identical(names(qr_frames$bic), names(fresh_frames$bic))) {
		stop("QR and fresh baseline BIC columns differ.", call. = FALSE)
	}
	if (nrow(qr_frames$vib) != nrow(fresh_frames$vib) || nrow(qr_frames$bic) != nrow(fresh_frames$bic)) {
		stop("QR and fresh baseline returned different row counts.", call. = FALSE)
	}

	identifier_cols_vib <- intersect(c("k", "combination_index", "factor_level"), names(qr_frames$vib))
	identifier_cols_bic <- intersect(c("k", "combination_index"), names(qr_frames$bic))

	for (name in identifier_cols_vib) {
		if (!identical(qr_frames$vib[[name]], fresh_frames$vib[[name]])) {
			stop(sprintf("QR and fresh baseline differ in vibration identifier column `%s`.", name), call. = FALSE)
		}
	}
	for (name in identifier_cols_bic) {
		if (!identical(qr_frames$bic[[name]], fresh_frames$bic[[name]])) {
			stop(sprintf("QR and fresh baseline differ in BIC identifier column `%s`.", name), call. = FALSE)
		}
	}

	estimate_diff <- max(abs(qr_frames$vib[[qr_frames$estimate_col]] - fresh_frames$vib[[fresh_frames$estimate_col]]))
	bic_diff <- max(abs(qr_frames$bic[["bic"]] - fresh_frames$bic[["bic"]]))

	if (!is.finite(estimate_diff) || estimate_diff > tol_estimate) {
		stop(
			sprintf("Validation failed: max estimate diff %.3e exceeds tolerance %.3e", estimate_diff, tol_estimate),
			call. = FALSE
		)
	}
	if (!is.finite(bic_diff) || bic_diff > tol_bic) {
		stop(
			sprintf("Validation failed: max BIC diff %.3e exceeds tolerance %.3e", bic_diff, tol_bic),
			call. = FALSE
		)
	}

	list(max_estimate_diff = estimate_diff, max_bic_diff = bic_diff)
}

time_many <- function(fun, reps) {
	times <- numeric(reps)
	for (idx in seq_len(reps)) {
		times[[idx]] <- system.time(fun())[["elapsed"]]
	}
	times
}

print_summary <- function(config, qr_times, fresh_times, validation) {
	k_label <- if (config$is_fixed_k) {
		as.character(config$k_min)
	} else {
		sprintf("%i:%i", config$k_min, config$k_max)
	}

	qr_median <- stats::median(qr_times)
	fresh_median <- stats::median(fresh_times)
	speedup <- fresh_median / qr_median

	cat(sprintf("Dataset           : %s\n", config$dataset))
	cat(sprintf("Rows              : %i\n", config$n_obs))
	cat(sprintf("Adjustors         : %i\n", config$n_adjust))
	cat(sprintf("Model count       : %i\n", config$n_models))
	cat(sprintf("k                 : %s\n", k_label))
	cat(sprintf("QR median seconds : %.6f\n", qr_median))
	cat(sprintf("Fresh QR seconds  : %.6f\n", fresh_median))
	cat(sprintf("Speedup           : %.2fx\n", speedup))
	cat(sprintf("Max estimate diff : %.3e\n", validation$max_estimate_diff))
	cat(sprintf("Max BIC diff      : %.3e\n", validation$max_bic_diff))
}

opts <- parse_args(commandArgs(trailingOnly = TRUE))
if (isTRUE(opts$help)) {
	print_help()
	quit(save = "no", status = 0L)
}

config <- resolve_config(opts)

qr_validation <- run_qr_once(config)
fresh_validation <- run_fresh_once(config)
validation <- validate_results(qr_validation, fresh_validation, config$is_fixed_k)

invisible(run_qr_once(config))
invisible(run_fresh_once(config))

qr_times <- time_many(function() run_qr_once(config), config$reps)
fresh_times <- time_many(function() run_fresh_once(config), config$reps)
print_summary(config, qr_times, fresh_times, validation)
