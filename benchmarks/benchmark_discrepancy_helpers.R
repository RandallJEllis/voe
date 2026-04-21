parse_cli_args <- function(args, defaults) {
	opts <- defaults
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
		opts[[key]] <- value
	}
	opts
}

int_arg <- function(value, name) {
	out <- as.integer(value)
	if (is.na(out)) {
		stop(sprintf("`%s` must be an integer.", name), call. = FALSE)
	}
	out
}

num_arg <- function(value, name) {
	out <- as.numeric(value)
	if (is.na(out)) {
		stop(sprintf("`%s` must be numeric.", name), call. = FALSE)
	}
	out
}

split_ints <- function(value, name) {
	if (!nzchar(value)) {
		stop(sprintf("`%s` must not be empty.", name), call. = FALSE)
	}
	parts <- strsplit(value, ",", fixed = TRUE)[[1]]
	ints <- as.integer(trimws(parts))
	if (anyNA(ints)) {
		stop(sprintf("`%s` must be a comma-separated list of integers.", name), call. = FALSE)
	}
	ints
}

time_many <- function(fun, reps) {
	times <- numeric(reps)
	for (idx in seq_len(reps)) {
		times[[idx]] <- system.time(fun())[["elapsed"]]
	}
	times
}

build_multivariate_formula <- function(outcome_names, rhs_terms) {
	stats::as.formula(
		sprintf("cbind(%s) ~ %s",
		        paste(outcome_names, collapse = ", "),
		        paste(rhs_terms, collapse = " + "))
	)
}

single_outcome_formula <- function(outcome_name, rhs_terms) {
	stats::as.formula(
		sprintf("%s ~ %s", outcome_name, paste(rhs_terms, collapse = " + "))
	)
}

make_many_outcomes_data <- function(n_obs, n_covariates, n_outcomes, seed) {
	if (n_obs < 2L) {
		stop("`n_obs` must be at least 2.", call. = FALSE)
	}
	if (n_covariates < 1L) {
		stop("`n_covariates` must be at least 1.", call. = FALSE)
	}
	if (n_outcomes < 1L) {
		stop("`n_outcomes` must be at least 1.", call. = FALSE)
	}

	set.seed(seed)
	exposure <- stats::rnorm(n_obs)
	cov_names <- sprintf("z%i", seq_len(n_covariates))
	cov_matrix <- matrix(stats::rnorm(n_obs * n_covariates), nrow = n_obs, ncol = n_covariates)
	colnames(cov_matrix) <- cov_names

	data <- data.frame(x = exposure, cov_matrix, check.names = FALSE)
	outcome_names <- sprintf("y%i", seq_len(n_outcomes))

	base_effect <- seq(0.25, 0.05, length.out = min(8L, n_covariates))
	for (idx in seq_along(outcome_names)) {
		weights <- numeric(n_covariates)
		if (length(base_effect)) {
			sign_scale <- if ((idx %% 2L) == 0L) -1 else 1
			weights[seq_along(base_effect)] <- sign_scale * base_effect * (1 + idx / (4 * n_outcomes))
		}
		y_signal <- (0.6 + 0.15 * sin(idx / 5)) * exposure + cov_matrix %*% weights
		data[[outcome_names[[idx]]]] <- as.numeric(y_signal + stats::rnorm(n_obs, sd = 0.8))
	}

	list(
		data = data,
		predictor_names = c("x", cov_names),
		outcome_names = outcome_names
	)
}

run_many_outcomes_qr <- function(data, predictor_names, outcome_names) {
	X <- as.matrix(cbind("(Intercept)" = 1, data[, predictor_names, drop = FALSE]))
	qrX <- qr(X)
	degf <- nrow(X) - qrX$rank
	p <- qrX$rank
	p1 <- seq_len(p)
	Rinv <- chol2inv(qrX$qr[p1, p1, drop = FALSE])

	results <- lapply(outcome_names, function(outcome_name) {
		y <- as.matrix(data[[outcome_name]])
		beta <- qr.coef(qrX, y)
		residuals <- qr.resid(qrX, y)
		rss <- sum(residuals^2)
		se <- sqrt(diag(Rinv) * rss / degf)
		est <- beta[qrX$pivot[p1]]
		tval <- est / se
		pvalues <- 2 * stats::pt(abs(tval), degf, lower.tail = FALSE)

		data.frame(
			outcome = outcome_name,
			term = colnames(X)[qrX$pivot[p1]],
			Estimate = as.numeric(est),
			Std.Error = as.numeric(se),
			t.value = as.numeric(tval),
			Pr = as.numeric(pvalues),
			stringsAsFactors = FALSE,
			check.names = FALSE
		)
	})

	do.call(rbind, results)
}

run_many_outcomes_lm <- function(data, predictor_names, outcome_names) {
	results <- lapply(outcome_names, function(outcome_name) {
		form <- single_outcome_formula(outcome_name, predictor_names)
		coef_table <- stats::coef(summary(stats::lm(form, data = data)))
		out <- data.frame(
			outcome = outcome_name,
			term = rownames(coef_table),
			Estimate = coef_table[, "Estimate"],
			Std.Error = coef_table[, "Std. Error"],
			t.value = coef_table[, "t value"],
			Pr = coef_table[, "Pr(>|t|)"],
			stringsAsFactors = FALSE,
			check.names = FALSE
		)
		rownames(out) <- NULL
		out
	})

	do.call(rbind, results)
}

validate_many_outcomes <- function(qr_result, lm_result, tol = 1e-8) {
	order_cols <- c("outcome", "term")
	qr_sorted <- qr_result[do.call(order, qr_result[order_cols]), , drop = FALSE]
	lm_sorted <- lm_result[do.call(order, lm_result[order_cols]), , drop = FALSE]
	rownames(qr_sorted) <- NULL
	rownames(lm_sorted) <- NULL

	if (!identical(qr_sorted[, order_cols], lm_sorted[, order_cols])) {
		stop("QR and lm many-outcome outputs are not aligned.", call. = FALSE)
	}

	numeric_cols <- c("Estimate", "Std.Error", "t.value", "Pr")
	max_diff <- max(abs(as.matrix(qr_sorted[, numeric_cols]) - as.matrix(lm_sorted[, numeric_cols])))
	if (!is.finite(max_diff) || max_diff > tol) {
		stop(
			sprintf("Validation failed: max many-outcome diff %.3e exceeds tolerance %.3e", max_diff, tol),
			call. = FALSE
		)
	}

	list(max_diff = max_diff)
}

print_many_outcomes_summary <- function(config, qr_times, baseline_times, validation) {
	qr_median <- stats::median(qr_times)
	baseline_median <- stats::median(baseline_times)
	cat(sprintf("Rows              : %i\n", config$n_obs))
	cat(sprintf("Covariates        : %i\n", config$n_covariates))
	cat(sprintf("Design columns    : %i\n", config$n_covariates + 2L))
	cat(sprintf("Outcomes          : %i\n", config$n_outcomes))
	cat(sprintf("QR median seconds : %.6f\n", qr_median))
	cat(sprintf("lm median seconds : %.6f\n", baseline_median))
	cat(sprintf("Speedup           : %.2fx\n", baseline_median / qr_median))
	cat(sprintf("Max diff          : %.3e\n", validation$max_diff))
}

make_voe_multioutcome_data <- function(n_obs, n_adjust, n_outcomes, seed) {
	if (n_obs < 2L) {
		stop("`n_obs` must be at least 2.", call. = FALSE)
	}
	if (n_adjust < 1L) {
		stop("`n_adjust` must be at least 1.", call. = FALSE)
	}
	if (n_outcomes < 1L) {
		stop("`n_outcomes` must be at least 1.", call. = FALSE)
	}

	set.seed(seed)
	data <- data.frame(
		x = stats::rnorm(n_obs),
		age = stats::rnorm(n_obs)
	)

	adjust_names <- sprintf("z%i", seq_len(n_adjust))
	for (idx in seq_along(adjust_names)) {
		name <- adjust_names[[idx]]
		if (idx %% 4L == 0L) {
			data[[name]] <- factor(sample(c("a", "b", "c", "d"), n_obs, replace = TRUE))
		} else {
			data[[name]] <- stats::rnorm(n_obs)
		}
	}

	outcome_names <- sprintf("y%i", seq_len(n_outcomes))
	for (idx in seq_along(outcome_names)) {
		y_signal <- (0.7 + 0.1 * cos(idx / 4)) * data$x - (0.35 + 0.05 * sin(idx / 6)) * data$age
		for (adj_idx in seq_len(min(4L, n_adjust))) {
			name <- adjust_names[[adj_idx]]
			if (is.factor(data[[name]])) {
				effect_map <- c(a = -0.3, b = 0.0, c = 0.25, d = 0.5)
				y_signal <- y_signal + (0.8 + idx / (3 * n_outcomes)) * unname(effect_map[as.character(data[[name]])])
			} else {
				y_signal <- y_signal + (0.15 / adj_idx) * (1 + idx / (5 * n_outcomes)) * data[[name]]
			}
		}
		data[[outcome_names[[idx]]]] <- as.numeric(y_signal + stats::rnorm(n_obs, sd = 0.6))
	}

	list(
		data = data,
		outcome_names = outcome_names,
		base_terms = c("x", "age"),
		adjust_names = adjust_names,
		base_formula = build_multivariate_formula(outcome_names, c("x", "age")),
		adjustby = stats::as.formula(
			sprintf("~ %s", paste(adjust_names, collapse = " + "))
		)
	)
}

run_voe_multioutcome_qr <- function(config) {
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

combine_classic_fixed_results <- function(results, outcome_names) {
	vib_frames <- Map(function(result, outcome_name) {
		df <- as.data.frame(result$vibration, check.names = FALSE, stringsAsFactors = FALSE)
		df$outcome <- outcome_name
		df
	}, results, outcome_names)

	bic_frames <- Map(function(result, outcome_name) {
		df <- as.data.frame(result$bic, check.names = FALSE, stringsAsFactors = FALSE)
		df$outcome <- outcome_name
		df
	}, results, outcome_names)

	list(
		vibration = do.call(rbind, vib_frames),
		bic = do.call(rbind, bic_frames)
	)
}

combine_classic_exhaustive_results <- function(results, outcome_names) {
	vib_frames <- Map(function(result, outcome_name) {
		df <- as.data.frame(result$vibFrame, check.names = FALSE, stringsAsFactors = FALSE)
		df$outcome <- outcome_name
		df
	}, results, outcome_names)

	bic_frames <- Map(function(result, outcome_name) {
		df <- as.data.frame(result$bicFrame, check.names = FALSE, stringsAsFactors = FALSE)
		df$outcome <- outcome_name
		df
	}, results, outcome_names)

	list(
		vibFrame = do.call(rbind, vib_frames),
		bicFrame = do.call(rbind, bic_frames)
	)
}

run_voe_multioutcome_lm <- function(config) {
	rhs_terms <- config$base_terms
	results <- lapply(config$outcome_names, function(outcome_name) {
		base_formula <- single_outcome_formula(outcome_name, rhs_terms)
		if (config$is_fixed_k) {
			return(conductVibrationForK_classic(
				base_formula,
				config$data,
				config$adjustby,
				k = config$k_min,
				family = "gaussian",
				print_progress = FALSE
			))
		}

		per_k <- lapply(config$k_min:config$k_max, function(k) {
			conductVibrationForK_classic(
				base_formula,
				config$data,
				config$adjustby,
				k = k,
				family = "gaussian",
				print_progress = FALSE
			)
		})
		gatherFrames(per_k)
	})

	if (config$is_fixed_k) {
		return(combine_classic_fixed_results(results, config$outcome_names))
	}

	combine_classic_exhaustive_results(results, config$outcome_names)
}

normalize_voe_result <- function(result, is_fixed_k) {
	if (is_fixed_k) {
		vib <- as.data.frame(result$vibration, check.names = FALSE, stringsAsFactors = FALSE)
		bic <- as.data.frame(result$bic, check.names = FALSE, stringsAsFactors = FALSE)
		estimate_col <- "Estimate"
		order_cols_vib <- c("outcome", "combination_index", "factor_level")
		order_cols_bic <- c("outcome", "combination_index")
		vib_cols <- c("Estimate", "Std. Error", "t value", "Pr(>|t|)",
		             "combination_index", "factor_level", "outcome")
		bic_cols <- c("edf", "bic", "combination_index", "outcome")
	} else {
		vib <- as.data.frame(result$vibFrame, check.names = FALSE, stringsAsFactors = FALSE)
		bic <- as.data.frame(result$bicFrame, check.names = FALSE, stringsAsFactors = FALSE)
		estimate_col <- "estimate"
		order_cols_vib <- c("outcome", "k", "combination_index", "factor_level")
		order_cols_bic <- c("outcome", "k", "combination_index")
		vib_cols <- c("estimate", "se", "z", "pvalue",
		             "combination_index", "factor_level", "k", "outcome")
		bic_cols <- c("edf", "bic", "combination_index", "k", "outcome")
	}

	vib <- vib[do.call(order, vib[order_cols_vib]), , drop = FALSE]
	bic <- bic[do.call(order, bic[order_cols_bic]), , drop = FALSE]
	vib <- vib[, intersect(vib_cols, names(vib)), drop = FALSE]
	bic <- bic[, intersect(bic_cols, names(bic)), drop = FALSE]
	rownames(vib) <- NULL
	rownames(bic) <- NULL

	list(vib = vib, bic = bic, estimate_col = estimate_col)
}

validate_voe_multioutcome <- function(qr_result, lm_result, is_fixed_k, tol_estimate = 1e-8, tol_bic = 1e-8) {
	qr_norm <- normalize_voe_result(qr_result, is_fixed_k)
	lm_norm <- normalize_voe_result(lm_result, is_fixed_k)

	if (!identical(names(qr_norm$vib), names(lm_norm$vib))) {
		stop("QR and lm multi-outcome vibration columns differ.", call. = FALSE)
	}
	if (!identical(names(qr_norm$bic), names(lm_norm$bic))) {
		stop("QR and lm multi-outcome BIC columns differ.", call. = FALSE)
	}
	if (nrow(qr_norm$vib) != nrow(lm_norm$vib) || nrow(qr_norm$bic) != nrow(lm_norm$bic)) {
		stop("QR and lm multi-outcome results returned different row counts.", call. = FALSE)
	}

	vib_id_cols <- intersect(c("outcome", "k", "combination_index", "factor_level"), names(qr_norm$vib))
	bic_id_cols <- intersect(c("outcome", "k", "combination_index"), names(qr_norm$bic))
	for (name in vib_id_cols) {
		qr_col <- qr_norm$vib[[name]]
		lm_col <- lm_norm$vib[[name]]
		same <- if (is.numeric(qr_col) || is.integer(qr_col)) {
			isTRUE(all.equal(as.numeric(qr_col), as.numeric(lm_col), tolerance = 0, check.attributes = FALSE))
		} else {
			identical(as.character(qr_col), as.character(lm_col))
		}
		if (!same) {
			stop(sprintf("QR and lm multi-outcome vibration identifier `%s` differs.", name), call. = FALSE)
		}
	}
	for (name in bic_id_cols) {
		qr_col <- qr_norm$bic[[name]]
		lm_col <- lm_norm$bic[[name]]
		same <- if (is.numeric(qr_col) || is.integer(qr_col)) {
			isTRUE(all.equal(as.numeric(qr_col), as.numeric(lm_col), tolerance = 0, check.attributes = FALSE))
		} else {
			identical(as.character(qr_col), as.character(lm_col))
		}
		if (!same) {
			stop(sprintf("QR and lm multi-outcome BIC identifier `%s` differs.", name), call. = FALSE)
		}
	}

	estimate_diff <- max(abs(qr_norm$vib[[qr_norm$estimate_col]] - lm_norm$vib[[lm_norm$estimate_col]]))
	bic_diff <- max(abs(qr_norm$bic[["bic"]] - lm_norm$bic[["bic"]]))
	if (!is.finite(estimate_diff) || estimate_diff > tol_estimate) {
		stop(
			sprintf("Validation failed: max multi-outcome estimate diff %.3e exceeds tolerance %.3e",
			        estimate_diff, tol_estimate),
			call. = FALSE
		)
	}
	if (!is.finite(bic_diff) || bic_diff > tol_bic) {
		stop(
			sprintf("Validation failed: max multi-outcome BIC diff %.3e exceeds tolerance %.3e",
			        bic_diff, tol_bic),
			call. = FALSE
		)
	}

	list(max_estimate_diff = estimate_diff, max_bic_diff = bic_diff)
}

print_voe_multioutcome_summary <- function(config, qr_times, lm_times, validation) {
	qr_median <- stats::median(qr_times)
	lm_median <- stats::median(lm_times)
	k_label <- if (config$is_fixed_k) as.character(config$k_min) else sprintf("%i:%i", config$k_min, config$k_max)

	cat(sprintf("Rows              : %i\n", nrow(config$data)))
	cat(sprintf("Adjustors         : %i\n", length(config$adjust_names)))
	cat(sprintf("Outcomes          : %i\n", length(config$outcome_names)))
	cat(sprintf("Model count       : %i\n", config$n_models))
	cat(sprintf("k                 : %s\n", k_label))
	cat(sprintf("QR median seconds : %.6f\n", qr_median))
	cat(sprintf("lm median seconds : %.6f\n", lm_median))
	cat(sprintf("Speedup           : %.2fx\n", lm_median / qr_median))
	cat(sprintf("Max estimate diff : %.3e\n", validation$max_estimate_diff))
	cat(sprintf("Max BIC diff      : %.3e\n", validation$max_bic_diff))
}
