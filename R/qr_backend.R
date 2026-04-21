# Internal QR backend for gaussian VoE models.
# This reuses a shared model matrix when all candidate models share the same rows.

adjustment_terms <- function(adjustby) {
	if (inherits(adjustby, "formula")) {
		return(attr(stats::terms(adjustby), "term.labels"))
	}
	as.character(adjustby)
}

add_adjustors_to_formula <- function(base_formula, adjustingVariables) {
	form <- base_formula
	if (length(adjustingVariables)) {
		addStr <- stats::as.formula(sprintf("~ . + %s", paste(adjustingVariables, collapse = "+")))
		form <- stats::update.formula(base_formula, addStr)
	}
	form
}

combination_matrix <- function(n, k) {
	if (k < 0L || k > n) {
		stop(sprintf("k must be between 0 and %i", n), call. = FALSE)
	}
	if (k == 0L) {
		return(matrix(integer(0), nrow = 0L, ncol = 1L))
	}
	utils::combn(n, k)
}

gaussian_bic_from_qr <- function(rss, n_obs, edf) {
	c(edf = edf, bic = n_obs * log(rss / n_obs) + log(n_obs) * edf)
}

find_exposure_rows <- function(coef_names, exposure_term, exposure_vars) {
	rowIndex <- integer(0)
	if (!is.null(exposure_term) && nzchar(exposure_term)) {
		rowIndex <- grep(exposure_term, coef_names, fixed = TRUE)
	}
	if (!length(rowIndex) && length(exposure_vars)) {
		rowIndex <- unique(unlist(lapply(exposure_vars, function(varname) {
			grep(varname, coef_names, fixed = TRUE)
		}), use.names = FALSE))
	}
	rowIndex
}

prepare_gaussian_qr_context <- function(base_formula, dataFrame, adjustby,
										 dots = list()) {
	supported_args <- c("weights", "subset", "offset", "contrasts", "na.action")
	dot_names <- names(dots)
	if (length(dots) && (is.null(dot_names) || any(!nzchar(dot_names)))) {
		return(list(
			supported = FALSE,
			reason = "QR backend requires named arguments in `...`."
		))
	}

	unsupported_args <- setdiff(dot_names, supported_args)
	if (length(unsupported_args)) {
		return(list(
			supported = FALSE,
			reason = sprintf(
				"QR backend does not support these lm arguments yet: %s",
				paste(sort(unsupported_args), collapse = ", ")
			)
		))
	}

	adjust_terms <- adjustment_terms(adjustby)
	full_formula <- add_adjustors_to_formula(base_formula, adjust_terms)

	mf_args <- list(
		formula = full_formula,
		data = dataFrame,
		drop.unused.levels = TRUE,
		na.action = stats::na.pass
	)
	for (arg_name in intersect(supported_args, dot_names)) {
		if (arg_name != "contrasts") {
			mf_args[[arg_name]] <- dots[[arg_name]]
		}
	}

	full_mf <- tryCatch(
		do.call(stats::model.frame, mf_args),
		error = function(err) err
	)
	if (inherits(full_mf, "error")) {
		return(list(supported = FALSE, reason = conditionMessage(full_mf)))
	}

	if (!all(stats::complete.cases(full_mf))) {
		return(list(
			supported = FALSE,
			reason = paste(
				"QR backend needs a shared complete-case dataset across the",
				"base model and all candidate adjustors."
			)
		))
	}

	y <- stats::model.response(full_mf)
	is_multi_outcome <- is.matrix(y) && is.numeric(y)
	if (!is.numeric(y) || (is_multi_outcome && is.null(ncol(y)))) {
		return(list(
			supported = FALSE,
			reason = "QR backend currently supports numeric gaussian responses only."
		))
	}

	offset <- stats::model.offset(full_mf)
	if (is_multi_outcome && !is.null(offset)) {
		return(list(
			supported = FALSE,
			reason = "QR backend does not support offsets with multivariate gaussian responses."
		))
	}
	if (is.null(offset)) {
		offset <- rep(0, if (is_multi_outcome) nrow(y) else length(y))
	}

	weights <- stats::model.weights(full_mf)
	if (is.null(weights)) {
		weights <- rep(1, if (is_multi_outcome) nrow(y) else length(y))
	}
	if (!is.numeric(weights) || any(weights < 0)) {
		return(list(
			supported = FALSE,
			reason = "QR backend requires non-negative numeric weights."
		))
	}

	full_terms <- attr(full_mf, "terms")
	contrasts_arg <- dots$contrasts
	X <- tryCatch(
		stats::model.matrix(full_terms, full_mf, contrasts.arg = contrasts_arg),
		error = function(err) err
	)
	if (inherits(X, "error")) {
		return(list(supported = FALSE, reason = conditionMessage(X)))
	}

	sqrt_weights <- sqrt(weights)
	X_weighted <- X * sqrt_weights
	if (is_multi_outcome) {
		Y_weighted <- sweep(y, 1L, offset, "-", check.margin = FALSE)
		Y_weighted <- sweep(Y_weighted, 1L, sqrt_weights, "*", check.margin = FALSE)
		outcome_names <- colnames(y)
		if (is.null(outcome_names)) {
			outcome_names <- paste0("outcome_", seq_len(ncol(y)))
		}
	} else {
		y_weighted <- (y - offset) * sqrt_weights
		outcome_names <- NULL
	}

	if (qr(X_weighted)$rank < ncol(X_weighted)) {
		return(list(
			supported = FALSE,
			reason = paste(
				"The full gaussian design matrix is rank deficient, so the QR",
				"backend falls back to per-model lm fits."
			)
		))
	}

	term_labels_full <- attr(full_terms, "term.labels")
	base_terms <- attr(stats::terms(base_formula), "term.labels")
	base_positions <- match(base_terms, term_labels_full)
	adjust_positions <- match(adjust_terms, term_labels_full)

	if (anyNA(base_positions) || anyNA(adjust_positions)) {
		return(list(
			supported = FALSE,
			reason = "Could not align model-matrix terms with the requested adjustors."
		))
	}

	assign_vec <- attr(X, "assign")
	fixed_col_indices <- which(assign_vec == 0L | assign_vec %in% base_positions)
	adjust_groups <- lapply(adjust_positions, function(term_pos) {
		col_idx <- which(assign_vec == term_pos)
		list(
			col_indices = col_idx,
			coef_names = colnames(X)[col_idx]
		)
	})

	exposure_term <- base_terms[1]
	exposure_vars <- if (length(base_terms)) {
		all.vars(stats::as.formula(sprintf("~%s", exposure_term)))
	} else {
		character(0)
	}

	list(
		supported = TRUE,
		adjust = adjust_terms,
		base_formula = base_formula,
		n_obs = nrow(X_weighted),
		is_multi_outcome = is_multi_outcome,
		y_weighted = if (is_multi_outcome) NULL else as.numeric(y_weighted),
		Y_weighted = if (is_multi_outcome) Y_weighted else NULL,
		outcome_names = outcome_names,
		X_weighted = X_weighted,
		fixed_col_indices = fixed_col_indices,
		fixed_coef_names = colnames(X)[fixed_col_indices],
		adjust_groups = adjust_groups,
		exposure_term = exposure_term,
		exposure_vars = exposure_vars
	)
}

givens_params <- function(a, b) {
	if (b == 0) {
		return(list(c = 1, s = 0, r = a))
	}
	if (a == 0) {
		return(list(c = 0, s = 1, r = b))
	}
	r <- sqrt(a * a + b * b)
	list(c = a / r, s = b / r, r = r)
}

apply_givens_rows <- function(M, j, cs, ss, col_start = 1L) {
	ncols <- ncol(M)
	if (col_start > ncols) {
		return(M)
	}
	cols <- col_start:ncols
	tmp <- cs * M[j, cols] + ss * M[j + 1L, cols]
	M[j + 1L, cols] <- -ss * M[j, cols] + cs * M[j + 1L, cols]
	M[j, cols] <- tmp
	M
}

apply_givens_cols <- function(Q, j, cs, ss) {
	tmp <- cs * Q[, j] + ss * Q[, j + 1L]
	Q[, j + 1L] <- -ss * Q[, j] + cs * Q[, j + 1L]
	Q[, j] <- tmp
	Q
}

qr_state_from_cols <- function(X_data, col_indices, tol = 1e-10) {
	if (length(col_indices) == 0L) {
		m <- nrow(X_data)
		return(list(
			Q = matrix(0, m, 0L),
			R = matrix(0, 0L, 0L),
			col_indices = integer(0L),
			m = m,
			p = 0L,
			tol = tol
		))
	}

	X_sub <- X_data[, col_indices, drop = FALSE]
	m <- nrow(X_sub)
	p <- ncol(X_sub)
	qr_obj <- qr(X_sub)
	Q <- qr.Q(qr_obj)[, seq_len(p), drop = FALSE]
	R <- qr.R(qr_obj)[seq_len(p), seq_len(p), drop = FALSE]

	signs <- sign(diag(R))
	signs[signs == 0] <- 1
	Q <- Q %*% diag(signs, p, p)
	R <- diag(signs, p, p) %*% R

	list(
		Q = Q,
		R = R,
		col_indices = as.integer(col_indices),
		m = m,
		p = p,
		tol = tol
	)
}

qr_add_column <- function(state, z, col_idx) {
	p <- state$p
	m <- state$m
	tol <- state$tol

	if (p == 0L) {
		norm_z <- sqrt(sum(z^2))
		if (norm_z < tol) {
			warning("Near-rank-deficient column encountered; clamping norm.", call. = FALSE)
			norm_z <- tol
		}
		q_new <- z / norm_z
		return(list(
			Q = matrix(q_new, m, 1L),
			R = matrix(norm_z, 1L, 1L),
			col_indices = as.integer(col_idx),
			m = m,
			p = 1L,
			tol = tol
		))
	}

	p_vec <- crossprod(state$Q, z)
	residual <- z - state$Q %*% p_vec
	norm_r <- sqrt(sum(residual^2))

	if (norm_r < tol) {
		warning("Near-rank-deficient column encountered; clamping norm.", call. = FALSE)
		norm_r <- tol
	}

	q_new <- residual / norm_r
	Q_new <- cbind(state$Q, q_new)
	R_new <- rbind(
		cbind(state$R, p_vec),
		c(rep(0, p), norm_r)
	)

	list(
		Q = Q_new,
		R = R_new,
		col_indices = c(state$col_indices, as.integer(col_idx)),
		m = m,
		p = p + 1L,
		tol = tol
	)
}

qr_remove_column <- function(state, pos) {
	p <- state$p
	if (p == 0L) {
		stop("Cannot remove a column from an empty QR state.", call. = FALSE)
	}
	if (pos < 1L || pos > p) {
		stop("pos is out of range in qr_remove_column.", call. = FALSE)
	}

	if (p == 1L) {
		return(list(
			Q = matrix(0, state$m, 0L),
			R = matrix(0, 0L, 0L),
			col_indices = integer(0L),
			m = state$m,
			p = 0L,
			tol = state$tol
		))
	}

	Q <- state$Q
	R_work <- state$R[, -pos, drop = FALSE]
	if (pos <= p - 1L) {
		for (j in seq.int(pos, p - 1L)) {
			gp <- givens_params(R_work[j, j], R_work[j + 1L, j])
			R_work <- apply_givens_rows(R_work, j, gp$c, gp$s, col_start = j)
			Q <- apply_givens_cols(Q, j, gp$c, gp$s)
		}
	}

	list(
		Q = Q[, seq_len(p - 1L), drop = FALSE],
		R = R_work[seq_len(p - 1L), , drop = FALSE],
		col_indices = state$col_indices[-pos],
		m = state$m,
		p = p - 1L,
		tol = state$tol
	)
}

qr_new_outcome <- function(state, y) {
	p <- state$p
	if (p == 0L) {
		m <- state$m
		rss <- sum(y^2)
		return(list(
			beta = numeric(0L),
			fitted = rep(0, m),
			residuals = y,
			rss = rss
		))
	}

	Qty <- crossprod(state$Q, y)
	beta <- backsolve(state$R, Qty)
	fitted <- state$Q %*% Qty
	residuals <- y - fitted

	list(
		beta = as.numeric(beta),
		fitted = as.numeric(fitted),
		residuals = as.numeric(residuals),
		rss = sum(residuals^2)
	)
}

qr_batch_outcomes <- function(state, Y) {
	p <- state$p
	q <- ncol(Y)
	m <- state$m
	if (p == 0L) {
		return(list(
			beta = matrix(0, 0L, q),
			fitted = matrix(0, m, q),
			residuals = Y,
			rss = colSums(Y^2)
		))
	}

	QtY <- crossprod(state$Q, Y)
	beta <- backsolve(state$R, QtY)
	fitted <- state$Q %*% QtY
	residuals <- Y - fitted

	list(
		beta = beta,
		fitted = fitted,
		residuals = residuals,
		rss = colSums(residuals^2)
	)
}

compute_se <- function(R, sigma) {
	p <- nrow(R)
	if (p == 0L) {
		if (length(sigma) == 1L) {
			return(numeric(0L))
		}
		return(matrix(0, 0L, length(sigma)))
	}
	R_inv <- backsolve(R, diag(p))
	col_sd <- sqrt(rowSums(R_inv^2))
	if (length(sigma) == 1L) {
		return(col_sd * sigma)
	}
	outer(col_sd, sigma)
}

update_gaussian_qr_state <- function(context, state, coef_names,
										 previous_terms, current_terms) {
	remove_terms <- setdiff(previous_terms, current_terms)
	add_terms <- setdiff(current_terms, previous_terms)

	if (length(remove_terms)) {
		remove_positions <- integer(0)
		for (term_idx in remove_terms) {
			term_cols <- context$adjust_groups[[term_idx]]$col_indices
			remove_positions <- c(remove_positions, match(term_cols, state$col_indices))
		}
		remove_positions <- sort(remove_positions, decreasing = TRUE)
		for (pos in remove_positions) {
			state <- qr_remove_column(state, pos)
			coef_names <- coef_names[-pos]
		}
	}

	if (length(add_terms)) {
		for (term_idx in add_terms) {
			term_group <- context$adjust_groups[[term_idx]]
			for (col_pos in seq_along(term_group$col_indices)) {
				col_idx <- term_group$col_indices[col_pos]
				state <- qr_add_column(state, context$X_weighted[, col_idx], col_idx)
				coef_names <- c(coef_names, term_group$coef_names[col_pos])
			}
		}
	}

	list(state = state, coef_names = coef_names)
}

fit_gaussian_qr_model <- function(context, state, coef_names) {
	fit <- qr_new_outcome(state, context$y_weighted)
	n_params <- state$p
	df_res <- context$n_obs - n_params

	se <- rep(NA_real_, n_params)
	t_stat <- rep(NA_real_, n_params)
	pvalue <- rep(NA_real_, n_params)
	if (n_params > 0L && df_res > 0L) {
		sigma <- sqrt(fit$rss / df_res)
		se <- compute_se(state$R, sigma)
		t_stat <- fit$beta / se
		pvalue <- 2 * stats::pt(-abs(t_stat), df = df_res)
	}

	coef_table <- cbind(
		"Estimate" = fit$beta,
		"Std. Error" = se,
		"t value" = t_stat,
		"Pr(>|t|)" = pvalue
	)
	rownames(coef_table) <- coef_names

	list(
		coef_table = coef_table,
		bic = gaussian_bic_from_qr(fit$rss, context$n_obs, n_params)
	)
}

fit_gaussian_qr_model_multi <- function(context, state, coef_names) {
	batch <- qr_batch_outcomes(state, context$Y_weighted)
	n_params <- state$p
	df_res <- context$n_obs - n_params
	n_outcomes <- ncol(context$Y_weighted)

	se <- matrix(NA_real_, nrow = n_params, ncol = n_outcomes)
	t_stat <- matrix(NA_real_, nrow = n_params, ncol = n_outcomes)
	pvalue <- matrix(NA_real_, nrow = n_params, ncol = n_outcomes)
	if (n_params > 0L && df_res > 0L) {
		sigma <- sqrt(batch$rss / df_res)
		se <- compute_se(state$R, sigma)
		t_stat <- batch$beta / se
		pvalue <- 2 * stats::pt(-abs(t_stat), df = df_res)
	}

	coef_tables <- lapply(seq_len(n_outcomes), function(ii) {
		coef_table <- cbind(
			"Estimate" = batch$beta[, ii],
			"Std. Error" = se[, ii],
			"t value" = t_stat[, ii],
			"Pr(>|t|)" = pvalue[, ii]
		)
		rownames(coef_table) <- coef_names
		coef_table
	})
	names(coef_tables) <- context$outcome_names

	bic_frame <- data.frame(
		edf = rep(n_params, n_outcomes),
		bic = vapply(batch$rss, function(rss) {
			unname(gaussian_bic_from_qr(rss, context$n_obs, n_params)["bic"])
		}, numeric(1L)),
		outcome = context$outcome_names,
		stringsAsFactors = FALSE
	)

	list(
		coef_tables = coef_tables,
		bic_frame = bic_frame
	)
}

assemble_fixed_k_result <- function(model_results, coef_colnames, k, varComb,
										 family, base_formula, adjust_terms) {
	total_rows <- sum(vapply(model_results, function(x) x$n_levels, integer(1)))
	vibFrame <- matrix(NA, nrow = total_rows, ncol = length(coef_colnames) + 2)
	colnames(vibFrame) <- c(coef_colnames, "combination_index", "factor_level")

	bicFrame <- matrix(NA, nrow = length(model_results), ncol = 3)
	colnames(bicFrame) <- c("edf", "bic", "combination_index")

	row_counter <- 1L
	for (ii in seq_along(model_results)) {
		model_result <- model_results[[ii]]
		bicFrame[ii, "edf"] <- model_result$bic_edf
		bicFrame[ii, "bic"] <- model_result$bic_val
		bicFrame[ii, "combination_index"] <- model_result$combo_idx

		for (jj in seq_len(model_result$n_levels)) {
			vibFrame[row_counter, seq_along(coef_colnames)] <- model_result$coefs[jj, ]
			vibFrame[row_counter, length(coef_colnames) + 1L] <- model_result$combo_idx
			vibFrame[row_counter, length(coef_colnames) + 2L] <- jj
			row_counter <- row_counter + 1L
		}
	}

	list(
		vibration = vibFrame,
		bic = bicFrame,
		k = k,
		combinations = varComb,
		family = family,
		base_formula = base_formula,
		adjust = adjust_terms
	)
}

assemble_fixed_k_result_multi <- function(vib_rows, bic_rows, k, varComb,
											  family, base_formula, adjust_terms) {
	list(
		vibration = do.call(rbind, vib_rows),
		bic = do.call(rbind, bic_rows),
		k = k,
		combinations = varComb,
		family = family,
		base_formula = base_formula,
		adjust = adjust_terms
	)
}

conductVibrationForK_gaussian_qr_prepared <- function(context, k,
													  print_progress = TRUE) {
	if (isTRUE(context$is_multi_outcome)) {
		return(conductVibrationForK_gaussian_qr_multi_prepared(context, k, print_progress))
	}
	n_adjust <- length(context$adjust_groups)
	varComb <- combination_matrix(n_adjust, k)
	n_models <- ncol(varComb)

	if (print_progress) {
		cat(sprintf("using QR backend for gaussian models; %i combinations at k=%i\n", n_models, k))
	}

	model_results <- vector("list", n_models)
	previous_terms <- integer(0)
	state <- qr_state_from_cols(context$X_weighted, context$fixed_col_indices)
	coef_names <- context$fixed_coef_names
	coef_colnames <- NULL

	for (ii in seq_len(n_models)) {
		current_terms <- if (k == 0L) integer(0) else varComb[, ii]
		updated <- update_gaussian_qr_state(
			context, state, coef_names, previous_terms, current_terms
		)
		state <- updated$state
		coef_names <- updated$coef_names
		previous_terms <- current_terms

		if (print_progress && (ii == 1L || ii %% 100L == 0L || ii == n_models)) {
			cat(sprintf("%i/%i\n", ii, n_models))
		}

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

conductVibrationForK_gaussian_qr_multi_prepared <- function(context, k,
															print_progress = TRUE) {
	n_adjust <- length(context$adjust_groups)
	varComb <- combination_matrix(n_adjust, k)
	n_models <- ncol(varComb)

	if (print_progress) {
		cat(sprintf(
			"using QR backend for gaussian models; %i combinations at k=%i across %i outcomes\n",
			n_models, k, length(context$outcome_names)
		))
	}

	previous_terms <- integer(0)
	state <- qr_state_from_cols(context$X_weighted, context$fixed_col_indices)
	coef_names <- context$fixed_coef_names
	vib_rows <- list()
	bic_rows <- list()

	for (ii in seq_len(n_models)) {
		current_terms <- if (k == 0L) integer(0) else varComb[, ii]
		updated <- update_gaussian_qr_state(
			context, state, coef_names, previous_terms, current_terms
		)
		state <- updated$state
		coef_names <- updated$coef_names
		previous_terms <- current_terms

		if (print_progress && (ii == 1L || ii %% 100L == 0L || ii == n_models)) {
			cat(sprintf("%i/%i\n", ii, n_models))
		}

		model_fit <- fit_gaussian_qr_model_multi(context, state, coef_names)
		for (outcome_idx in seq_along(model_fit$coef_tables)) {
			coef_table <- model_fit$coef_tables[[outcome_idx]]
			rowIndex <- find_exposure_rows(
				rownames(coef_table),
				context$exposure_term,
				context$exposure_vars
			)
			if (!length(rowIndex)) {
				next
			}

			vib_rows[[length(vib_rows) + 1L]] <- data.frame(
				coef_table[rowIndex, , drop = FALSE],
				combination_index = ii,
				factor_level = seq_along(rowIndex),
				outcome = names(model_fit$coef_tables)[outcome_idx],
				check.names = FALSE,
				stringsAsFactors = FALSE
			)

			bic_rows[[length(bic_rows) + 1L]] <- data.frame(
				model_fit$bic_frame[outcome_idx, , drop = FALSE],
				combination_index = ii,
				check.names = FALSE,
				stringsAsFactors = FALSE
			)
		}
	}

	if (!length(vib_rows)) {
		return(NULL)
	}

	assemble_fixed_k_result_multi(
		vib_rows = vib_rows,
		bic_rows = bic_rows,
		k = k,
		varComb = varComb,
		family = "gaussian",
		base_formula = context$base_formula,
		adjust_terms = context$adjust
	)
}
