test_that("gaussian QR backend matches classic lm fits for fixed k", {
	set.seed(1)
	dat <- data.frame(
		y = rnorm(30),
		x = rnorm(30),
		z1 = rnorm(30),
		z2 = factor(sample(letters[1:3], 30, TRUE)),
		w = runif(30, 0.5, 2)
	)

	classic <- voe:::conductVibrationForK_classic(
		y ~ x, dat, ~ z1 + z2,
		k = 1, family = "gaussian",
		print_progress = FALSE,
		weights = dat$w
	)
	fast <- conductVibrationForK(
		y ~ x, dat, ~ z1 + z2,
		k = 1, family = "gaussian",
		print_progress = FALSE,
		weights = dat$w
	)

	expect_equal(fast$vibration, classic$vibration, tolerance = 1e-8)
	expect_equal(fast$bic, classic$bic, tolerance = 1e-8)
})

test_that("gaussian QR backend stays aligned with classic fits for mixed adjustors at higher k", {
	set.seed(11)
	n <- 180
	dat <- data.frame(
		y = rnorm(n),
		x = rnorm(n),
		age = rnorm(n),
		z_num1 = rnorm(n),
		z_num2 = rnorm(n),
		z_fac1 = factor(sample(letters[1:4], n, TRUE)),
		z_fac2 = factor(sample(c("low", "mid", "high"), n, TRUE)),
		z_bin1 = rbinom(n, 1, 0.4),
		z_bin2 = rbinom(n, 1, 0.3)
	)

	dat$y <- 0.8 * dat$x - 0.4 * dat$age + 0.3 * dat$z_num1 - 0.2 * dat$z_num2 +
		c(a = -0.2, b = 0.0, c = 0.25, d = 0.45)[as.character(dat$z_fac1)] +
		c(low = -0.15, mid = 0.05, high = 0.3)[as.character(dat$z_fac2)] +
		0.35 * dat$z_bin1 - 0.25 * dat$z_bin2 + rnorm(n, sd = 0.35)

	classic <- voe:::conductVibrationForK_classic(
		y ~ x + age,
		dat,
		~ z_num1 + z_num2 + z_fac1 + z_fac2 + z_bin1 + z_bin2,
		k = 4,
		family = "gaussian",
		print_progress = FALSE
	)
	fast <- conductVibrationForK(
		y ~ x + age,
		dat,
		~ z_num1 + z_num2 + z_fac1 + z_fac2 + z_bin1 + z_bin2,
		k = 4,
		family = "gaussian",
		print_progress = FALSE
	)

	expect_equal(fast$vibration, classic$vibration, tolerance = 1e-8)
	expect_equal(fast$bic, classic$bic, tolerance = 1e-8)
})

test_that("conductVibration includes the full adjustment model by default", {
	set.seed(2)
	dat <- data.frame(
		y = rnorm(25),
		x = rnorm(25),
		z1 = rnorm(25),
		z2 = rnorm(25)
	)

	res <- conductVibration(
		y ~ x, dat, ~ z1 + z2,
		family = "gaussian",
		print_progress = FALSE
	)

	expect_equal(nrow(res$bicFrame), choose(2, 1) + choose(2, 2))
	expect_true(all(c("estimate", "se", "z", "pvalue", "k") %in% colnames(res$vibFrame)))
})

test_that("gaussian QR backend reuses the same design across multiple outcomes", {
	set.seed(22)
	dat <- data.frame(
		y1 = rnorm(40),
		y2 = rnorm(40),
		x = rnorm(40),
		z1 = rnorm(40),
		z2 = factor(sample(letters[1:3], 40, TRUE)),
		w = runif(40, 0.5, 2)
	)

	multi <- conductVibration(
		cbind(y1, y2) ~ x, dat, ~ z1 + z2,
		family = "gaussian",
		print_progress = FALSE,
		weights = dat$w
	)
	single_y1 <- conductVibration(
		y1 ~ x, dat, ~ z1 + z2,
		family = "gaussian",
		print_progress = FALSE,
		weights = dat$w
	)
	single_y2 <- conductVibration(
		y2 ~ x, dat, ~ z1 + z2,
		family = "gaussian",
		print_progress = FALSE,
		weights = dat$w
	)

	expected_vib <- rbind(
		transform(single_y1$vibFrame, outcome = "y1"),
		transform(single_y2$vibFrame, outcome = "y2")
	)
	expected_bic <- rbind(
		transform(single_y1$bicFrame, outcome = "y1"),
		transform(single_y2$bicFrame, outcome = "y2")
	)

	sort_vib <- function(df) {
		df <- df[, colnames(multi$vibFrame), drop = FALSE]
		df <- df[order(df$outcome, df$k, df$combination_index, df$factor_level), , drop = FALSE]
		rownames(df) <- NULL
		df
	}
	sort_bic <- function(df) {
		df <- df[, colnames(multi$bicFrame), drop = FALSE]
		df <- df[order(df$outcome, df$k, df$combination_index), , drop = FALSE]
		rownames(df) <- NULL
		df
	}

	expect_equal(sort_vib(multi$vibFrame), sort_vib(expected_vib), tolerance = 1e-8)
	expect_equal(sort_bic(multi$bicFrame), sort_bic(expected_bic), tolerance = 1e-8)
})

test_that("gaussian QR backend falls back cleanly when rows differ across models", {
	set.seed(3)
	dat <- data.frame(
		y = rnorm(20),
		x = rnorm(20),
		z1 = rnorm(20),
		z2 = rnorm(20)
	)
	dat$z2[1] <- NA_real_

	classic <- voe:::conductVibrationForK_classic(
		y ~ x, dat, ~ z1 + z2,
		k = 1, family = "gaussian",
		print_progress = FALSE
	)
	res <- conductVibrationForK(
		y ~ x, dat, ~ z1 + z2,
		k = 1, family = "gaussian",
		print_progress = FALSE
	)

	expect_equal(res$vibration, classic$vibration, tolerance = 1e-8)
	expect_equal(res$bic, classic$bic, tolerance = 1e-8)
})

test_that("binomial VoE returns harmonized coefficient columns", {
	set.seed(4)
	dat <- data.frame(
		y = rbinom(40, 1, 0.5),
		x = rnorm(40),
		z1 = rnorm(40)
	)

	res <- conductVibration(
		y ~ x, dat, ~ z1,
		family = "binomial",
		print_progress = FALSE
	)

	expect_equal(nrow(res$bicFrame), 1L)
	expect_true(all(c("estimate", "se", "z", "pvalue", "HR", "k") %in% colnames(res$vibFrame)))
})
