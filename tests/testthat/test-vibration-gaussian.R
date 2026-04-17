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
