#!/usr/bin/env Rscript

args_full <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args_full, value = TRUE)
if (!length(file_arg)) {
	stop("Unable to determine script path.", call. = FALSE)
}

parse_args <- function(args) {
	opts <- list(
		r_csv = "",
		python_csv = "",
		x_col = "",
		r_native_col = "lm_seconds",
		r_qr_col = "qr_seconds",
		python_native_col = "native_seconds",
		python_qr_col = "qr_seconds",
		x_label = "",
		title = "",
		output_csv = "",
		output_plot = "",
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
		opts[[key]] <- value
	}
	opts
}

print_help <- function() {
	cat(
		paste(
			"Usage:",
			"  Rscript benchmarks/plot_cross_language_4line.R [options]",
			"",
			"Options:",
			"  --r-csv PATH",
			"  --python-csv PATH",
			"  --x-col NAME",
			"  --r-native-col NAME",
			"  --r-qr-col NAME",
			"  --python-native-col NAME",
			"  --python-qr-col NAME",
			"  --x-label TEXT",
			"  --title TEXT",
			"  --output-csv PATH",
			"  --output-plot PATH",
			"  --help",
			sep = "\n"
		)
	)
}

opts <- parse_args(commandArgs(trailingOnly = TRUE))
if (isTRUE(opts$help)) {
	print_help()
	quit(save = "no", status = 0L)
}

required <- c("r_csv", "python_csv", "x_col", "output_csv", "output_plot")
missing <- required[!nzchar(unlist(opts[required]))]
if (length(missing)) {
	stop(sprintf("Missing required options: %s", paste(missing, collapse = ", ")), call. = FALSE)
}

r_data <- utils::read.csv(opts$r_csv, stringsAsFactors = FALSE)
python_data <- utils::read.csv(opts$python_csv, stringsAsFactors = FALSE)

if (!all(c(opts$x_col, opts$r_native_col, opts$r_qr_col) %in% names(r_data))) {
	stop("R CSV does not contain the requested columns.", call. = FALSE)
}
if (!all(c(opts$x_col, opts$python_native_col, opts$python_qr_col) %in% names(python_data))) {
	stop("Python CSV does not contain the requested columns.", call. = FALSE)
}

combined <- rbind(
	data.frame(
		x = r_data[[opts$x_col]],
		seconds = r_data[[opts$r_native_col]],
		implementation = "Native R",
		stringsAsFactors = FALSE
	),
	data.frame(
		x = r_data[[opts$x_col]],
		seconds = r_data[[opts$r_qr_col]],
		implementation = "QR-update R",
		stringsAsFactors = FALSE
	),
	data.frame(
		x = python_data[[opts$x_col]],
		seconds = python_data[[opts$python_native_col]],
		implementation = "Native Python",
		stringsAsFactors = FALSE
	),
	data.frame(
		x = python_data[[opts$x_col]],
		seconds = python_data[[opts$python_qr_col]],
		implementation = "QR-update Python",
		stringsAsFactors = FALSE
	)
)

utils::write.csv(combined, opts$output_csv, row.names = FALSE)
cat(sprintf("Wrote %s\n", opts$output_csv))

line_info <- list(
	"Native R" = list(col = "#C44E52", pch = 16, lty = 1),
	"QR-update R" = list(col = "#4C78A8", pch = 17, lty = 1),
	"Native Python" = list(col = "#DD8452", pch = 15, lty = 2),
	"QR-update Python" = list(col = "#55A868", pch = 18, lty = 2)
)

x_label <- if (nzchar(opts$x_label)) opts$x_label else opts$x_col
plot_title <- if (nzchar(opts$title)) opts$title else "Cross-language runtime comparison"

grDevices::png(opts$output_plot, width = 1200, height = 720, res = 150)
on.exit(grDevices::dev.off(), add = TRUE)

plot(
	range(combined$x),
	range(combined$seconds),
	type = "n",
	xlab = x_label,
	ylab = "Seconds",
	main = plot_title
)
grid()

for (name in names(line_info)) {
	subset <- combined[combined$implementation == name, , drop = FALSE]
	subset <- subset[order(subset$x), , drop = FALSE]
	info <- line_info[[name]]
	lines(subset$x, subset$seconds, col = info$col, lwd = 2, lty = info$lty)
	points(subset$x, subset$seconds, col = info$col, pch = info$pch, cex = 1.1)
}

legend(
	"topleft",
	legend = names(line_info),
	col = vapply(line_info, `[[`, character(1L), "col"),
	pch = vapply(line_info, `[[`, numeric(1L), "pch"),
	lty = vapply(line_info, `[[`, numeric(1L), "lty"),
	lwd = 2,
	bty = "n"
)

cat(sprintf("Wrote %s\n", opts$output_plot))
