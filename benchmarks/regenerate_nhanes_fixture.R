#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
if (!length(file_arg)) {
	stop("Unable to determine script path.", call. = FALSE)
}

script_path <- normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = TRUE)
bench_dir <- dirname(script_path)
repo_root <- normalizePath(file.path(bench_dir, ".."), winslash = "/", mustWork = TRUE)

source_path <- file.path(repo_root, "data", "nhanes9904_VoE.Rdata")
output_dir <- file.path(bench_dir, "data")
output_path <- file.path(output_dir, "nhanes_voe_gaussian.csv")

if (!file.exists(source_path)) {
	stop(sprintf("Expected source dataset at %s", source_path), call. = FALSE)
}

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

load(source_path)
if (!exists("mainTab")) {
	stop("`mainTab` was not found after loading nhanes9904_VoE.Rdata.", call. = FALSE)
}

required_cols <- c(
	"LBXVID",
	"LBXBCD",
	"RIDAGEYR",
	"male",
	"SES_LEVEL",
	"current_past_smoking",
	"education",
	"RIDRETH1",
	"any_cad",
	"any_ht",
	"any_diabetes"
)

missing_cols <- setdiff(required_cols, names(mainTab))
if (length(missing_cols)) {
	stop(
		sprintf("Missing expected NHANES columns: %s", paste(missing_cols, collapse = ", ")),
		call. = FALSE
	)
}

bench_df <- mainTab[, required_cols, drop = FALSE]
bench_df$LBXBCD_log_z <- as.numeric(scale(log(bench_df$LBXBCD)))
bench_df <- bench_df[, c(
	"LBXVID",
	"LBXBCD_log_z",
	"RIDAGEYR",
	"male",
	"SES_LEVEL",
	"current_past_smoking",
	"education",
	"RIDRETH1",
	"any_cad",
	"any_ht",
	"any_diabetes"
)]

bench_df <- bench_df[stats::complete.cases(bench_df), , drop = FALSE]
rownames(bench_df) <- NULL

utils::write.csv(bench_df, output_path, row.names = FALSE)

message(sprintf("Wrote %s", output_path))
message(sprintf("Rows: %i", nrow(bench_df)))
message(sprintf("Columns: %i", ncol(bench_df)))
