suppressPackageStartupMessages(library(Seurat))
suppressPackageStartupMessages(library(Matrix))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("usage: Rscript prepare_s_s_00005_case.R OUT_DIR N_CELLS SEED [RDS_PATH]")
}
out_dir <- normalizePath(args[[1]], mustWork = FALSE)
n_cells_req <- as.integer(args[[2]])
seed <- as.integer(args[[3]])
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

rds_path <- if (length(args) >= 4) args[[4]] else "D:/sc_dataset/case_study_datasets/S_S_00005/S_S_00005.rds"
if (!file.exists(rds_path)) stop("RDS input does not exist: ", rds_path)
x <- readRDS(rds_path)
counts <- GetAssayData(x, assay = "RNA", layer = "counts")
meta <- x@meta.data

set.seed(seed)
n_total <- ncol(counts)
if (is.na(n_cells_req) || n_cells_req <= 0 || n_cells_req >= n_total) {
  keep <- seq_len(n_total)
} else {
  keep <- sort(sample(seq_len(n_total), size = n_cells_req, replace = FALSE))
}

# Store cells x genes as a CSC matrix, using the same raw RNA counts for both
# methods.  The binary slots avoid the much larger text MatrixMarket format.
mat <- as(t(counts[, keep, drop = FALSE]), "dgCMatrix")
writeBin(as.integer(dim(mat)), file.path(out_dir, "shape.i32"), size = 4, endian = "little")
writeBin(as.integer(mat@p), file.path(out_dir, "indptr.i32"), size = 4, endian = "little")
writeBin(as.integer(mat@i), file.path(out_dir, "indices.i32"), size = 4, endian = "little")
writeBin(as.numeric(mat@x), file.path(out_dir, "data.f32"), size = 4, endian = "little")
writeLines(rownames(mat), file.path(out_dir, "cells.tsv"), useBytes = TRUE)
writeLines(colnames(mat), file.path(out_dir, "genes.tsv"), useBytes = TRUE)
write.csv(meta[keep, , drop = FALSE], file.path(out_dir, "metadata.csv"), quote = TRUE)
write.csv(data.frame(original_index = keep, cell_id = rownames(mat)),
          file.path(out_dir, "sampling_map.csv"), row.names = FALSE, quote = TRUE)
writeLines(normalizePath(rds_path), file.path(out_dir, "source_rds_path.txt"), useBytes = TRUE)

cat("prepared_cells=", nrow(mat), " prepared_genes=", ncol(mat),
    " nnzero=", length(mat@x), " seed=", seed, "\n", sep = "")
