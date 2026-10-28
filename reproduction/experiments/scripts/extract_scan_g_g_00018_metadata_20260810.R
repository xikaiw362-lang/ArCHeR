args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: Rscript extract_scan_g_g_00018_metadata_20260810.R INPUT_RDS OUTPUT_DIR")
}

input_rds <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- args[[2]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

obj <- readRDS(input_rds)
if (!inherits(obj, "Seurat")) stop("Input is not a Seurat object")
metadata <- obj@meta.data
if (!"shoudong" %in% colnames(metadata)) stop("Missing shoudong metadata column")
if (!"umap" %in% names(obj@reductions)) stop("Missing UMAP reduction")

embedding <- obj@reductions[["umap"]]@cell.embeddings
if (!identical(rownames(metadata), rownames(embedding))) {
  stop("Metadata and UMAP barcode orders differ")
}

metadata_out <- data.frame(
  barcode = rownames(metadata),
  metadata,
  check.names = FALSE,
  stringsAsFactors = FALSE
)
umap_out <- data.frame(
  barcode = rownames(embedding),
  SCAN_UMAP1 = embedding[, 1],
  SCAN_UMAP2 = embedding[, 2],
  shoudong = as.character(metadata$shoudong),
  check.names = FALSE,
  stringsAsFactors = FALSE
)

write.csv(metadata_out, file.path(output_dir, "G_G_00018_SCAN_cell_metadata_20260810.csv"), row.names = FALSE)
write.csv(umap_out, file.path(output_dir, "G_G_00018_SCAN_UMAP_shoudong_20260810.csv"), row.names = FALSE)

cat("cells=", nrow(metadata_out), "\n", sep = "")
cat("shoudong_levels=", length(unique(metadata_out$shoudong)), "\n", sep = "")
print(sort(table(metadata_out$shoudong), decreasing = TRUE))
