suppressMessages({library(Matrix)})
o <- readRDS('D:/sc_dataset/case_study_datasets/G_G_00018/source_scan_seurat_20260810/G_G_00018.rds')
cat(class(o), '\n')
a <- o[['RNA']]
m <- tryCatch(a$counts, error=function(e) tryCatch(a@layers$counts, error=function(e) a@counts))
if (is.null(rownames(m))) { rownames(m) <- rownames(o); colnames(m) <- colnames(o) }
cat(dim(m), '\n')
dir <- 'D:/sc_dataset/paper/ArCHeR_figures_readable_20261026/.cache_gg18_counts/'
dir.create(dir, showWarnings=FALSE)
writeMM(m, paste0(dir,'counts.mtx')); writeLines(rownames(m), paste0(dir,'genes.txt')); writeLines(colnames(m), paste0(dir,'cells.txt'))
