#!/usr/bin/env Rscript

source("../../../.Rprofile", chdir = TRUE)
suppressPackageStartupMessages({
    library(DropSeq.util)
    library(Matrix)
})

dge <- loadSparseDge("F_GRCm38.81.P60Cortex_noRep5_FRONTALonly.raw.dge.txt.gz")
outcomes <- readRDS("F_GRCm38.81.P60Cortex_noRep5_FRONTALonly.cell_cluster_outcomes.RDS")
outcomes <- outcomes[!is.na(outcomes$cluster), c("cluster", "subcluster")]
dge <- dge[, rownames(outcomes)]

writeMM(dge, "F_GRCm38.81.P60Cortex_noRep5_FRONTALonly.raw.dge.mtx")
write.table(rownames(dge), "F_GRCm38.81.P60Cortex_noRep5_FRONTALonly.raw.dge.rownames", quote = FALSE, row.names = FALSE, col.names = FALSE)
write.table(colnames(dge), "F_GRCm38.81.P60Cortex_noRep5_FRONTALonly.raw.dge.colnames", quote = FALSE, row.names = FALSE, col.names = FALSE)
write.csv(outcomes, "F_GRCm38.81.P60Cortex_noRep5_FRONTALonly.cell_cluster_outcomes.csv", quote = FALSE)
