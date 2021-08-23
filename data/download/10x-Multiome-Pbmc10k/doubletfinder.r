suppressPackageStartupMessages({
    source("../../../.Rprofile", chdir = TRUE)
    library(ggplot2)
    library(Seurat)
    library(DoubletFinder)
})

inputdata.10x <- Read10X_h5("pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5")
meta.data <- read.csv("wnn_meta_data.csv", row.names = 1)

rna.counts <- inputdata.10x$`Gene Expression`[, rownames(meta.data)]
rna.so <- CreateSeuratObject(counts = rna.counts, assay = "RNA", meta.data = meta.data)

rna.so <- NormalizeData(rna.so)

rna.so <- FindVariableFeatures(rna.so, selection.method = "vst", nfeatures = 2000)

rna.so <- ScaleData(rna.so)

rna.so <- RunPCA(rna.so, npcs = 30, verbose=FALSE)

rna.so <- RunUMAP(rna.so, dims = 1:30)

sweep.res.list <- paramSweep_v3(rna.so, PCs = 1:30, sct = FALSE)

sweep.stats <- summarizeSweep(sweep.res.list, GT = FALSE)

bcmvn <- find.pK(sweep.stats)

options(repr.plot.width=12, repr.plot.height=6)
ggplot(data=bcmvn, mapping=aes(x=pK, y=BCmetric, group=1)) + geom_point() + geom_line()

annotations <- rna.so@meta.data$celltype
homotypic.prop <- modelHomotypic(annotations)
nExp_poi <- round(0.075*nrow(rna.so@meta.data))
# nExp_poi.adj <- round(nExp_poi*(1-homotypic.prop))

rna.so <- doubletFinder_v3(rna.so, PCs = 1:30, pN = 0.25, pK = 0.005, nExp = nExp_poi, reuse.pANN = FALSE, sct = FALSE)

classification_key <- paste0("DF.classifications_0.25_0.005_", nExp_poi)
write.csv(data.frame(
    doubletfinder = rna.so@meta.data[, classification_key],
    row.names = rownames(rna.so@meta.data)
), "doubletfinder_inference.csv", quote = FALSE)
