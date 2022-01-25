# ---
# jupyter:
#   jupytext:
#     formats: ipynb,R:percent
#     text_representation:
#       extension: .R
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: R 4.0.2
#     language: R
#     name: ir_4.0.2
# ---

# %%
suppressPackageStartupMessages({
    source("../../.Rprofile", chdir = TRUE)
    library(dplyr)
    library(cicero)
})

# %%
PATH <- "e01_cicero"
if (!dir.exists(PATH)) {
    dir.create(PATH)
}

# %% [markdown]
# # Read data

# %%
atac <- read_h5ad("s01_preprocessing/atac.h5ad")

# %%
cellinfo <- atac$obs
head(cellinfo, n = 2)

# %%
peakinfo <- atac$var %>% select(
    chr = chrom, bp1 = chromStart, bp2 = chromEnd
) %>% mutate(
    site_name = paste(chr, bp1, bp2, sep = "_")
)
rownames(peakinfo) <- peakinfo$site_name
head(peakinfo, n = 2)

# %%
indata <- t(atac$X)
indata@x[indata@x > 0] <- 1
rownames(indata) <- rownames(peakinfo)
colnames(indata) <- rownames(cellinfo)

# %%
input_cds <- newCellDataSet(
    indata,
    phenoData = methods::new("AnnotatedDataFrame", data = cellinfo),
    featureData = methods::new("AnnotatedDataFrame", data = peakinfo),
    expressionFamily = VGAM::binomialff(),
    lowerDetectionLimit = 0
)

# %% [markdown]
# # Cicero

# %%
set.seed(2021)
input_cds <- detectGenes(input_cds)
input_cds <- estimateSizeFactors(input_cds)
input_cds <- reduceDimension(
    input_cds, max_components = 2, num_dim = 6,
    reduction_method = "tSNE", norm_method = "none"
)

# %%
tsne_coords <- t(reducedDimA(input_cds))
row.names(tsne_coords) <- row.names(pData(input_cds))
cicero_cds <- make_cicero_cds(input_cds, reduced_coordinates = tsne_coords)

# %%
human.hg38.genome <- read.table("../../data/genome/GRCh38.p12.genome.fa.fai", stringsAsFactors = TRUE)[, 1:2]

# %%
conns <- run_cicero(cicero_cds, human.hg38.genome, window=3e5)
head(conns)

# %%
write.csv(conns, gzfile(file.path(PATH, "cicero_conns.csv.gz")), row.names = FALSE, quote = FALSE)
