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
    source(".Rprofile")
    library(Seurat)
    library(Signac)
})

# %%
PATH <- "e02_cca_anchor"
if (!dir.exists(PATH)) {
    dir.create(PATH)
}

# %% [markdown]
# # Read data

# %%
rna <- read_h5ad("e01_preprocessing/rna_agg.h5ad")

# %%
atac <- read_h5ad("e01_preprocessing/atac_agg.h5ad")

# %%
atac2rna <- read_h5ad("e01_preprocessing/atac2rna_agg.h5ad")

# %%
rna.so <- CreateSeuratObject(
    counts = Matrix::t(rna$X), assay = "RNA", meta.data = rna$obs
)

# %%
atac.so <- CreateSeuratObject(
    counts = Matrix::t(atac$X), assay = "ATAC", meta.data = atac$obs
)

# %%
atac.so[["ACTIVITY"]] <- CreateAssayObject(counts = Matrix::t(atac2rna$X))

# %% [markdown]
# # Preprocessing

# %% [markdown]
# ## RNA

# %%
rna.so <- NormalizeData(rna.so)

# %%
VariableFeatures(rna.so) <- rownames(rna$var)[rna$var$highly_variable]

# %%
rna.so <- ScaleData(rna.so)

# %%
rna.so <- RunPCA(rna.so, seed.use = 0, verbose = FALSE)

# %% [markdown]
# ## ATAC2RNA

# %%
DefaultAssay(atac.so) <- "ACTIVITY"

# %%
VariableFeatures(atac.so) <- rownames(rna$var)[rna$var$highly_variable]

# %%
atac.so <- NormalizeData(atac.so)

# %%
atac.so <- ScaleData(atac.so)

# %% [markdown]
# ## ATAC

# %%
DefaultAssay(atac.so) <- "ATAC"

# %%
atac.so <- RunTFIDF(atac.so)

# %%
atac.so <- FindTopFeatures(atac.so, min.cutoff = "q0")

# %%
atac.so <- RunSVD(atac.so)

# %%
rm(rna, atac, atac2rna)

# %% [markdown]
# # Integration

# %%
transfer.anchors <- FindTransferAnchors(
    reference = rna.so, query = atac.so,
    features = VariableFeatures(rna.so),
    reference.assay = "RNA", query.assay = "ACTIVITY",
    reduction = "cca"
)

# %%
refdata <- GetAssayData(
    rna.so, assay = "RNA", slot = "data"
)[VariableFeatures(rna.so), ]

# %%
imputation <- TransferData(
    anchorset = transfer.anchors, refdata = refdata,
    weight.reduction = atac.so[["lsi"]], dims = 1:30
)

# %%
atac.so[["RNA"]] <- imputation

# %%
coembed.so <- merge(x = rna.so, y = atac.so)
coembed.so <- ScaleData(
    coembed.so, features = VariableFeatures(rna.so), do.scale = FALSE
)
coembed.so <- RunPCA(
    coembed.so, features = VariableFeatures(rna.so),
    seed.use = 0, verbose = FALSE
)

# %% [markdown]
# # Save results

# %%
combined_latent <- Embeddings(coembed.so)

# %%
rna_latent <- combined_latent[colnames(rna.so), ]

# %%
atac_latent <- combined_latent[colnames(atac.so), ]

# %%
write.table(
    rna_latent, file.path(PATH, "rna_latent.csv"),
    sep = ",", row.names = TRUE, col.names = FALSE, quote = FALSE
)

# %%
write.table(
    atac_latent, file.path(PATH, "atac_latent.csv"),
    sep = ",", row.names = TRUE, col.names = FALSE, quote = FALSE
)
