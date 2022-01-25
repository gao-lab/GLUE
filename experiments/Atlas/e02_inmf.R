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
    library(hdf5r)
    library(rliger)
})

# %%
PATH <- "e02_inmf"
if (!dir.exists(PATH)) {
    dir.create(PATH)
}

# %% [markdown]
# # Read data

# %%
int.liger <- createLiger(list(
    atac = "e01_export_old_h5ad/atac2rna.h5ad",
    rna = "e01_export_old_h5ad/rna.h5ad"
), format.type = "AnnData")

# %%
rna.var <- int.liger@raw.data$rna[["var"]][]
atac.var <- int.liger@raw.data$atac[["var"]][]
hvg <- rna.var$index[rna.var$highly_variable]
stopifnot(all(hvg %in% rna.var$index))
stopifnot(all(hvg %in% atac.var$index))

# %%
hvg <- intersect(hvg, rna.var$index)
hvg <- intersect(hvg, atac.var$index)

# %%
rna_cells <- int.liger@raw.data$rna[["obs"]][]$cell
atac_cells <- int.liger@raw.data$atac[["obs"]][]$cell

# %% [markdown]
# # Preprocessing

# %%
int.liger <- normalize(int.liger)

# %%
int.liger@var.genes <- hvg
int.liger <- scaleNotCenter(int.liger)

# %% [markdown]
# # Integration

# %%
int.liger <- online_iNMF(int.liger, k = 20, miniBatch_size = 5000, seed = 0)
int.liger <- quantile_norm(int.liger, rand.seed = 0)

# %% [markdown]
# # Save results

# %%
combined_latent <- int.liger@H.norm

# %%
missing_cells <- setdiff(
    union(rna_cells, atac_cells),
    rownames(combined_latent)
)  # Because of cell filtering in scaleNotCenter
length(missing_cells)

# %%
rna_latent <- combined_latent[rna_cells, ]
atac_latent <- combined_latent[atac_cells, ]
write.table(
    rna_latent, file.path(PATH, "rna_latent.csv"),
    sep = ",", row.names = TRUE, col.names = FALSE, quote = FALSE, na = ""
)
write.table(
    atac_latent, file.path(PATH, "atac_latent.csv"),
    sep = ",", row.names = TRUE, col.names = FALSE, quote = FALSE, na = ""
)
