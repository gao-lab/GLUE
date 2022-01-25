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
PATH <- "e01_inmf"
if (!dir.exists(PATH)) {
    dir.create(PATH)
}

# %% [markdown]
# # Read data

# %%
rna <- read_h5ad("s01_preprocessing/rna.h5ad")
atac <- read_h5ad("s01_preprocessing/atac2rna.h5ad")
met <- read_h5ad("s01_preprocessing/met2rna.h5ad")

# %%
int.liger <- createLiger(list(
    rna = Matrix::t(rna$X),
    atac = Matrix::t(atac$X),
    met = Matrix::t(max(met$X) - met$X)
))

# %%
hvg <- rownames(rna$var)[rna$var$highly_variable]

# %%
stopifnot(all(hvg %in% rownames(rna$var)))
stopifnot(all(hvg %in% rownames(atac$var)))
stopifnot(all(hvg %in% rownames(met$var)))

# %%
hvg <- intersect(hvg, rownames(int.liger@raw.data$rna))  # Because of feature filtering in createLiger
hvg <- intersect(hvg, rownames(int.liger@raw.data$atac))  # Because of feature filtering in createLiger
hvg <- intersect(hvg, rownames(int.liger@raw.data$met))  # Because of feature filtering in createLiger

# %%
rna_cells <- rownames(rna$obs)
atac_cells <- rownames(atac$obs)
met_cells <- rownames(met$obs)
n_cells <- nrow(rna$obs) + nrow(atac$obs) + nrow(met$obs)
min_cells <- min(nrow(rna$obs), nrow(atac$obs), nrow(met$obs))

# %%
rm(rna, atac)
gc()  # Reduce memory usage

# %% [markdown]
# # Data preprocessing

# %%
int.liger <- normalize(int.liger)

# %%
int.liger@var.genes <- hvg
int.liger <- scaleNotCenter(int.liger)

# %% [markdown]
# # Run iNMF

# %%
int.liger <- online_iNMF(int.liger, k = 20, miniBatch_size = min(5000, min_cells), seed = 0)
int.liger <- quantile_norm(int.liger, rand.seed = 0)

# %% [markdown]
# # Save results

# %%
combined_latent <- int.liger@H.norm
rna_latent <- combined_latent[rna_cells, ]
atac_latent <- combined_latent[atac_cells, ]
met_latent <- combined_latent[met_cells, ]
write.table(
    rna_latent, file.path(PATH, "rna_latent.csv"),
    sep = ",", row.names = TRUE, col.names = FALSE, quote = FALSE, na = ""
)
write.table(
    atac_latent, file.path(PATH, "atac_latent.csv"),
    sep = ",", row.names = TRUE, col.names = FALSE, quote = FALSE, na = ""
)
write.table(
    met_latent, file.path(PATH, "met_latent.csv"),
    sep = ",", row.names = TRUE, col.names = FALSE, quote = FALSE, na = ""
)
