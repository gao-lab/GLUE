# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import umap
from matplotlib import patches, rcParams

import scglue

# %%
scglue.plot.set_publication_params()
rcParams["figure.figsize"] = (7, 7)

PATH = "e03_visualize"
os.makedirs(PATH, exist_ok=True)

# %% [markdown]
# # Read aggregated data

# %%
rna = anndata.read_h5ad("e01_preprocessing/rna_agg.h5ad", backed="r")
atac = anndata.read_h5ad("e01_preprocessing/atac_agg.h5ad", backed="r")

# %% [markdown]
# # Seurat v3

# %% [markdown]
# ## Read latent

# %%
rna.obsm["X_latent"] = pd.read_csv(
    "e02_cca_anchor/rna_latent.csv", header=None, index_col=0
).loc[rna.obs_names].to_numpy()
atac.obsm["X_latent"] = pd.read_csv(
    "e02_cca_anchor/atac_latent.csv", header=None, index_col=0
).loc[atac.obs_names].to_numpy()

# %%
combined = anndata.AnnData(
    obs=pd.concat([rna.obs, atac.obs], join="inner"),
    obsm={"X_latent": np.concatenate([rna.obsm["X_latent"], atac.obsm["X_latent"]])}
)
combined.obs["cell_type"] = pd.Categorical(
    combined.obs["cell_type"], categories=np.unique(np.concatenate([
        rna.obs["cell_type"].cat.categories,
        atac.obs["cell_type"].cat.categories
    ]))
)

# %% [markdown]
# ## Plotting

# %%
sc.pp.neighbors(
    combined, use_rep="X_latent",
    n_pcs=combined.obsm["X_latent"].shape[1],
    metric="cosine"
)
sc.tl.umap(combined)

# %%
fig = sc.pl.umap(
    combined, color="cell_type", 
    title="Cell type", return_fig=True
)
ct_handles, ct_labels = fig.axes[0].get_legend_handles_labels()
fig.axes[0].get_legend().remove()
fig.savefig(f"{PATH}/cca_combined_agg_ct.pdf")

# %%
fig = sc.pl.umap(
    combined, color="domain",
    title="Omics layer", return_fig=True
)
domain_handles, domain_labels = fig.axes[0].get_legend_handles_labels()
fig.axes[0].get_legend().remove()
fig.savefig(f"{PATH}/cca_combined_agg_domain.pdf")

# %%
fig, ax = plt.subplots()
ax.set_visible(False)
placeholder = patches.Rectangle((0, 0), 1, 1, visible=False)
handles = [placeholder, *domain_handles, placeholder, placeholder, *ct_handles]
labels = ["Omics layer", *domain_labels, "", "Cell type", *ct_labels]
fig.legend(handles, labels, ncol=5, frameon=False)
fig.savefig(f"{PATH}/cca_combined_agg_legend.pdf")

# %%
combined_umap = pd.DataFrame(combined.obsm["X_umap"], index=combined.obs_names)
combined_umap.to_csv(f"{PATH}/cca_combined_agg_umap.csv", header=False, index=True)

# %% [markdown]
# # Read full data

# %%
rna = anndata.read_h5ad("e01_preprocessing/rna.h5ad", backed="r")
atac = anndata.read_h5ad("e01_preprocessing/atac.h5ad", backed="r")

# %% [markdown]
# # iNMF

# %%
rna.obsm["X_latent"] = pd.read_csv(
    "e02_inmf/rna_latent.csv", header=None, index_col=0
).loc[rna.obs_names].to_numpy()
atac.obsm["X_latent"] = pd.read_csv(
    "e02_inmf/atac_latent.csv", header=None, index_col=0
).loc[atac.obs_names].to_numpy()

# %%
combined = anndata.AnnData(
    obs=pd.concat([rna.obs, atac.obs], join="inner"),
    obsm={"X_latent": np.concatenate([rna.obsm["X_latent"], atac.obsm["X_latent"]])}
)

# %% [markdown]
# ## Plotting

# %%
combined.obsm["X_latent"] += np.random.RandomState(0).randn(
    *combined.obsm["X_latent"].shape
) * 2e-4  # Add a slight amount of noise to avoid UMAP segfault

# %%
sc.pp.neighbors(
    combined, use_rep="X_latent",
    n_pcs=combined.obsm["X_latent"].shape[1],
    metric="cosine"
)
sc.tl.umap(combined)

# %%
fig = sc.pl.umap(
    combined, color="cell_type", 
    title="Cell type", return_fig=True
)
ct_handles, ct_labels = fig.axes[0].get_legend_handles_labels()
fig.axes[0].get_legend().remove()
fig.savefig(f"{PATH}/inmf_combined_ct.pdf")

# %%
fig = sc.pl.umap(
    combined, color="domain",
    title="Omics layer", return_fig=True
)
domain_handles, domain_labels = fig.axes[0].get_legend_handles_labels()
fig.axes[0].get_legend().remove()
fig.savefig(f"{PATH}/inmf_combined_domain.pdf")

# %%
fig, ax = plt.subplots()
ax.set_visible(False)
placeholder = patches.Rectangle((0, 0), 1, 1, visible=False)
handles = [placeholder, *domain_handles, placeholder, placeholder, *ct_handles]
labels = ["Omics layer", *domain_labels, "", "Cell type", *ct_labels]
fig.legend(handles, labels, ncol=5, frameon=False)
fig.savefig(f"{PATH}/inmf_combined_legend.pdf")

# %%
combined_umap = pd.DataFrame(combined.obsm["X_umap"], index=combined.obs_names)
combined_umap.to_csv(f"{PATH}/inmf_combined_umap.csv", header=False, index=True)
