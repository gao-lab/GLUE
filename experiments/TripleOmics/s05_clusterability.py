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
from math import ceil

import anndata as ad
import plotly.io as pio
import scanpy as sc
import yaml
from matplotlib import pyplot as plt
from matplotlib import rcParams

import scglue

# %%
scglue.plot.set_publication_params()
rcParams["figure.figsize"] = (4, 4)

PATH = "s05_clusterability"
os.makedirs(PATH, exist_ok=True)

# %%
with open("manual_colors.yaml", "r") as f:
    MANUAL_COLORS = yaml.load(f, Loader=yaml.FullLoader)

# %% [markdown]
# # Read data

# %%
rna = ad.read_h5ad("s03_markers/rna_filtered.h5ad")
rna_single = ad.read_h5ad("s01_preprocessing/rna.h5ad")
rna.obsm["X_pca"] = rna_single.obsm["X_pca"]
rna.obsm["X_umap"] = rna_single.obsm["X_umap"]
rna.uns["cell_type_colors"] = [MANUAL_COLORS[item] for item in rna.obs["cell_type"].cat.categories]
rna.uns["common_cell_type_colors"] = [MANUAL_COLORS[item] for item in rna.obs["common_cell_type"].cat.categories]
rna

# %%
met = ad.read_h5ad("s03_markers/met_filtered.h5ad")
met_single = ad.read_h5ad("s01_preprocessing/met.h5ad")
met.obsm["X_pca"] = met_single.obsm["X_pca"]
met.obsm["X_umap"] = met_single.obsm["X_umap"]
met.uns["cell_type_colors"] = [MANUAL_COLORS[item] for item in met.obs["cell_type"].cat.categories]
met.uns["common_cell_type_colors"] = [MANUAL_COLORS[item] for item in met.obs["common_cell_type"].cat.categories]
met

# %%
atac = ad.read_h5ad("s03_markers/atac_filtered.h5ad")
atac_single = ad.read_h5ad("s01_preprocessing/atac.h5ad")
atac.obsm["X_lsi"] = atac_single.obsm["X_lsi"]
atac.obsm["X_umap"] = atac_single.obsm["X_umap"]
atac.uns["cell_type_colors"] = [MANUAL_COLORS[item] for item in atac.obs["cell_type"].cat.categories]
atac.uns["common_cell_type_colors"] = [MANUAL_COLORS[item] for item in atac.obs["common_cell_type"].cat.categories]
atac

# %% [markdown]
# # Unimodal clustering

# %% [markdown]
# ## RNA

# %%
sc.pp.neighbors(rna, use_rep="X_pca", metric="cosine")
sc.tl.leiden(rna, resolution=1.0)

# %%
fig = sc.pl.umap(rna, color="leiden", return_fig=True)
fig.axes[0].set_title("Leiden clustering")
fig.savefig(f"{PATH}/rna_leiden_uni.pdf")

# %%
fig = sc.pl.umap(rna, color="common_cell_type", return_fig=True)
fig.axes[0].set_title("scRNA-seq common cell type")
fig.savefig(f"{PATH}/rna_cmct_uni.pdf")

# %%
link_cutoff = ceil(rna.shape[0] * 0.001)
LEIDEN_COLORS = {
    leiden: color for leiden, color in
    zip(rna.obs["leiden"].cat.categories, rna.uns["leiden_colors"])
}
fig = scglue.plot.sankey(
    rna.obs["leiden"], rna.obs["common_cell_type"],
    title="scRNA-seq cluster mapping",
    left_color=lambda x: LEIDEN_COLORS[x],
    right_color=lambda x: MANUAL_COLORS[x],
    link_color=lambda x: "rgba(0.9,0.9,0.9,0.2)" if x["value"] <= link_cutoff else "#CCCCCC"
)
pio.write_image(fig, f"{PATH}/rna_sankey.png", scale=10)

# %% [markdown]
# # Methylation

# %%
sc.pp.neighbors(met, use_rep="X_pca", metric="cosine")
sc.tl.leiden(met, resolution=1.0)

# %%
fig = sc.pl.umap(met, color="leiden", return_fig=True)
fig.axes[0].set_title("Leiden clustering")
fig.savefig(f"{PATH}/met_leiden_uni.pdf")

# %%
fig = sc.pl.umap(met, color="common_cell_type", return_fig=True)
fig.axes[0].set_title("snmC-seq common cell type")
fig.savefig(f"{PATH}/met_cmct_uni.pdf")

# %%
link_cutoff = ceil(met.shape[0] * 0.001)
LEIDEN_COLORS = {
    leiden: color for leiden, color in
    zip(met.obs["leiden"].cat.categories, met.uns["leiden_colors"])
}
fig = scglue.plot.sankey(
    met.obs["leiden"], met.obs["common_cell_type"],
    title="snmC-seq cluster mapping",
    left_color=lambda x: LEIDEN_COLORS[x],
    right_color=lambda x: MANUAL_COLORS[x],
    link_color=lambda x: "rgba(0.9,0.9,0.9,0.2)" if x["value"] <= link_cutoff else "#CCCCCC"
)
pio.write_image(fig, f"{PATH}/met_sankey.png", scale=10)

# %% [markdown]
# ## ATAC

# %%
sc.pp.neighbors(atac, use_rep="X_lsi", metric="cosine")
sc.tl.leiden(atac, resolution=1.2)

# %%
fig = sc.pl.umap(atac, color="leiden", return_fig=True)
fig.axes[0].set_title("Leiden clustering")
fig.savefig(f"{PATH}/atac_leiden_uni.pdf")

# %%
fig = sc.pl.umap(atac, color="common_cell_type", return_fig=True)
fig.axes[0].set_title("scATAC-seq common cell type")
fig.savefig(f"{PATH}/atac_cmct_uni.pdf")

# %%
link_cutoff = ceil(atac.shape[0] * 0.001)
LEIDEN_COLORS = {
    leiden: color for leiden, color in
    zip(atac.obs["leiden"].cat.categories, atac.uns["leiden_colors"])
}
fig = scglue.plot.sankey(
    atac.obs["leiden"], atac.obs["common_cell_type"],
    title="scATAC-seq cluster mapping",
    left_color=lambda x: LEIDEN_COLORS[x],
    right_color=lambda x: MANUAL_COLORS[x],
    link_color=lambda x: "rgba(0.9,0.9,0.9,0.2)" if x["value"] <= link_cutoff else "#CCCCCC"
)
pio.write_image(fig, f"{PATH}/atac_sankey.png", scale=10)

# %%
fig = sc.pl.umap(atac, color=[
    "Sall1", "Prox1", "Unc5b", "Zcchc24", "Vav2", 
    "Ndnf", "Cplx3", "Kit", "Myo3b", "Gm14204"
], ncols=5, return_fig=True)
fig.savefig(f"{PATH}/atac_vip_ndnf_markers_uni.pdf")
