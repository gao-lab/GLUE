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
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib import patches, rcParams

import scglue

# %%
scglue.plot.set_publication_params()
rcParams["figure.figsize"] = (7, 7)

PATH = "s02_glue_pretrain"
os.makedirs(PATH, exist_ok=True)

# %% [markdown]
# # Read data

# %%
rna_agg = anndata.read_h5ad("s01_preprocessing/rna_agg.h5ad")
atac_agg = anndata.read_h5ad("s01_preprocessing/atac_agg.h5ad")
prior = nx.read_graphml("s01_preprocessing/sub.graphml.gz")

# %% [markdown]
# # GLUE

# %%
vertices = sorted(prior.nodes)
scglue.models.configure_dataset(rna_agg, "NB", use_highly_variable=True, use_rep="X_pca")
scglue.models.configure_dataset(atac_agg, "NB", use_highly_variable=True, use_rep="X_lsi")
glue = scglue.models.SCGLUEModel(
    {"rna": rna_agg, "atac": atac_agg}, vertices,
    h_dim=512, random_seed=0
)

# %% tags=[]
glue.compile()
glue.fit(
    {"rna": rna_agg, "atac": atac_agg}, prior,
    edge_weight="weight", edge_sign="sign",
    align_burnin=np.inf, safe_burnin=False,
    directory=PATH
)
glue.save(f"{PATH}/final.dill")

# %%
rna_agg.obsm["X_glue"] = glue.encode_data("rna", rna_agg)
atac_agg.obsm["X_glue"] = glue.encode_data("atac", atac_agg)

# %% [markdown]
# # Visualization

# %%
combined_agg = anndata.AnnData(
    obs=pd.concat([rna_agg.obs, atac_agg.obs], join="inner"),
    obsm={"X_glue": np.concatenate([rna_agg.obsm["X_glue"], atac_agg.obsm["X_glue"]])}
)

# %%
sc.pp.neighbors(
    combined_agg, n_pcs=combined_agg.obsm["X_glue"].shape[1],
    use_rep="X_glue", metric="cosine"
)
sc.tl.umap(combined_agg)

# %%
fig = sc.pl.umap(
    combined_agg, color="cell_type",
    title="Cell type (metacells)", return_fig=True
)
ct_handles, ct_labels = fig.axes[0].get_legend_handles_labels()
fig.axes[0].get_legend().remove()
fig.savefig(f"{PATH}/combined_ct.pdf")

# %%
fig = sc.pl.umap(
    combined_agg, color="domain",
    title="Omics layer (metacells)", return_fig=True
)
domain_handles, domain_labels = fig.axes[0].get_legend_handles_labels()
fig.axes[0].get_legend().remove()
fig.savefig(f"{PATH}/combined_domain.pdf")

# %%
fig, ax = plt.subplots()
ax.set_visible(False)
placeholder = patches.Rectangle((0, 0), 1, 1, visible=False)
handles = [placeholder, *domain_handles, placeholder, placeholder, *ct_handles]
labels = ["Omics layer", *domain_labels, "", "Cell type", *ct_labels]
fig.legend(handles, labels, ncol=5, frameon=False)
fig.savefig(f"{PATH}/combined_legend.pdf")

# %%
rna_agg.obsm["X_glue_umap"] = combined_agg[rna_agg.obs_names, :].obsm["X_umap"]
atac_agg.obsm["X_glue_umap"] = combined_agg[atac_agg.obs_names, :].obsm["X_umap"]

# %%
fig = sc.pl.embedding(
    rna_agg, "X_glue_umap", color="cell_type",
    title="scRNA-seq cell type (metacells)", return_fig=True,
    legend_loc="on data", legend_fontsize=4, legend_fontoutline=0.5
)
fig.axes[0].set_xlabel("UMAP1")
fig.axes[0].set_ylabel("UMAP2")
fig.savefig(f"{PATH}/rna_ct.pdf")

# %%
fig = sc.pl.embedding(
    atac_agg, "X_glue_umap", color="cell_type",
    title="scATAC-seq cell type (metacells)", return_fig=True,
    legend_loc="on data", legend_fontsize=4, legend_fontoutline=0.5
)
fig.axes[0].set_xlabel("UMAP1")
fig.axes[0].set_ylabel("UMAP2")
fig.savefig(f"{PATH}/atac_ct.pdf")

# %% [markdown]
# # Save results

# %% tags=[]
rna_agg.write(f"{PATH}/rna_agg.h5ad", compression="gzip")
atac_agg.write(f"{PATH}/atac_agg.h5ad", compression="gzip")
combined_agg.write(f"{PATH}/combined_agg.h5ad", compression="gzip")
