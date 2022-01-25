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

PATH = "s04_glue_final"
os.makedirs(PATH, exist_ok=True)

# %% [markdown]
# # Aggregated data

# %% [markdown]
# ## Read data

# %%
rna_agg = anndata.read_h5ad("s03_unsupervised_balancing/rna_agg_balanced.h5ad")
atac_agg = anndata.read_h5ad("s03_unsupervised_balancing/atac_agg_balanced.h5ad")
prior = nx.read_graphml("s01_preprocessing/sub.graphml.gz")

# %% [markdown]
# ## GLUE

# %%
vertices = sorted(prior.nodes)
scglue.models.configure_dataset(rna_agg, "NB", use_highly_variable=True, use_rep="X_pca", use_dsc_weight="nm_balancing")
scglue.models.configure_dataset(atac_agg, "NB", use_highly_variable=True, use_rep="X_lsi", use_dsc_weight="nm_balancing")
glue = scglue.models.SCGLUEModel(
    {"rna": rna_agg, "atac": atac_agg}, vertices,
    h_dim=512, random_seed=0
)

# %% tags=[]
glue_pretrain = scglue.models.load_model("s02_glue_pretrain/final.dill")
glue.adopt_pretrained_model(glue_pretrain)

# %% tags=[]
glue.compile()
glue.fit(
    {"rna": rna_agg, "atac": atac_agg}, prior,
    edge_weight="weight", edge_sign="sign",
    directory=f"{PATH}/agg"
)
glue.save(f"{PATH}/agg/final.dill")

# %%
rna_agg.obsm["X_glue"] = glue.encode_data("rna", rna_agg)
atac_agg.obsm["X_glue"] = glue.encode_data("atac", atac_agg)

# %% [markdown]
# ## Visualization

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
fig.savefig(f"{PATH}/agg/combined_ct.pdf")

# %%
fig = sc.pl.umap(
    combined_agg, color="domain",
    title="Omics layer (metacells)", return_fig=True
)
domain_handles, domain_labels = fig.axes[0].get_legend_handles_labels()
fig.axes[0].get_legend().remove()
fig.savefig(f"{PATH}/agg/combined_domain.pdf")

# %%
fig, ax = plt.subplots()
ax.set_visible(False)
placeholder = patches.Rectangle((0, 0), 1, 1, visible=False)
handles = [placeholder, *domain_handles, placeholder, placeholder, *ct_handles]
labels = ["Omics layer", *domain_labels, "", "Cell type", *ct_labels]
fig.legend(handles, labels, ncol=5, frameon=False)
fig.savefig(f"{PATH}/agg/combined_legend.pdf")

# %%
rna_agg.obsm["X_glue_umap"] = combined_agg[rna_agg.obs_names, :].obsm["X_umap"]
atac_agg.obsm["X_glue_umap"] = combined_agg[atac_agg.obs_names, :].obsm["X_umap"]

# %%
rna_agg.obs["cell_type"].cat.set_categories(combined_agg.obs["cell_type"].cat.categories, inplace=True)
atac_agg.obs["cell_type"].cat.set_categories(combined_agg.obs["cell_type"].cat.categories, inplace=True)

# %%
rna_agg.uns["cell_type_colors"] = combined_agg.uns["cell_type_colors"]
atac_agg.uns["cell_type_colors"] = combined_agg.uns["cell_type_colors"]

# %%
fig = sc.pl.embedding(
    rna_agg, "X_glue_umap", color="cell_type",
    title="scRNA-seq cell type (metacells)", return_fig=True,
    legend_loc="on data", legend_fontsize=4, legend_fontoutline=0.5
)
fig.axes[0].set_xlabel("UMAP1")
fig.axes[0].set_ylabel("UMAP2")
fig.savefig(f"{PATH}/agg/rna_ct.pdf")

# %%
fig = sc.pl.embedding(
    atac_agg, "X_glue_umap", color="cell_type",
    title="scATAC-seq cell type (metacells)", return_fig=True,
    legend_loc="on data", legend_fontsize=4, legend_fontoutline=0.5
)
fig.axes[0].set_xlabel("UMAP1")
fig.axes[0].set_ylabel("UMAP2")
fig.savefig(f"{PATH}/agg/atac_ct.pdf")

# %% [markdown]
# ## Save results

# %%
rna_agg.write(f"{PATH}/agg/rna_agg.h5ad", compression="gzip")
atac_agg.write(f"{PATH}/agg/atac_agg.h5ad", compression="gzip")
combined_agg.write(f"{PATH}/agg/combined_agg.h5ad", compression="gzip")

# %% [markdown]
# # Full data

# %% [markdown]
# ## Read data

# %%
rna = anndata.read_h5ad("s03_unsupervised_balancing/rna_balanced.h5ad")
atac = anndata.read_h5ad("s03_unsupervised_balancing/atac_balanced.h5ad")
prior = nx.read_graphml("s01_preprocessing/sub.graphml.gz")

# %% [markdown]
# ## GLUE

# %%
vertices = sorted(prior.nodes)
scglue.models.configure_dataset(rna, "NB", use_highly_variable=True, use_rep="X_pca", use_dsc_weight="nc_balancing")
scglue.models.configure_dataset(atac, "NB", use_highly_variable=True, use_rep="X_lsi", use_dsc_weight="nc_balancing")
glue = scglue.models.SCGLUEModel(
    {"rna": rna, "atac": atac}, vertices,
    h_dim=512, random_seed=0
)

# %% tags=[]
glue_agg = scglue.models.load_model(f"{PATH}/agg/final.dill")
glue.adopt_pretrained_model(glue_agg)

# %% tags=[]
glue.compile(lr=1e-3)
glue.fit(
    {"rna": rna, "atac": atac}, prior,
    edge_weight="weight", edge_sign="sign",
    align_burnin=0, data_batch_size=512,
    directory=f"{PATH}/full"
)
glue.save(f"{PATH}/full/final.dill")

# %%
rna.obsm["X_glue"] = glue.encode_data("rna", rna)
atac.obsm["X_glue"] = glue.encode_data("atac", atac)

# %% [markdown]
# ## Visualization

# %%
combined = anndata.AnnData(
    obs=pd.concat([rna.obs, atac.obs], join="inner"),
    obsm={"X_glue": np.concatenate([rna.obsm["X_glue"], atac.obsm["X_glue"]])}
)

# %%
sc.pp.neighbors(
    combined, n_pcs=combined.obsm["X_glue"].shape[1],
    use_rep="X_glue", metric="cosine"
)
sc.tl.umap(combined)

# %%
fig = sc.pl.umap(
    combined, color="cell_type",
    title="Cell type", return_fig=True
)
ct_handles, ct_labels = fig.axes[0].get_legend_handles_labels()
fig.axes[0].get_legend().remove()
fig.savefig(f"{PATH}/full/combined_ct.pdf")

# %%
fig = sc.pl.umap(
    combined, color="domain",
    title="Omics layer", return_fig=True
)
domain_handles, domain_labels = fig.axes[0].get_legend_handles_labels()
fig.axes[0].get_legend().remove()
fig.savefig(f"{PATH}/full/combined_domain.pdf")

# %%
fig, ax = plt.subplots()
ax.set_visible(False)
placeholder = patches.Rectangle((0, 0), 1, 1, visible=False)
handles = [placeholder, *domain_handles, placeholder, placeholder, *ct_handles]
labels = ["Omics layer", *domain_labels, "", "Cell type", *ct_labels]
fig.legend(handles, labels, ncol=5, frameon=False)
fig.savefig(f"{PATH}/full/combined_legend.pdf")

# %%
rna.obsm["X_glue_umap"] = combined[rna.obs_names, :].obsm["X_umap"]
atac.obsm["X_glue_umap"] = combined[atac.obs_names, :].obsm["X_umap"]

# %%
rna.obs["cell_type"].cat.set_categories(combined.obs["cell_type"].cat.categories, inplace=True)
atac.obs["cell_type"].cat.set_categories(combined.obs["cell_type"].cat.categories, inplace=True)

# %%
rna.uns["cell_type_colors"] = combined.uns["cell_type_colors"]
atac.uns["cell_type_colors"] = combined.uns["cell_type_colors"]

# %%
fig = sc.pl.embedding(
    rna, "X_glue_umap", color="cell_type",
    title="scRNA-seq cell type", return_fig=True,
    legend_loc="on data", legend_fontsize=4, legend_fontoutline=0.5
)
fig.axes[0].set_xlabel("UMAP1")
fig.axes[0].set_ylabel("UMAP2")
fig.savefig(f"{PATH}/full/rna_ct.pdf")

# %%
fig = sc.pl.embedding(
    atac, "X_glue_umap", color="cell_type",
    title="scATAC-seq cell type", return_fig=True,
    legend_loc="on data", legend_fontsize=4, legend_fontoutline=0.5
)
fig.axes[0].set_xlabel("UMAP1")
fig.axes[0].set_ylabel("UMAP2")
fig.savefig(f"{PATH}/full/atac_ct.pdf")

# %% [markdown]
# ## Save results

# %%
rna.write(f"{PATH}/full/rna.h5ad", compression="gzip")
atac.write(f"{PATH}/full/atac.h5ad", compression="gzip")
combined.write(f"{PATH}/full/combined.h5ad", compression="gzip")
