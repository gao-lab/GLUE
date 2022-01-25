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
from pathlib import Path

import anndata
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib import rcParams

import scglue

# %%
scglue.plot.set_publication_params()
rcParams["figure.figsize"] = (4, 4)

path = Path("s02_glue")
path.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Read data

# %%
rna = anndata.read_h5ad("s01_preprocessing/rna.h5ad")
met = anndata.read_h5ad("s01_preprocessing/met.h5ad")
atac = anndata.read_h5ad("s01_preprocessing/atac.h5ad")
met2rna = anndata.read_h5ad("s01_preprocessing/met2rna.h5ad")
atac2rna = anndata.read_h5ad("s01_preprocessing/atac2rna.h5ad")
prior = nx.read_graphml("s01_preprocessing/sub.graphml.gz")

# %% [markdown]
# # GLUE

# %%
vertices = sorted(prior.nodes)
scglue.models.configure_dataset(rna, "NB", use_highly_variable=True, use_rep="X_pca")
scglue.models.configure_dataset(met, "ZILN", use_highly_variable=True, use_rep="X_pca")
scglue.models.configure_dataset(atac, "NB", use_highly_variable=True, use_rep="X_lsi")

# %% tags=[]
glue = scglue.models.SCGLUEModel(
    {"rna": rna, "met": met, "atac": atac}, vertices,
    random_seed=0
)
glue.compile()
glue.fit(
    {"rna": rna, "met": met, "atac": atac},
    prior, edge_weight="weight", edge_sign="sign",
    align_burnin=np.inf, safe_burnin=False,
    directory=path / "pretrain"
)
glue.save(path / "pretrain" / "final.dill")

# %%
rna.obsm["X_glue"] = glue.encode_data("rna", rna)
met.obsm["X_glue"] = glue.encode_data("met", met)
atac.obsm["X_glue"] = glue.encode_data("atac", atac)
scglue.data.estimate_balancing_weight(
    rna, met, atac, use_rep="X_glue"
)

# %%
scglue.models.configure_dataset(rna, "NB", use_highly_variable=True, use_rep="X_pca", use_dsc_weight="balancing_weight")
scglue.models.configure_dataset(met, "ZILN", use_highly_variable=True, use_rep="X_pca", use_dsc_weight="balancing_weight")
scglue.models.configure_dataset(atac, "NB", use_highly_variable=True, use_rep="X_lsi", use_dsc_weight="balancing_weight")

# %%
glue = scglue.models.SCGLUEModel(
    {"rna": rna, "met": met, "atac": atac}, vertices,
    random_seed=0
)
glue.adopt_pretrained_model(scglue.models.load_model(
    path / "pretrain" / "final.dill"
))
glue.compile()
glue.fit(
    {"rna": rna, "met": met, "atac": atac},
    prior, edge_weight="weight", edge_sign="sign",
    directory=path / "fine-tune"
)
glue.save(path / "fine-tune" / "final.dill")

# %% [markdown]
# # Visualization

# %% tags=[]
rna.obsm["X_glue"] = glue.encode_data("rna", rna)
met.obsm["X_glue"] = glue.encode_data("met", met)
atac.obsm["X_glue"] = glue.encode_data("atac", atac)
met2rna.obsm["X_glue"] = met.obsm["X_glue"]
atac2rna.obsm["X_glue"] = atac.obsm["X_glue"]

# %%
combined = anndata.AnnData(
    obs=pd.concat([rna.obs, met.obs, atac.obs], join="inner"),
    obsm={"X_glue": np.concatenate([
        rna.obsm["X_glue"], met.obsm["X_glue"], atac.obsm["X_glue"]
    ])}
)
combined.obs["domain"] = pd.Categorical(
    combined.obs["domain"],
    categories=["scRNA-seq", "scATAC-seq", "snmC-seq"]
)
combined.uns["domain_colors"] = list(sns.color_palette(n_colors=3).as_hex())
combined

# %%
sc.pp.neighbors(combined, n_pcs=50, use_rep="X_glue", metric="cosine")
sc.tl.umap(combined)

# %%
rna.obsm["X_glue_umap"] = combined[rna.obs_names, :].obsm["X_umap"]
met.obsm["X_glue_umap"] = combined[met.obs_names, :].obsm["X_umap"]
atac.obsm["X_glue_umap"] = combined[atac.obs_names, :].obsm["X_umap"]
met2rna.obsm["X_glue_umap"] = met.obsm["X_glue_umap"]
atac2rna.obsm["X_glue_umap"] = atac.obsm["X_glue_umap"]

# %%
fig, ax = plt.subplots()
ax.scatter(
    x=rna.obsm["X_glue_umap"][:, 0], y=rna.obsm["X_glue_umap"][:, 1],
    label="scRNA-seq", s=0.5, c=combined.uns["domain_colors"][0],
    edgecolor=None, rasterized=True
)
ax.scatter(
    x=atac.obsm["X_glue_umap"][:, 0], y=atac.obsm["X_glue_umap"][:, 1],
    label="scATAC-seq", s=1.0, c=combined.uns["domain_colors"][1],
    edgecolor=None, rasterized=True
)
ax.scatter(
    x=met.obsm["X_glue_umap"][:, 0], y=met.obsm["X_glue_umap"][:, 1],
    label="snmC-seq", s=0.7, c=combined.uns["domain_colors"][2],
    edgecolor=None, rasterized=True
)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.set_title("Omics layer")
lgnd = ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
for handle in lgnd.legendHandles:
    handle.set_sizes([30.0])
fig.savefig(path / "combined_domain.pdf")

# %%
fig = sc.pl.embedding(rna, "X_glue_umap", color="cell_type", title="scRNA-seq cell type", return_fig=True)
fig.axes[0].set_xlabel("UMAP1")
fig.axes[0].set_ylabel("UMAP2")
fig.savefig(path / "rna_ct.pdf")

# %%
fig = sc.pl.embedding(met, "X_glue_umap", color="cell_type", title="snmC-seq cell type", return_fig=True)
fig.axes[0].set_xlabel("UMAP1")
fig.axes[0].set_ylabel("UMAP2")
fig.savefig(path / "met_ct.pdf")

# %%
fig = sc.pl.embedding(atac, "X_glue_umap", color="cell_type", title="scATAC-seq cell type", return_fig=True)
fig.axes[0].set_xlabel("UMAP1")
fig.axes[0].set_ylabel("UMAP2")
fig.savefig(path / "atac_ct.pdf")

# %%
rna.write(path / "rna.h5ad", compression="gzip")
met.write(path / "met.h5ad", compression="gzip")
atac.write(path / "atac.h5ad", compression="gzip")
met2rna.write(path / "met2rna.h5ad", compression="gzip")
atac2rna.write(path / "atac2rna.h5ad", compression="gzip")
combined.write(path / "combined.h5ad", compression="gzip")
