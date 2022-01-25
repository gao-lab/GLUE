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
import networkx as nx
from networkx.algorithms.bipartite import biadjacency_matrix

import scglue

# %%
PATH = "e01_preprocessing"
os.makedirs(PATH, exist_ok=True)

# %% [markdown]
# # Read data

# %%
rna = anndata.read_h5ad("../../data/dataset/Cao-2020.h5ad")
atac = anndata.read_h5ad("../../data/dataset/Domcke-2020.h5ad")

# %%
rna_pp = anndata.read_h5ad("s01_preprocessing/rna.h5ad", backed="r")
atac_pp = anndata.read_h5ad("s01_preprocessing/atac.h5ad", backed="r")

# %%
graph = nx.read_graphml("s01_preprocessing/full.graphml.gz")

# %% [markdown]
# # Update meta

# %%
rna.var["highly_variable"] = [item in rna_pp.var_names for item in rna.var_names]
atac.var["highly_variable"] = [item in atac_pp.var_names for item in atac.var_names]

# %%
rna.obs["metacell"] = rna_pp.obs["metacell"]
atac.obs["metacell"] = atac_pp.obs["metacell"]
rna.obs["organ_balancing"] = rna_pp.obs["organ_balancing"]
atac.obs["organ_balancing"] = atac_pp.obs["organ_balancing"]
rna.obs["n_cells"] = 1
atac.obs["n_cells"] = 1

# %%
rna.obsm["X_pca"] = rna_pp.obsm["X_pca"]
rna.obsm["X_umap"] = rna_pp.obsm["X_umap"]
atac.obsm["X_lsi"] = atac_pp.obsm["X_lsi"]
atac.obsm["X_umap"] = atac_pp.obsm["X_umap"]

# %% [markdown]
# # Aggregation

# %%
rna_agg = scglue.data.aggregate_obs(
    rna, by="metacell", X_agg="sum",
    obs_agg={
        "cell_type": "majority", "Organ": "majority", "domain": "majority",
        "n_cells": "sum", "organ_balancing": "sum"
    },
    obsm_agg={"X_pca": "mean", "X_umap": "mean"}
)
rna_agg

# %%
atac_agg = scglue.data.aggregate_obs(
    atac, by="metacell", X_agg="sum",
    obs_agg={
        "cell_type": "majority", "tissue": "majority", "domain": "majority",
        "n_cells": "sum", "organ_balancing": "sum"
    },
    obsm_agg={"X_lsi": "mean", "X_umap": "mean"}
)
atac_agg

# %% [markdown]
# # Convert data

# %%
atac2rna = anndata.AnnData(
    X=atac.X @ biadjacency_matrix(graph, atac.var_names, rna.var_names),
    obs=atac.obs, var=rna.var
)

# %%
atac2rna_agg = anndata.AnnData(
    X=atac_agg.X @ biadjacency_matrix(graph, atac_agg.var_names, rna_agg.var_names),
    obs=atac_agg.obs, var=rna_agg.var
)

# %% [markdown]
# # Save data

# %%
rna.write(f"{PATH}/rna.h5ad", compression="gzip")
atac.write(f"{PATH}/atac.h5ad", compression="gzip")
atac2rna.write(f"{PATH}/atac2rna.h5ad", compression="gzip")

# %%
rna_agg.write(f"{PATH}/rna_agg.h5ad", compression="gzip")
atac_agg.write(f"{PATH}/atac_agg.h5ad", compression="gzip")
atac2rna_agg.write(f"{PATH}/atac2rna_agg.h5ad", compression="gzip")
