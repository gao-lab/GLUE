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

# %%
PATH = "t01_preprocessing"
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

# %% [markdown]
# # Subsample

# %%
rna = rna[rna_pp.obs["mask"], :]
atac = atac[atac_pp.obs["mask"], :]

# %% [markdown]
# # Convert data

# %%
atac2rna = anndata.AnnData(
    X=atac.X @ biadjacency_matrix(graph, atac.var_names, rna.var_names),
    obs=atac.obs, var=rna.var
)

# %% [markdown]
# # Save data

# %%
rna.write(f"{PATH}/rna.h5ad", compression="gzip")
atac.write(f"{PATH}/atac.h5ad", compression="gzip")
atac2rna.write(f"{PATH}/atac2rna.h5ad", compression="gzip")
