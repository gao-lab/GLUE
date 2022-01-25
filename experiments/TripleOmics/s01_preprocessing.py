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

# %% tags=[]
import itertools
import os

import anndata
import networkx as nx
import numpy as np
import scanpy as sc
import yaml
from matplotlib import rcParams

import scglue

# %%
scglue.plot.set_publication_params()
rcParams["figure.figsize"] = (4, 4)

PATH = "s01_preprocessing"
os.makedirs(PATH, exist_ok=True)

# %% [markdown]
# # Read data

# %%
rna = anndata.read_h5ad("../../data/dataset/Saunders-2018.h5ad")
rna

# %%
met = anndata.read_h5ad("../../data/dataset/Luo-2017.h5ad")
met

# %%
atac = anndata.read_h5ad("../../data/dataset/10x-ATAC-Brain5k.h5ad")
atac

# %%
with open("manual_colors.yaml", "r") as f:
    MANUAL_COLORS = yaml.load(f, Loader=yaml.FullLoader)

# %% [markdown]
# # Build prior graph

# %%
graph = scglue.genomics.window_graph(
    scglue.genomics.Bed(rna.var.assign(name=rna.var_names)).expand(2e3, 0),
    scglue.genomics.Bed(atac.var.assign(name=atac.var_names)),
    window_size=0, attr_fn=lambda l, r, d: {"weight": 1.0, "sign": 1}
)

# %%
for i in met.var_names:
    if i.endswith("_mCH"):
        j = i.replace("_mCH", "")
    elif i.endswith("_mCG"):
        j = i.replace("_mCG", "")
    else:
        raise ValueError("Unexpected var name!")
    graph.add_edge(j, i, weight=1.0, sign=-1)

# %%
hvg_reachable = scglue.graph.reachable_vertices(graph, rna.var.query("highly_variable").index)

# %%
met.var["highly_variable"] = [item in hvg_reachable for item in met.var_names]
met.var["highly_variable"].sum()

# %%
atac.var["highly_variable"] = [item in hvg_reachable for item in atac.var_names]
atac.var["highly_variable"].sum()

# %%
graph = scglue.graph.compose_multigraph(graph, graph.reverse())
for i in itertools.chain(rna.var_names, met.var_names, atac.var_names):
    graph.add_edge(i, i, weight=1.0, sign=1)

# %%
subgraph = graph.subgraph(hvg_reachable)

# %%
nx.write_graphml(graph, f"{PATH}/full.graphml.gz")
nx.write_graphml(subgraph, f"{PATH}/sub.graphml.gz")

# %% [markdown]
# # RNA

# %%
rna.layers["raw_count"] = rna.X.copy()
sc.pp.normalize_total(rna)
sc.pp.log1p(rna)
sc.pp.scale(rna, max_value=10)
sc.tl.pca(rna, n_comps=100, use_highly_variable=True, svd_solver="auto")

# %%
rna.X = rna.layers["raw_count"]
del rna.layers["raw_count"]

# %%
sc.pp.neighbors(rna, n_pcs=100, metric="cosine")
sc.tl.umap(rna)

# %%
rna.obs["cell_type"].cat.set_categories([
    "Layer2/3", "Layer5a", "Layer5", "Layer5b", "Layer6",
    "Claustrum", "CGE", "MGE"
], inplace=True)
rna.uns["cell_type_colors"] = [MANUAL_COLORS[item] for item in rna.obs["cell_type"].cat.categories]

# %%
fig = sc.pl.umap(rna, color="cell_type", title="scRNA-seq cell type", return_fig=True)
fig.savefig(f"{PATH}/rna_ct.pdf")

# %% [markdown]
# # Methylation

# %%
met.X = met.layers["norm"].copy()
sc.pp.log1p(met)
sc.pp.scale(met, max_value=10)
sc.tl.pca(met, n_comps=100, use_highly_variable=True, svd_solver="auto")

# %%
met.X = met.layers["norm"]
del met.layers["norm"]

# %%
sc.pp.neighbors(met, n_pcs=100, metric="cosine")
sc.tl.umap(met)

# %%
met.obs["cell_type"].cat.set_categories([
    "mL2/3", "mL4", "mL5-1", "mDL-1", "mDL-2", "mL5-2",
    "mL6-1", "mL6-2", "mDL-3", "mIn-1", "mVip",
    "mNdnf-1", "mNdnf-2", "mPv", "mSst-1", "mSst-2"
], inplace=True)
met.uns["cell_type_colors"] = [MANUAL_COLORS[item] for item in met.obs["cell_type"].cat.categories]

# %%
fig = sc.pl.umap(met, color="cell_type", title="snmC-seq cell type", return_fig=True)
fig.savefig(f"{PATH}/met_ct.pdf")

# %% [markdown]
# # ATAC

# %%
scglue.data.lsi(atac, n_components=100, use_highly_variable=False, n_iter=15)

# %%
sc.pp.neighbors(atac, n_pcs=100, use_rep="X_lsi", metric="cosine")
sc.tl.umap(atac)

# %%
atac.obs["cell_type"].cat.set_categories([
    "L2/3 IT", "L4", "L5 IT", "L6 IT", "L5 PT",
    "NP", "L6 CT", "Vip", "Pvalb", "Sst"
], inplace=True)
atac.uns["cell_type_colors"] = [MANUAL_COLORS[item] for item in atac.obs["cell_type"].cat.categories]

# %%
fig = sc.pl.umap(atac, color="cell_type", title="scATAC-seq cell type", return_fig=True)
fig.savefig(f"{PATH}/atac_ct.pdf")

# %% [markdown]
# # Convert data for marker analysis

# %% [markdown]
# ## ATAC-to-RNA

# %%
biadj = nx.algorithms.bipartite.biadjacency_matrix(graph, atac.var_names, rna.var_names)
atac2rna = anndata.AnnData(X=atac.X @ biadj, obs=atac.obs, var=rna.var, uns=atac.uns)

# %% [markdown]
# ## MET-to-RNA

# %%
biadj = nx.algorithms.bipartite.biadjacency_matrix(graph, met.var_names, rna.var_names)
met2rna = anndata.AnnData(X=met.X @ biadj, obs=met.obs, var=rna.var, uns=met.uns)

# %% [markdown]
# # Save results

# %%
rna.write(f"{PATH}/rna.h5ad", compression="gzip")
met.write(f"{PATH}/met.h5ad", compression="gzip")
atac.write(f"{PATH}/atac.h5ad", compression="gzip")
atac2rna.write(f"{PATH}/atac2rna.h5ad", compression="gzip")
met2rna.write(f"{PATH}/met2rna.h5ad", compression="gzip")
