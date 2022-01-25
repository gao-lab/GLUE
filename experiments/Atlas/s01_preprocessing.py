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
import functools
import gc
import itertools
import os

import anndata
import faiss
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib import rcParams
from sklearn.preprocessing import Normalizer
from sklearn.utils.extmath import randomized_svd

import scglue

# %%
scglue.plot.set_publication_params()
rcParams["figure.figsize"] = (7, 7)

PATH = "s01_preprocessing"
os.makedirs(PATH, exist_ok=True)

# %% [markdown]
# **NOTE:** Requires considerable amount of memory. Peak memory usage is ~200G.

# %% [markdown]
# # Read data

# %%
rna = anndata.read_h5ad("../../data/dataset/Cao-2020.h5ad", backed="r")
atac = anndata.read_h5ad("../../data/dataset/Domcke-2020.h5ad", backed="r")

# %% [markdown]
# # Organ balancing

# %%
rna_organ_fracs = rna.obs["Organ"].str.lower().value_counts() / rna.shape[0]
atac_organ_fracs = atac.obs["tissue"].str.lower().value_counts() / atac.shape[0]
cmp_organ_fracs = pd.DataFrame({"rna": rna_organ_fracs, "atac": atac_organ_fracs})

# %%
fig, ax = plt.subplots(figsize=(4, 4))
ax = sns.scatterplot(
    x="rna", y="atac", data=cmp_organ_fracs, ax=ax,
    edgecolor=None, s=25
)
ax.axline((0, 0), (0.45, 0.45), c="darkred", ls="--")
ax.set_xlabel("scRNA-seq")
ax.set_ylabel("scATAC-seq")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.savefig(f"{PATH}/cmp_organ_fracs.pdf", bbox_inches="tight")

# %%
organ_min_fracs = cmp_organ_fracs.min(axis=1)
organ_min_fracs

# %% [markdown]
# ## Subsample mask

# %% [markdown]
# Data subsampled with this mask have balanced organ composition across RNA and ATAC.

# %% tags=[]
rs = np.random.RandomState(0)
rna_subidx, atac_subidx = [], []
for organ, min_frac in organ_min_fracs.iteritems():
    print(f"Dealing with {organ}...")
    rna_idx = np.where(rna.obs["Organ"].str.lower() == organ)[0]
    rna_subidx.append(rs.choice(rna_idx, round(min_frac * rna.shape[0]), replace=False))
    atac_idx = np.where(atac.obs["tissue"].str.lower() == organ)[0]
    atac_subidx.append(rs.choice(atac_idx, round(min_frac * atac.shape[0]), replace=False))
rna_subidx = np.concatenate(rna_subidx)
rna_mask = np.zeros(rna.shape[0], dtype=bool)
rna_mask[rna_subidx] = True
rna.obs["mask"] = rna_mask
atac_subidx = np.concatenate(atac_subidx)
atac_mask = np.zeros(atac.shape[0], dtype=bool)
atac_mask[atac_subidx] = True
atac.obs["mask"] = atac_mask

# %% [markdown]
# ## Balancing weights

# %%
rna_organ_balancing = np.sqrt(cmp_organ_fracs["atac"] / cmp_organ_fracs["rna"])
atac_organ_balancing = np.sqrt(cmp_organ_fracs["rna"] / cmp_organ_fracs["atac"])

# %%
rna.obs["organ_balancing"] = rna_organ_balancing.loc[rna.obs["Organ"].str.lower()].to_numpy()
atac.obs["organ_balancing"] = atac_organ_balancing.loc[atac.obs["tissue"].str.lower()].to_numpy()

# %% [markdown]
# # RNA

# %%
rna = rna.to_memory()

# %% [markdown]
# ## Gene selection

# %% tags=[]
hvg_df = sc.pp.highly_variable_genes(rna[rna.obs["mask"], :], n_top_genes=4000, flavor="seurat_v3", inplace=False)
rna.var = rna.var.assign(**hvg_df.to_dict(orient="series"))

# %% [markdown]
# ## Dimension reduction

# %% [markdown]
# We derive PCA from the subsampled (organ balanced) data, and then apply the transformation on the full dataset.

# %%
rna.layers["raw_counts"] = rna.X.copy()
sc.pp.normalize_total(rna)
sc.pp.log1p(rna)
rna = rna[:, rna.var.highly_variable]
gc.collect()

# %%
rna.write(f"{PATH}/rna_ckpt1.h5ad", compression="gzip")
# rna = anndata.read_h5ad(f"{PATH}/rna_ckpt1.h5ad")

# %%
X = rna.X
X_masked = X[rna.obs["mask"]]
mean = X_masked.mean(axis=0).A1
std = np.sqrt(X_masked.power(2).mean(axis=0).A1 - mean ** 2)
X = X.toarray()
X -= mean
X /= std
X = X.clip(-10, 10)
X_masked = X[rna.obs["mask"]]

# %%
u, s, vh = randomized_svd(X_masked.T @ X_masked, 100, n_iter=15, random_state=0)
rna.obsm["X_pca"] = X @ vh.T

# %%
rna.X = rna.layers["raw_counts"]
del rna.layers["raw_counts"], X, X_masked, mean, std, u, s, vh
gc.collect()

# %%
rna.write(f"{PATH}/rna_ckpt2.h5ad", compression="gzip")
# rna = anndata.read_h5ad(f"{PATH}/rna_ckpt2.h5ad")

# %%
sc.pp.neighbors(rna, n_pcs=rna.obsm["X_pca"].shape[1], metric="cosine")
sc.tl.umap(rna)
del rna.obsp["connectivities"], rna.obsp["distances"]
gc.collect()

# %%
fig = sc.pl.umap(
    rna, color="cell_type", title="scRNA-seq cell type", return_fig=True,
    legend_loc="on data", legend_fontsize=4, legend_fontoutline=0.5
)
fig.savefig(f"{PATH}/rna_ct.pdf")

# %%
rna.write(f"{PATH}/rna_ckpt3.h5ad", compression="gzip")
# rna = anndata.read_h5ad(f"{PATH}/rna_ckpt3.h5ad")

# %% [markdown]
# ## Aggregate data

# %% [markdown]
# Will help accelerate model training

# %%
kmeans = faiss.Kmeans(rna.obsm["X_pca"].shape[1], 100000, gpu=True, seed=0)
kmeans.train(rna.obsm["X_pca"][rna.obs["mask"]])
_, rna.obs["metacell"] = kmeans.index.search(rna.obsm["X_pca"], 1)

# %%
rna.obs["metacell"] = pd.Categorical(rna.obs["metacell"])
rna.obs["metacell"].cat.rename_categories(lambda x: f"rna-metacell-{x}", inplace=True)
rna.obs["n_cells"] = 1

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
fig = sc.pl.umap(
    rna_agg, color="cell_type", title="scRNA-seq cell type (metacells)", return_fig=True,
    legend_loc="on data", legend_fontsize=4, legend_fontoutline=0.5
)
fig.savefig(f"{PATH}/rna_agg_ct.pdf")

# %% [markdown]
# ## Save data

# %%
rna.write(f"{PATH}/rna.h5ad", compression="gzip")
rna_agg.write(f"{PATH}/rna_agg.h5ad", compression="gzip")

# %% [markdown]
# # Prior graph

# %%
peaks = scglue.genomics.Bed(atac.var.assign(name=atac.var.index))
genes = scglue.genomics.Bed(rna.var.assign(name=rna.var.index))

# %%
graph = scglue.genomics.window_graph(
    genes.expand(2000, 0), peaks, 0,
    attr_fn=lambda l, r, d: {
        "weight": scglue.genomics.dist_power_decay(abs(d))
    }
)

# %%
hvg_reachable = scglue.graph.reachable_vertices(graph, rna.var.query("highly_variable").index)

# %%
graph = scglue.graph.compose_multigraph(graph, graph.reverse())
for item in itertools.chain(rna.var_names, atac.var_names):
    graph.add_edge(item, item, weight=1.0)
nx.set_edge_attributes(graph, 1, name="sign")

# %%
subgraph = graph.subgraph(hvg_reachable)

# %% tags=[]
nx.write_graphml(graph, f"{PATH}/full.graphml.gz")
nx.write_graphml(subgraph, f"{PATH}/sub.graphml.gz")

# %% [markdown]
# Make up some space...

# %%
del rna, rna_agg, graph, subgraph, peaks, genes
gc.collect()

# %% [markdown]
# # ATAC

# %%
atac = atac.to_memory()

# %%
atac.var["highly_variable"] = [item in hvg_reachable for item in atac.var_names]
atac.var["highly_variable"].sum()

# %% [markdown]
# ## Dimension reduction

# %% [markdown]
# We derive PCA from the subsampled (organ balanced) data, and then apply the transformation on the full dataset.

# %%
X = scglue.num.tfidf(atac.X)
X = Normalizer(norm="l1").fit_transform(X)
X = np.log1p(X * 1e4)

# %%
X_masked = X[atac.obs["mask"]]
u, s, vh = randomized_svd(X_masked, 100, n_iter=15, random_state=0)
X_lsi = X @ vh.T / s
X_lsi -= X_lsi.mean(axis=1, keepdims=True)
X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
atac.obsm["X_lsi"] = X_lsi.astype(np.float32)

# %%
atac = atac[:, atac.var["highly_variable"]]
del X, X_masked, X_lsi, u, s, vh
gc.collect()

# %%
atac.write(f"{PATH}/atac_ckpt1.h5ad", compression="gzip")
# atac = anndata.read_h5ad(f"{PATH}/atac_ckpt1.h5ad")

# %% tags=[]
sc.pp.neighbors(atac, n_pcs=atac.obsm["X_lsi"].shape[1], use_rep="X_lsi", metric="cosine")
sc.tl.umap(atac)
del atac.obsp["connectivities"], atac.obsp["distances"]
gc.collect()

# %%
fig = sc.pl.umap(
    atac, color="cell_type", title="scATAC-seq cell type", return_fig=True,
    legend_loc="on data", legend_fontsize=4, legend_fontoutline=0.5
)
fig.savefig(f"{PATH}/atac_ct.pdf")

# %%
atac.write(f"{PATH}/atac_ckpt2.h5ad", compression="gzip")
# atac = anndata.read_h5ad(f"{PATH}/atac_ckpt2.h5ad")

# %% [markdown]
# ## Aggregate data

# %% [markdown]
# Will help accelerate model training

# %%
kmeans = faiss.Kmeans(atac.obsm["X_lsi"].shape[1], 40000, gpu=True, seed=0)
kmeans.train(atac.obsm["X_lsi"][atac.obs["mask"]])
_, atac.obs["metacell"] = kmeans.index.search(atac.obsm["X_lsi"], 1)

# %%
atac.obs["metacell"] = pd.Categorical(atac.obs["metacell"])
atac.obs["metacell"].cat.rename_categories(lambda x: f"atac-metacell-{x}", inplace=True)
atac.obs["n_cells"] = 1

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

# %%
fig = sc.pl.umap(
    atac_agg, color="cell_type", title="scATAC-seq cell type (metacells)", return_fig=True,
    legend_loc="on data", legend_fontsize=4, legend_fontoutline=0.5
)
fig.savefig(f"{PATH}/atac_agg_ct.pdf")

# %% [markdown]
# ## Save data

# %% tags=[]
atac.write(f"{PATH}/atac.h5ad", compression="gzip")
atac_agg.write(f"{PATH}/atac_agg.h5ad", compression="gzip")
