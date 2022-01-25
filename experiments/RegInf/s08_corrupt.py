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
import pickle

import anndata
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib import rcParams
from networkx.algorithms.bipartite import biadjacency_matrix

import scglue
import utils

# %%
scglue.plot.set_publication_params()
rcParams["figure.figsize"] = (4, 4)

PATH = "s08_corrupt"
os.makedirs(PATH, exist_ok=True)

# %% [markdown]
# # Read data

# %%
rna = anndata.read_h5ad("s01_preprocessing/rna.h5ad")
atac = anndata.read_h5ad("s01_preprocessing/atac.h5ad")

# %% [markdown]
# # Select high-confidence cis-regulatory links

# %%
dist = nx.read_graphml("s01_preprocessing/dist.graphml.gz")
dist.number_of_edges()

# %%
eqtl = nx.read_graphml("s01_preprocessing/eqtl.graphml.gz")
eqtl.number_of_edges()

# %%
pchic = nx.read_graphml("s01_preprocessing/pchic.graphml.gz")
pchic.number_of_edges()

# %%
corr = utils.metacell_corr(
    rna, atac, "X_pca", n_meta=200,
    skeleton=dist, method="spr"
)
corr = corr.edge_subgraph(
    e for e, attr in dict(corr.edges).items()
    if attr["corr"] >= 0.7
)
corr.number_of_edges()

# %%
# peak_tf_binding = pd.read_pickle("s04_infer_gene_tf/peak_tf_binding.pkl.gz")
# trrust = pd.read_table(
#     "../../data/database/TRRUST-v2/trrust_rawdata.human.tsv",
#     header=None, names=["tf", "target", "direction", "pmid"]
# ).query("direction != 'Repression'")
# common_tfs = list(set(peak_tf_binding["tf"]).intersection(trrust["tf"]))
# peak_tf_binding = nx.from_pandas_edgelist(peak_tf_binding, source="tf", target="peak")
# trrust = nx.from_pandas_edgelist(trrust, source="target", target="tf")
# tfknown = biadjacency_matrix(
#     trrust, rna.var_names, common_tfs, weight=None
# ) @ biadjacency_matrix(
#     peak_tf_binding, common_tfs, atac.var_names, weight=None
# )
# tfknown = tfknown.tocoo()
# tfknown.eliminate_zeros()
# tfknown = pd.DataFrame({
#     "gene": rna.var_names[tfknown.row],
#     "peak": atac.var_names[tfknown.col]
# })
# tfknown = nx.from_pandas_edgelist(tfknown, source="gene", target="peak")
# tfknown.number_of_edges()  # Very little overlap with other connections

# %%
hiconf = set((e[0], e[1]) for e in eqtl.edges) & \
    set((e[0], e[1]) for e in pchic.edges) & \
    set((e[0], e[1]) for e in corr.edges)
len(hiconf)

# %% [markdown]
# # Corrupt prior graph

# %%
prior = nx.read_graphml("s01_preprocessing/dcq_prior.graphml.gz")
n_original_edges = prior.number_of_edges()

# %%
hiconf = {e for e in hiconf if prior.has_edge(e[0], e[1])}
len(hiconf)

# %%
hard_corrupted_prior = prior.edge_subgraph(
    e for e in prior.edges
    if (e[0], e[1]) not in hiconf and (e[1], e[0]) not in hiconf
)
n_corrupted_edges = hard_corrupted_prior.number_of_edges()
assert (n_original_edges - n_corrupted_edges) / 2 / 3 == len(hiconf)

# %%
soft_corrupted_prior = prior.edge_subgraph(
    e for e, attr in dict(prior.edges).items()
    if attr["type"] == "dist" or ((e[0], e[1]) not in hiconf and (e[1], e[0]) not in hiconf)
)
n_corrupted_edges = soft_corrupted_prior.number_of_edges()
assert (n_original_edges - n_corrupted_edges) / 2 / 2 == len(hiconf)

# %% [markdown]
# # Save results

# %%
with open(f"{PATH}/hiconf.pkl.gz", "wb") as f:
    pickle.dump(hiconf, f)

# %%
nx.write_graphml(hard_corrupted_prior, f"{PATH}/hard_corrupted_dcq_prior.graphml.gz")
nx.write_graphml(soft_corrupted_prior, f"{PATH}/soft_corrupted_dcq_prior.graphml.gz")
