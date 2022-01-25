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
import matplotlib.pyplot as plt
import matplotlib_venn
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams

import scglue

# %%
scglue.plot.set_publication_params()
rcParams["figure.figsize"] = (4, 4)

# %%
PATH = "s09_recovery"
os.makedirs(PATH, exist_ok=True)

# %% [markdown]
# # Read data

# %%
rna = anndata.read_h5ad("s01_preprocessing/rna.h5ad")
atac = anndata.read_h5ad("s01_preprocessing/atac.h5ad")

# %%
genes = rna.var.query("dcq_highly_variable")
peaks = atac.var.query("dcq_highly_variable")

# %%
dist_graph = nx.read_graphml("s01_preprocessing/dist.graphml.gz")

# %%
with open("s08_corrupt/hiconf.pkl.gz", "rb") as f:
    hiconf = pickle.load(f)

# %%
discover_rate = []

# %% [markdown]
# # Hard-corrupted

# %%
feature_embeddings = [
    pd.read_csv(f"s02_glue/prior:hard_corrupted_dcq/seed:{i}/feature_embeddings.csv", header=None, index_col=0)
    for i in range(4)
]
glue = scglue.genomics.regulatory_inference(
    feature_embeddings[0].index,
    [feature_embedding.to_numpy() for feature_embedding in feature_embeddings],
    dist_graph.subgraph([*genes.index, *peaks.index]),
    alternative="greater", random_state=0
)
glue_edgelist = nx.to_pandas_edgelist(glue)
glue_hiconf_edgelist = nx.to_pandas_edgelist(
    glue.edge_subgraph(hiconf)
)

# %%
hard_hiconf_called = {(r["source"], r["target"]) for _, r in glue_hiconf_edgelist.query("qval < 0.05").iterrows()}
hard_total_called = {(r["source"], r["target"]) for _, r in glue_edgelist.query("qval < 0.05").iterrows()}
len(hard_hiconf_called), len(hard_total_called)

# %%
discover_rate.append({
    "Corruption": "Hard",
    "Connections": "Corrupted",
    "Discovery rate": len(hard_hiconf_called) / glue_hiconf_edgelist.shape[0]
})
discover_rate[-1]["Discovery rate"]

# %%
discover_rate.append({
    "Corruption": "Hard",
    "Connections": "Overall",
    "Discovery rate": len(hard_total_called) / glue_edgelist.shape[0]
})
discover_rate[-1]["Discovery rate"]

# %% [markdown]
# # Soft-corrupted

# %%
feature_embeddings = [
    pd.read_csv(f"s02_glue/prior:soft_corrupted_dcq/seed:{i}/feature_embeddings.csv", header=None, index_col=0)
    for i in range(4)
]
glue = scglue.genomics.regulatory_inference(
    feature_embeddings[0].index,
    [feature_embedding.to_numpy() for feature_embedding in feature_embeddings],
    dist_graph.subgraph([*genes.index, *peaks.index]),
    alternative="greater", random_state=0
)
glue_edgelist = nx.to_pandas_edgelist(glue)
glue_hiconf_edgelist = nx.to_pandas_edgelist(
    glue.edge_subgraph(hiconf)
)

# %%
soft_hiconf_called = {(r["source"], r["target"]) for _, r in glue_hiconf_edgelist.query("qval < 0.05").iterrows()}
soft_total_called = {(r["source"], r["target"]) for _, r in glue_edgelist.query("qval < 0.05").iterrows()}
len(soft_hiconf_called), len(soft_total_called)

# %%
discover_rate.append({
    "Corruption": "Soft",
    "Connections": "Corrupted",
    "Discovery rate": len(soft_hiconf_called) / glue_hiconf_edgelist.shape[0]
})
discover_rate[-1]["Discovery rate"]

# %%
discover_rate.append({
    "Corruption": "Soft",
    "Connections": "Overall",
    "Discovery rate": len(soft_total_called) / glue_edgelist.shape[0]
})
discover_rate[-1]["Discovery rate"]

# %% [markdown]
# # Uncorrupted

# %%
feature_embeddings = [
    pd.read_csv(f"s02_glue/prior:dcq/seed:{i}/feature_embeddings.csv", header=None, index_col=0)
    for i in range(4)
]
glue = scglue.genomics.regulatory_inference(
    feature_embeddings[0].index,
    [feature_embedding.to_numpy() for feature_embedding in feature_embeddings],
    dist_graph.subgraph([*genes.index, *peaks.index]),
    alternative="greater", random_state=0
)
glue_edgelist = nx.to_pandas_edgelist(glue)
glue_hiconf_edgelist = nx.to_pandas_edgelist(
    glue.edge_subgraph(hiconf)
)

# %%
none_hiconf_called = {(r["source"], r["target"]) for _, r in glue_hiconf_edgelist.query("qval < 0.05").iterrows()}
none_total_called = {(r["source"], r["target"]) for _, r in glue_edgelist.query("qval < 0.05").iterrows()}
len(none_hiconf_called), len(none_total_called)

# %%
discover_rate.append({
    "Corruption": "None",
    "Connections": "Corrupted",
    "Discovery rate": len(none_hiconf_called) / glue_hiconf_edgelist.shape[0]
})
discover_rate[-1]["Discovery rate"]

# %%
discover_rate.append({
    "Corruption": "None",
    "Connections": "Overall",
    "Discovery rate": len(none_total_called) / glue_edgelist.shape[0]
})
discover_rate[-1]["Discovery rate"]

# %% [markdown]
# # Plotting

# %%
discover_rate = pd.DataFrame.from_records(discover_rate)
discover_rate

# %%
ax = sns.barplot(x="Corruption", y="Discovery rate", hue="Connections", data=discover_rate)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(title="Connections", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
ax.get_figure().savefig(f"{PATH}/discover_rate.pdf")

# %%
fig, ax = plt.subplots()
venn = matplotlib_venn.venn3(
    [hard_hiconf_called, soft_hiconf_called, none_hiconf_called],
    set_labels=["Hard-corrupted", "Soft-corrupted", "Uncorrupted"],
    set_colors=sns.color_palette(n_colors=3)[::-1], alpha=0.75, ax=ax
)
fig.savefig(f"{PATH}/venn.pdf")
