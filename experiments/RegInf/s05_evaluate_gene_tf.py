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
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
from matplotlib import rcParams
from networkx.algorithms.bipartite import biadjacency_matrix

import scglue

# %%
scglue.plot.set_publication_params()
rcParams["figure.figsize"] = (4, 4)

PATH = "s05_evaluate_gene_tf"
os.makedirs(PATH, exist_ok=True)

# %% [markdown]
# # Prepare TRRUST v2

# %%
genes = anndata.read_h5ad(
    "s01_preprocessing/rna.h5ad", backed="r"
).var.query("dcq_highly_variable").index.to_numpy()

# %%
trrust = pd.read_table(
    "../../data/database/TRRUST-v2/trrust_rawdata.human.tsv",
    header=None, names=["tf", "target", "direction", "pmid"]
).query("direction != 'Repression'")
trrust.head()

# %%
tfs = np.loadtxt("s04_infer_gene_tf/tfs.txt", dtype=str)
tfs_used = np.intersect1d(tfs, trrust["tf"])
tfs_used.size

# %%
# oracle_tfs = np.loadtxt("e02_celloracle/tfs.txt", dtype=str)
# tfs_used = np.intersect1d(tfs_used, oracle_tfs)
# tfs_used.size

# %%
trrust = nx.from_pandas_edgelist(trrust, source="tf", target="target", create_using=nx.DiGraph)
trrust.number_of_edges()

# %%
trrust_mat = biadjacency_matrix(trrust, tfs_used, genes)
trrust_flag = trrust_mat.toarray().ravel()

# %% [markdown]
# # Fisher's test

# %%
pvals = {}

# %% [markdown]
# ## Distance

# %%
dist_merged = nx.read_graphml("s04_infer_gene_tf/dist_merged.graphml.gz")
dist_mat = biadjacency_matrix(dist_merged, tfs_used, genes)
dist_flag = dist_mat.toarray().ravel()

# %%
dist_crosstab = pd.crosstab(
    pd.Series(dist_flag, name="dist"),
    pd.Series(trrust_flag, name="trrust")
).iloc[::-1, ::-1]
dist_crosstab

# %%
pvals["dist"] = scipy.stats.fisher_exact(dist_crosstab)[1]
pvals["dist"]

# %% [markdown]
# ## pcHi-C

# %%
pchic_merged = nx.read_graphml("s04_infer_gene_tf/pchic_merged.graphml.gz")
pchic_mat = biadjacency_matrix(pchic_merged, tfs_used, genes)
pchic_flag = pchic_mat.toarray().ravel()

# %%
pchic_crosstab = pd.crosstab(
    pd.Series(pchic_flag, name="pchic"),
    pd.Series(trrust_flag, name="trrust")
).iloc[::-1, ::-1]
pchic_crosstab

# %%
pvals["pchic"] = scipy.stats.fisher_exact(pchic_crosstab)[1]
pvals["pchic"]

# %% [markdown]
# ## eQTL

# %%
eqtl_merged = nx.read_graphml("s04_infer_gene_tf/eqtl_merged.graphml.gz")
eqtl_mat = biadjacency_matrix(eqtl_merged, tfs_used, genes)
eqtl_flag = eqtl_mat.toarray().ravel()

# %%
eqtl_crosstab = pd.crosstab(
    pd.Series(eqtl_flag, name="eqtl"),
    pd.Series(trrust_flag, name="trrust")
).iloc[::-1, ::-1]
eqtl_crosstab

# %%
pvals["eqtl"] = scipy.stats.fisher_exact(eqtl_crosstab)[1]
pvals["eqtl"]

# %% [markdown]
# ## Correlation

# %%
corr_merged = nx.read_graphml("s04_infer_gene_tf/corr_merged.graphml.gz")
corr_mat = biadjacency_matrix(corr_merged, tfs_used, genes)
corr_flag = corr_mat.toarray().ravel()

# %%
corr_crosstab = pd.crosstab(
    pd.Series(corr_flag, name="corr"),
    pd.Series(trrust_flag, name="trrust")
).iloc[::-1, ::-1]
corr_crosstab

# %%
pvals["corr"] = scipy.stats.fisher_exact(corr_crosstab)[1]
pvals["corr"]

# %% [markdown]
# ## CellOracle

# %%
celloracle_merged = nx.read_graphml("e02_celloracle/celloracle.graphml.gz")
celloracle_mat = biadjacency_matrix(celloracle_merged, tfs_used, genes)
celloracle_flag = celloracle_mat.toarray().ravel()

# %%
celloracle_crosstab = pd.crosstab(
    pd.Series(celloracle_flag, name="celloracle"),
    pd.Series(trrust_flag, name="trrust")
).iloc[::-1, ::-1]
celloracle_crosstab

# %%
pvals["celloracle"] = scipy.stats.fisher_exact(celloracle_crosstab)[1]
pvals["celloracle"]

# %% [markdown]
# ## GLUE

# %%
glue_merged = nx.read_graphml("s04_infer_gene_tf/glue_merged.graphml.gz")
glue_mat = biadjacency_matrix(glue_merged, tfs_used, genes)
glue_flag = glue_mat.toarray().ravel()

# %%
glue_crosstab = pd.crosstab(
    pd.Series(glue_flag, name="glue"),
    pd.Series(trrust_flag, name="trrust")
).iloc[::-1, ::-1]
glue_crosstab

# %%
tf_idx, gene_idx = np.divmod(
    np.where(np.logical_and(glue_flag, trrust_flag))[0], 6000
)
pd.DataFrame({
    "TF": tfs_used[tf_idx],
    "gene": genes[gene_idx]
})

# %%
pvals["glue"] = scipy.stats.fisher_exact(glue_crosstab)[1]
pvals["glue"]

# %% [markdown]
# # Plot comparison

# %%
df = pd.DataFrame.from_dict(pvals, orient="index", columns=["pval"])
df.index.name = "conn"
df = df.reset_index()
df["conn"] = df["conn"].replace({
    "dist": "Distance", "pchic": "pcHi-C", "eqtl": "eQTL", "corr": "Correlation",
    "celloracle": "CellOracle", "glue": "GLUE"
})
df["nlog10_pval"] = -np.log10(df["pval"])

# %%
palette = sns.color_palette()
palette = [*palette[:4], palette[5], palette[4]]

# %%
ax = sns.barplot(x="conn", y="nlog10_pval", data=df, saturation=1.0, palette=palette)
ax.set_xlabel("Cis-regulatory region")
ax.set_ylabel("-log10 $P$-value")
for item in ax.get_xticklabels():
    item.set_rotation(67.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_figure().savefig(f"{PATH}/trrust_pval-with-celloracle.pdf")

# %%
palette = sns.color_palette()

# %%
ax = sns.barplot(x="conn", y="nlog10_pval", data=df.query('conn != "CellOracle"'), saturation=1.0, palette=palette)
ax.set_xlabel("Cis-regulatory region")
ax.set_ylabel("-log10 $P$-value")
for item in ax.get_xticklabels():
    item.set_rotation(67.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_figure().savefig(f"{PATH}/trrust_pval.pdf")
