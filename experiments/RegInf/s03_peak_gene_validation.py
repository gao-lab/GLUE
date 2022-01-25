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
import operator
import os

import anndata
import faiss
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats
import seaborn as sns
import sklearn.cluster
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.metrics
import yaml
from matplotlib import rcParams
from networkx.algorithms.bipartite import biadjacency_matrix

import scglue
import utils

# %%
scglue.plot.set_publication_params()
DIST_BINS = [0, 25, 50, 75, 100, 125, 150]  # in KB

PATH = "s03_peak_gene_validation"
os.makedirs(PATH, exist_ok=True)

np.random.seed(0)

# %%
with open("../../evaluation/config/display.yaml", "r") as f:
    palette = yaml.load(f, Loader=yaml.Loader)["palette"]
palette["Cicero"] = "#8C564B"
palette["Spearman"] = "#17BECF"
palette["LASSO"] = "#BCBD22"

# %% [markdown]
# # Read data

# %%
rna = anndata.read_h5ad("s01_preprocessing/rna.h5ad")
atac = anndata.read_h5ad("s01_preprocessing/atac.h5ad")

# %%
genes = scglue.genomics.Bed(rna.var.assign(name=rna.var_names).query("d_highly_variable"))
peaks = scglue.genomics.Bed(atac.var.assign(name=atac.var_names).query("d_highly_variable"))
tss = genes.strand_specific_start_site()
promoters = tss.expand(2000, 0)

# %%
dist_graph = nx.read_graphml("s01_preprocessing/dist.graphml.gz")
pchic_graph = nx.read_graphml("s01_preprocessing/pchic.graphml.gz")
eqtl_graph = nx.read_graphml("s01_preprocessing/eqtl.graphml.gz")

# %% [markdown]
# # Gene-peak connection

# %% [markdown]
# ## Distance

# %%
dist = biadjacency_matrix(dist_graph, genes.index, peaks.index, weight="dist", dtype=np.float32)

# %% [markdown]
# ## pcHi-C

# %%
pchic = biadjacency_matrix(pchic_graph, genes.index, peaks.index, weight=None, dtype=np.float32)

# %% [markdown]
# ## eQTL

# %%
eqtl = biadjacency_matrix(eqtl_graph, genes.index, peaks.index, weight=None, dtype=np.float32)

# %% [markdown]
# ## Correlation

# %%
corr = biadjacency_matrix(
    utils.metacell_corr(
        rna, atac, "X_pca", n_meta=200, skeleton=dist_graph, method="spr"
    ), genes.index, peaks.index, weight="corr", dtype=np.float32
)

# %% [markdown]
# ## Sparse regression

# %%
regr = utils.metacell_regr(
    rna, atac, "X_pca", n_meta=200, skeleton=dist_graph.reverse(),
    model="Lasso", alpha=0.01, random_state=0  # Optimized for AUROC and AP
)

# %%
regr = biadjacency_matrix(
    regr, peaks.index, genes.index, weight="regr", dtype=np.float32
).T

# %% [markdown]
# ## Cicero

# %%
cicero = pd.read_csv("e01_cicero/cicero_conns.csv.gz").dropna().query("coaccess != 0")
cicero["Peak1"] = cicero["Peak1"].str.split("_").map(lambda x: f"{x[0]}:{x[1]}-{x[2]}")
cicero["Peak2"] = cicero["Peak2"].str.split("_").map(lambda x: f"{x[0]}:{x[1]}-{x[2]}")
cicero.head()

# %%
peak_gene_mapping = scglue.genomics.window_graph(peaks, promoters, 0)
peak_gene_mapping = nx.DiGraph(peak_gene_mapping)
peak_gene_mapping = nx.to_pandas_edgelist(
    peak_gene_mapping, source="Peak1", target="Gene"
).loc[:, ["Peak1", "Gene"]]

# %%
cicero = pd.merge(cicero, peak_gene_mapping)
cicero = nx.from_pandas_edgelist(
    cicero.dropna(), source="Gene", target="Peak2",
    edge_attr="coaccess", create_using=nx.DiGraph
)
cicero = biadjacency_matrix(cicero, genes.index, peaks.index, weight="coaccess", dtype=np.float32)

# %% [markdown]
# ## GLUE

# %%
feature_embeddings = [
    pd.read_csv(f"s02_glue/prior:d/seed:{i}/feature_embeddings.csv", header=None, index_col=0)
    for i in range(4)
]

# %%
glue_list = [
    scglue.genomics.regulatory_inference(
        feature_embedding.index,
        feature_embedding.to_numpy(),
        dist_graph.subgraph([*genes.index, *peaks.index]),
        alternative="greater", random_state=0
    ) for feature_embedding in feature_embeddings
]
glue_list = [
    biadjacency_matrix(glue, genes.index, peaks.index, weight="score", dtype=np.float32)
    for glue in glue_list
]

# %%
glue = scglue.genomics.regulatory_inference(
    feature_embeddings[0].index,
    [feature_embedding.to_numpy() for feature_embedding in feature_embeddings],
    dist_graph.subgraph([*genes.index, *peaks.index]),
    alternative="greater", random_state=0
)
glue = biadjacency_matrix(glue, genes.index, peaks.index, weight="score", dtype=np.float32)

# %% [markdown]
# # Windowing

# %%
window = biadjacency_matrix(
    dist_graph, genes.index, peaks.index, weight=None, dtype=np.float32
).tocoo()

# %%
dist = window.multiply(dist.toarray())
pchic = window.multiply(pchic.toarray())
eqtl = window.multiply(eqtl.toarray())
cicero = window.multiply(cicero.toarray())
corr = window.multiply(corr.toarray())
regr = window.multiply(regr.toarray())

glue = window.multiply(glue.toarray())
glue_list = [window.multiply(item.toarray()) for item in glue_list]

for mat in (dist, pchic, eqtl, cicero, corr, glue):
    assert np.all(window.row == mat.row)
    assert np.all(window.col == mat.col)

# %%
df = pd.DataFrame({
    "dist": dist.data.astype(int),
    "pchic": pchic.data.astype(bool),
    "eqtl": eqtl.data.astype(bool),
    "cicero": cicero.data,
    "corr": corr.data,
    "regr": regr.data,

    "glue": glue.data,
    **{f"glue{i}": item.data for i, item in enumerate(glue_list)}
})
df["pchic"] = pd.Categorical(df["pchic"], categories=[False, True])
df["eqtl"] = pd.Categorical(df["eqtl"], categories=[False, True])
df["dist_bin"] = utils.make_dist_bins(df["dist"], bins=DIST_BINS)


# %% [markdown]
# # Comparisons

# %% [markdown]
# ## Different random seeds

# %%
def corrfunc(x, y, ax=None, **kwargs):
    r"""
    Adapted from https://stackoverflow.com/questions/50832204/show-correlation-values-in-pairplot-using-seaborn-in-python
    """
    r, _ = scipy.stats.pearsonr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f'r = {r:.3f}', xy=(0.1, 0.9), xycoords=ax.transAxes)


# %%
g = sns.pairplot(
    df.loc[:, [f"glue{i}" for i in range(4)]].rename(
        columns=lambda x: x.replace("glue", "Seed = ")
    ).sample(frac=0.3),
    diag_kind="kde", height=2,
    plot_kws=dict(s=1, edgecolor=None, alpha=0.2, rasterized=True)
).map_lower(corrfunc).map_upper(corrfunc)
g.fig.savefig(f"{PATH}/glue_pairplot.pdf")

# %% [markdown]
# ## GLUE vs genomic distance

# %%
ax = sns.boxplot(
    x="dist_bin", y="glue", data=df.sample(frac=0.3),
    color="grey", width=0.7, showmeans=True,
    meanprops=dict(marker="^", markerfacecolor="white", markeredgecolor="black"),
    boxprops=dict(edgecolor="black"), medianprops=dict(color="black"),
    whiskerprops=dict(color="black"), capprops=dict(color="black"),
    flierprops=dict(marker=".", markerfacecolor="black", markeredgecolor="none", markersize=3)
)
ax.set_xlabel("Genomic distance")
ax.set_ylabel("GLUE regulatory score")
for item in ax.get_xticklabels():
    item.set_rotation(67.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_figure().savefig(f"{PATH}/dist_binned_glue.pdf")

# %% [markdown]
# ## GLUE vs correlation

# %%
g = sns.jointplot(
    x="corr", y="glue", hue="pchic", data=df.sample(frac=0.3),
    kind="kde", height=5,
    joint_kws=dict(common_norm=False, levels=6),
    marginal_kws=dict(common_norm=False, fill=True)
).plot_joint(sns.scatterplot, s=1, edgecolor=None, alpha=0.5, rasterized=True)
g.ax_joint.set_xlabel("Spearman correlation")
g.ax_joint.set_ylabel("GLUE regulatory score")
g.ax_joint.get_legend().set_title("pcHi-C")
g.fig.savefig(f"{PATH}/corr_glue_pchic.pdf")

# %%
g = sns.jointplot(
    x="corr", y="glue", hue="eqtl", data=df.sample(frac=0.3),
    kind="kde", height=5,
    joint_kws=dict(common_norm=False, levels=6),
    marginal_kws=dict(common_norm=False, fill=True)
).plot_joint(sns.scatterplot, s=1, edgecolor=None, alpha=0.5, rasterized=True)
g.ax_joint.set_xlabel("Spearman correlation")
g.ax_joint.set_ylabel("GLUE regulatory score")
g.ax_joint.get_legend().set_title("eQTL")
g.fig.savefig(f"{PATH}/corr_glue_eqtl.pdf")

# %%
scipy.stats.spearmanr(df["corr"], df["glue"])

# %% [markdown]
# ## GLUE vs pcHi-C

# %%
g = utils.boxplot(x="dist_bin", y="glue", hue="pchic", data=df)
g.ax_joint.legend(loc="center left", bbox_to_anchor=(1.25, 0.5), frameon=False, title="pcHi-C")
g.ax_joint.set_xlabel("Genomic distance")
g.ax_joint.set_ylabel("GLUE regulatory score")
for item in g.ax_joint.get_xticklabels():
    item.set_rotation(67.5)
g.fig.savefig(f"{PATH}/dist_binned_glue_pchic.pdf")

# %% [markdown]
# ## GLUE vs eQTL

# %%
g = utils.boxplot(x="dist_bin", y="glue", hue="eqtl", data=df)
g.ax_joint.legend(loc="center left", bbox_to_anchor=(1.25, 0.5), frameon=False, title="eQTL")
g.ax_joint.set_xlabel("Genomic distance")
g.ax_joint.set_ylabel("GLUE regulatory score")
for item in g.ax_joint.get_xticklabels():
    item.set_rotation(67.5)
g.fig.savefig(f"{PATH}/dist_binned_glue_eqtl.pdf")

# %% [markdown]
# # ROC & PRC

# %%
rcParams["figure.figsize"] = (4, 4)

# %%
cicero_auroc = sklearn.metrics.roc_auc_score(df["pchic"].astype(bool), df["cicero"])
corr_auroc = sklearn.metrics.roc_auc_score(df["pchic"].astype(bool), df["corr"])
regr_auroc = sklearn.metrics.roc_auc_score(df["pchic"].astype(bool), df["regr"])
glue_auroc = sklearn.metrics.roc_auc_score(df["pchic"].astype(bool), df["glue"])
ax = scglue.plot.roc(df["pchic"].astype(bool), df["cicero"], label=f"Cicero (AUROC = {cicero_auroc:.3f})", color=palette["Cicero"])
ax = scglue.plot.roc(df["pchic"].astype(bool), df["corr"], label=f"Spearman (AUROC = {corr_auroc:.3f})", color=palette["Spearman"], ax=ax)
ax = scglue.plot.roc(df["pchic"].astype(bool), df["regr"], label=f"LASSO (AUROC = {regr_auroc:.3f})", color=palette["LASSO"], ax=ax)
ax = scglue.plot.roc(df["pchic"].astype(bool), df["glue"], label=f"GLUE (AUROC = {glue_auroc:.3f})", color=palette["GLUE"], ax=ax)
ax.set_title("pcHi-C prediction")
ax.axline((0, 0), (1, 1), ls="--", c="grey")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="lower right", bbox_to_anchor=(1.05, 0.0), fontsize="small", frameon=True)
ax.get_figure().savefig(f"{PATH}/pchic_roc.pdf")

# %%
cicero_ap = sklearn.metrics.average_precision_score(df["pchic"].astype(bool), df["cicero"])
corr_ap = sklearn.metrics.average_precision_score(df["pchic"].astype(bool), df["corr"])
regr_ap = sklearn.metrics.average_precision_score(df["pchic"].astype(bool), df["regr"])
glue_ap = sklearn.metrics.average_precision_score(df["pchic"].astype(bool), df["glue"])
ax = scglue.plot.prc(df["pchic"].astype(bool), df["cicero"], label=f"Cicero (AP = {cicero_ap:.3f})", color=palette["Cicero"])
ax = scglue.plot.prc(df["pchic"].astype(bool), df["corr"], label=f"Spearman (AP = {corr_ap:.3f})", color=palette["Spearman"], ax=ax)
ax = scglue.plot.prc(df["pchic"].astype(bool), df["regr"], label=f"LASSO (AP = {regr_ap:.3f})", color=palette["LASSO"], ax=ax)
ax = scglue.plot.prc(df["pchic"].astype(bool), df["glue"], label=f"GLUE (AP = {glue_ap:.3f})", color=palette["GLUE"], ax=ax)
ax.set_title("pcHi-C prediction")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.95), fontsize="small", frameon=True)
ax.get_figure().savefig(f"{PATH}/pchic_prc.pdf")

# %%
df["pchic"].astype(bool).sum() / df["pchic"].size

# %%
cicero_auroc = sklearn.metrics.roc_auc_score(df["eqtl"].astype(bool), df["cicero"])
corr_auroc = sklearn.metrics.roc_auc_score(df["eqtl"].astype(bool), df["corr"])
regr_auroc = sklearn.metrics.roc_auc_score(df["eqtl"].astype(bool), df["regr"])
glue_auroc = sklearn.metrics.roc_auc_score(df["eqtl"].astype(bool), df["glue"])
ax = scglue.plot.roc(df["eqtl"].astype(bool), df["cicero"], label=f"Cicero (AUROC = {cicero_auroc:.3f})", color=palette["Cicero"])
ax = scglue.plot.roc(df["eqtl"].astype(bool), df["corr"], label=f"Spearman (AUROC = {corr_auroc:.3f})", color=palette["Spearman"], ax=ax)
ax = scglue.plot.roc(df["eqtl"].astype(bool), df["regr"], label=f"LASSO (AUROC = {regr_auroc:.3f})", color=palette["LASSO"], ax=ax)
ax = scglue.plot.roc(df["eqtl"].astype(bool), df["glue"], label=f"GLUE (AUROC = {glue_auroc:.3f})", color=palette["GLUE"], ax=ax)
ax.set_title("eQTL prediction")
ax.axline((0, 0), (1, 1), ls="--", c="grey")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="lower right", bbox_to_anchor=(1.05, 0.0), fontsize="small", frameon=True)
ax.get_figure().savefig(f"{PATH}/eqtl_roc.pdf")

# %%
cicero_ap = sklearn.metrics.average_precision_score(df["eqtl"].astype(bool), df["cicero"])
corr_ap = sklearn.metrics.average_precision_score(df["eqtl"].astype(bool), df["corr"])
regr_ap = sklearn.metrics.average_precision_score(df["eqtl"].astype(bool), df["regr"])
glue_ap = sklearn.metrics.average_precision_score(df["eqtl"].astype(bool), df["glue"])
ax = scglue.plot.prc(df["eqtl"].astype(bool), df["cicero"], label=f"Cicero (AP = {cicero_ap:.3f})", color=palette["Cicero"])
ax = scglue.plot.prc(df["eqtl"].astype(bool), df["corr"], label=f"Spearman (AP = {corr_ap:.3f})", color=palette["Spearman"], ax=ax)
ax = scglue.plot.prc(df["eqtl"].astype(bool), df["regr"], label=f"LASSO (AP = {regr_ap:.3f})", color=palette["LASSO"], ax=ax)
ax = scglue.plot.prc(df["eqtl"].astype(bool), df["glue"], label=f"GLUE (AP = {glue_ap:.3f})", color=palette["GLUE"], ax=ax)
ax.set_title("eQTL prediction")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.95), fontsize="small", frameon=True)
ax.get_figure().savefig(f"{PATH}/eqtl_prc.pdf")

# %%
df["eqtl"].astype(bool).sum() / df["eqtl"].size
