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
import operator
import os
from math import ceil

import anndata
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
import scipy.stats
import seaborn as sns
from matplotlib import rcParams
from networkx.algorithms.bipartite import biadjacency_matrix
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform

import scglue
import utils

# %%
scglue.plot.set_publication_params()
rcParams["figure.figsize"] = (4, 4)

PATH = "s04_infer_gene_tf"
os.makedirs(PATH, exist_ok=True)

np.random.seed(0)

# %% [markdown]
# # Read data

# %%
rna = anndata.read_h5ad("s01_preprocessing/rna.h5ad")
atac = anndata.read_h5ad("s01_preprocessing/atac.h5ad")

# %%
genes = scglue.genomics.Bed(rna.var.assign(name=rna.var_names).query("dcq_highly_variable"))
peaks = scglue.genomics.Bed(atac.var.assign(name=atac.var_names).query("dcq_highly_variable"))
tss = genes.strand_specific_start_site()
promoters = tss.expand(2000, 0)
flanks = tss.expand(500, 500)

# %%
dist_graph = nx.read_graphml("s01_preprocessing/dist.graphml.gz")  # Serves as genomic windows
pchic_graph = nx.read_graphml("s01_preprocessing/pchic.graphml.gz")
eqtl_graph = nx.read_graphml("s01_preprocessing/eqtl.graphml.gz")

# %%
chip = scglue.genomics.read_bed("../../data/chip/ENCODE/TF-human/combined-GRCh38.bed.gz")
tfs = scglue.genomics.Bed(rna.var.loc[np.intersect1d(np.unique(chip["name"]), rna.var_names), :])
tfs.index.name = "tfs"

# %% [markdown]
# # SCENIC: coexpression network

# %%
rna[:, np.union1d(genes.index, tfs.index)].write_loom(f"{PATH}/rna.loom")
np.savetxt(f"{PATH}/tfs.txt", tfs.index, fmt="%s")

# %% tags=[]
# !pyscenic grn {PATH}/rna.loom {PATH}/tfs.txt \
#     -o {PATH}/scenic_grn.csv --seed 0 --num_workers 20 \
#     --cell_id_attribute cells --gene_attribute genes

# %%
scenic_grn = pd.read_csv(f"{PATH}/scenic_grn.csv")
orphan_tfs = set(tfs.index).difference(genes.index)  # When treated as target genes cannot be included in cis-regulatory rankings
scenic_grn = scenic_grn.loc[[item not in orphan_tfs for item in scenic_grn["target"]], :]
scenic_grn.to_csv(f"{PATH}/scenic_grn.csv", index=False)

# %% [markdown]
# # Gene-peak connection

# %% [markdown]
# ## Distance

# %%
dist = biadjacency_matrix(dist_graph, genes.index, peaks.index, weight="dist", dtype=np.float32)

# %% [markdown]
# ## pcHi-C

# %%
pchic = biadjacency_matrix(pchic_graph, genes.index, peaks.index, weight="weight", dtype=np.float32)

# %% [markdown]
# ## eQTL

# %%
eqtl = biadjacency_matrix(eqtl_graph, genes.index, peaks.index, weight="weight", dtype=np.float32)

# %% [markdown]
# ## Correlation

# %%
corr = biadjacency_matrix(
    utils.metacell_corr(
        rna, atac, "X_pca", n_meta=200,
        skeleton=dist_graph.subgraph([*genes.index, *peaks.index]), method="spr"
    ), genes.index, peaks.index, weight="corr", dtype=np.float32
)

# %% [markdown]
# ## GLUE

# %%
feature_embeddings = [
    pd.read_csv(f"s02_glue/prior:dcq/seed:{i}/feature_embeddings.csv", header=None, index_col=0)
    for i in range(4)
]

# %%
glue_graph = scglue.genomics.regulatory_inference(
    feature_embeddings[0].index,
    [feature_embedding.to_numpy() for feature_embedding in feature_embeddings],
    dist_graph.subgraph([*genes.index, *peaks.index]),
    alternative="greater", random_state=0
)
glue = biadjacency_matrix(glue_graph, genes.index, peaks.index, weight="score", dtype=np.float32)
qval = biadjacency_matrix(glue_graph, genes.index, peaks.index, weight="qval", dtype=np.float32)

# %% [markdown]
# ## Windowing

# %%
window = biadjacency_matrix(
    dist_graph, genes.index, peaks.index, weight=None, dtype=np.float32
).tocoo()

dist = window.multiply(dist.toarray())
pchic = window.multiply(pchic.toarray())
eqtl = window.multiply(eqtl.toarray())
corr = window.multiply(corr.toarray())
glue = window.multiply(glue.toarray())
qval = window.multiply(qval.toarray())

for mat in (dist, pchic, eqtl, corr, glue, qval):
    assert np.all(window.row == mat.row)
    assert np.all(window.col == mat.col)

# %%
gene_peak_conn = pd.DataFrame({
    "gene": genes.index[window.row],
    "peak": peaks.index[window.col],
    "dist": dist.data.astype(int),
    "pchic": pchic.data.astype(bool),
    "eqtl": eqtl.data.astype(bool),
    "corr": corr.data,
    "glue": glue.data,
    "qval": qval.data
})
# gene_peak_conn["pchic"] = pd.Categorical(gene_peak_conn["pchic"], categories=[False, True])
# gene_peak_conn["eqtl"] = pd.Categorical(gene_peak_conn["eqtl"], categories=[False, True])
gene_peak_conn.to_pickle(f"{PATH}/gene_peak_conn.pkl.gz")

# %% [markdown]
# # Filtering gene-peak connection

# %% [markdown]
# ## GLUE

# %%
qval_cutoff = 0.05
_ = sns.scatterplot(
    x="glue", y="qval", data=gene_peak_conn.sample(n=2000),
    edgecolor=None, s=5, alpha=0.1
).axhline(y=qval_cutoff, c="darkred", ls="--")

# %%
gene_peak_conn_glue = gene_peak_conn.query(f"qval < {qval_cutoff}")
gene_peak_conn_glue.shape[0]

# %%
frac_pos = gene_peak_conn_glue.shape[0] / gene_peak_conn.shape[0]
frac_pos

# %%
glue_cutoff = gene_peak_conn.query(f"qval < {qval_cutoff}")["glue"].min()
glue_cutoff

# %%
g = sns.jointplot(
    x="corr", y="glue", hue="pchic", data=gene_peak_conn.sample(frac=0.3),
    kind="kde", height=5,
    joint_kws=dict(common_norm=False, levels=6),
    marginal_kws=dict(common_norm=False, fill=True)
).plot_joint(sns.scatterplot, s=1, edgecolor=None, alpha=0.5, rasterized=True)
g.ax_marg_y.axhline(y=glue_cutoff, ls="--", c="darkred")
g.ax_joint.axhline(y=glue_cutoff, ls="--", c="darkred")
g.ax_joint.set_xlabel("Spearman correlation")
g.ax_joint.set_ylabel("GLUE regulatory score")
g.ax_joint.get_legend().set_title("pcHi-C")
g.fig.savefig(f"{PATH}/corr_glue_pchic_glue_cutoff.pdf")

# %%
g = sns.jointplot(
    x="corr", y="glue", hue="eqtl", data=gene_peak_conn.sample(frac=0.3),
    kind="kde", height=5,
    joint_kws=dict(common_norm=False, levels=6),
    marginal_kws=dict(common_norm=False, fill=True)
).plot_joint(sns.scatterplot, s=1, edgecolor=None, alpha=0.5, rasterized=True)
g.ax_marg_y.axhline(y=glue_cutoff, ls="--", c="darkred")
g.ax_joint.axhline(y=glue_cutoff, ls="--", c="darkred")
g.ax_joint.set_xlabel("Spearman correlation")
g.ax_joint.set_ylabel("GLUE regulatory score")
g.ax_joint.get_legend().set_title("eQTL")
g.fig.savefig(f"{PATH}/corr_glue_eqtl_glue_cutoff.pdf")

# %%
glue_all_links = gene_peak_conn.loc[:, ["gene", "peak", "glue"]].merge(
    tss.df.iloc[:, :4], how="left", left_on="gene", right_index=True
).merge(
    peaks.df.iloc[:, :4], how="left", left_on="peak", right_index=True
).loc[:, [
    "chrom_x", "chromStart_x", "chromEnd_x",
    "chrom_y", "chromStart_y", "chromEnd_y",
    "glue", "gene"
]]
glue_all_links.to_csv(f"{PATH}/glue_all.annotated_links", sep="\t", index=False, header=False)
del glue_all_links

# %%
glue_links = gene_peak_conn_glue.loc[:, ["gene", "peak", "glue"]].merge(
    tss.df.iloc[:, :4], how="left", left_on="gene", right_index=True
).merge(
    peaks.df.iloc[:, :4], how="left", left_on="peak", right_index=True
).loc[:, [
    "chrom_x", "chromStart_x", "chromEnd_x",
    "chrom_y", "chromStart_y", "chromEnd_y",
    "glue", "gene"
]]
glue_links.to_csv(f"{PATH}/glue.annotated_links", sep="\t", index=False, header=False)
del glue_links

# %% [markdown]
# ## Distance

# %%
dist_cutoff = np.quantile(gene_peak_conn["dist"], frac_pos)
dist_cutoff

# %%
gene_peak_conn_dist = gene_peak_conn.query(f"dist < {dist_cutoff}")
gene_peak_conn_dist.shape[0]

# %% [markdown]
# ## pcHi-C

# %%
gene_peak_conn_pchic = gene_peak_conn.query("pchic")
gene_peak_conn_pchic.shape[0]

# %% [markdown]
# ## eQTL

# %%
gene_peak_conn_eqtl = gene_peak_conn.query("eqtl")
gene_peak_conn_eqtl.shape[0]

# %% [markdown]
# ## Correlation

# %%
corr_cutoff = np.quantile(gene_peak_conn["corr"], 1 - frac_pos)
corr_cutoff

# %%
g = sns.jointplot(
    x="corr", y="glue", hue="pchic", data=gene_peak_conn.sample(frac=0.3),
    kind="kde", height=5,
    joint_kws=dict(common_norm=False, levels=6),
    marginal_kws=dict(common_norm=False, fill=True)
).plot_joint(sns.scatterplot, s=1, edgecolor=None, alpha=0.5, rasterized=True)
g.ax_marg_x.axvline(x=corr_cutoff, ls="--", c="darkred")
g.ax_joint.axvline(x=corr_cutoff, ls="--", c="darkred")
g.ax_joint.set_xlabel("Spearman correlation")
g.ax_joint.set_ylabel("GLUE regulatory score")
g.ax_joint.get_legend().set_title("pcHi-C")
g.fig.savefig(f"{PATH}/corr_glue_pchic_corr_cutoff.pdf")

# %%
g = sns.jointplot(
    x="corr", y="glue", hue="eqtl", data=gene_peak_conn.sample(frac=0.3),
    kind="kde", height=5,
    joint_kws=dict(common_norm=False, levels=6),
    marginal_kws=dict(common_norm=False, fill=True)
).plot_joint(sns.scatterplot, s=1, edgecolor=None, alpha=0.5, rasterized=True)
g.ax_marg_x.axvline(x=corr_cutoff, ls="--", c="darkred")
g.ax_joint.axvline(x=corr_cutoff, ls="--", c="darkred")
g.ax_joint.set_xlabel("Spearman correlation")
g.ax_joint.set_ylabel("GLUE regulatory score")
g.ax_joint.get_legend().set_title("eQTL")
g.fig.savefig(f"{PATH}/corr_glue_eqtl_corr_cutoff.pdf")

# %%
gene_peak_conn_corr = gene_peak_conn.query(f"corr > {corr_cutoff}")
gene_peak_conn_corr.shape[0]

# %% [markdown]
# # TF binding

# %% [markdown]
# ## Flanks

# %%
flank_tf_binding = scglue.genomics.window_graph(flanks, chip, 0, right_sorted=True)
flank_tf_binding = nx.to_pandas_edgelist(flank_tf_binding, source="flank", target="tf")
flank_tf_binding.shape

# %%
s = set(tfs.index)
flank_tf_binding = flank_tf_binding.loc[[item in s for item in flank_tf_binding["tf"]], :]
flank_tf_binding.shape

# %%
flank_tf_binding.to_pickle(f"{PATH}/flank_tf_binding.pkl.gz")
# flank_tf_binding = pd.read_pickle(f"{PATH}/flank_tf_binding.pkl.gz")

# %% [markdown]
# ## Peaks

# %%
peak_tf_binding = scglue.genomics.window_graph(peaks, chip, 0, right_sorted=True)
peak_tf_binding = nx.to_pandas_edgelist(peak_tf_binding, source="peak", target="tf")
peak_tf_binding.shape

# %%
s = set(tfs.index)
peak_tf_binding = peak_tf_binding.loc[[item in s for item in peak_tf_binding["tf"]], :]
peak_tf_binding.shape

# %%
peak_tf_binding.to_pickle(f"{PATH}/peak_tf_binding.pkl.gz")
# peak_tf_binding = pd.read_pickle(f"{PATH}/peak_tf_binding.pkl.gz")

# %% [markdown]
# # Cis-regulatory ranking

# %% [markdown]
# ## Flank

# %%
observed_flank_tf = scipy.sparse.coo_matrix((
    np.ones(flank_tf_binding.shape[0], dtype=np.int16), (
        flanks.index.get_indexer(flank_tf_binding["flank"]),
        tfs.index.get_indexer(flank_tf_binding["tf"]),
    )
), shape=(flanks.index.size, tfs.index.size)).toarray()

# %%
rank_flank_tf = pd.DataFrame(
    scipy.stats.rankdata(-observed_flank_tf, axis=0),
    index=flanks.index, columns=tfs.index
)
rank_flank_tf.iloc[:5, :5]

# %% [markdown]
# ## Distance

# %%
enrichment_gene_tf_dist, rank_gene_tf_dist = utils.cis_regulatory_ranking(
    gene_peak_conn_dist, peak_tf_binding,
    genes, peaks, tfs, n_samples=1000, random_seed=0
)

# %%
enrichment_gene_tf_dist.to_pickle(f"{PATH}/enrichment_gene_tf_dist.pkl.gz")
rank_gene_tf_dist.to_pickle(f"{PATH}/rank_gene_tf_dist.pkl.gz")
# enrichment_gene_tf_dist = pd.read_pickle(f"{PATH}/enrichment_gene_tf_dist.pkl.gz")
# rank_gene_tf_dist = pd.read_pickle(f"{PATH}/rank_gene_tf_dist.pkl.gz")

# %% [markdown]
# ## pcHi-C

# %%
enrichment_gene_tf_pchic, rank_gene_tf_pchic = utils.cis_regulatory_ranking(
    gene_peak_conn_pchic, peak_tf_binding,
    genes, peaks, tfs, n_samples=1000, random_seed=0
)

# %%
enrichment_gene_tf_pchic.to_pickle(f"{PATH}/enrichment_gene_tf_pchic.pkl.gz")
rank_gene_tf_pchic.to_pickle(f"{PATH}/rank_gene_tf_pchic.pkl.gz")
# enrichment_gene_tf_pchic = pd.read_pickle(f"{PATH}/enrichment_gene_tf_pchic.pkl.gz")
# rank_gene_tf_pchic = pd.read_pickle(f"{PATH}/rank_gene_tf_pchic.pkl.gz")

# %% [markdown]
# ## eQTL

# %%
enrichment_gene_tf_eqtl, rank_gene_tf_eqtl = utils.cis_regulatory_ranking(
    gene_peak_conn_eqtl, peak_tf_binding,
    genes, peaks, tfs, n_samples=1000, random_seed=0
)

# %%
enrichment_gene_tf_eqtl.to_pickle(f"{PATH}/enrichment_gene_tf_eqtl.pkl.gz")
rank_gene_tf_eqtl.to_pickle(f"{PATH}/rank_gene_tf_eqtl.pkl.gz")
# enrichment_gene_tf_eqtl = pd.read_pickle(f"{PATH}/enrichment_gene_tf_eqtl.pkl.gz")
# rank_gene_tf_eqtl = pd.read_pickle(f"{PATH}/rank_gene_tf_eqtl.pkl.gz")

# %% [markdown]
# ## Correlation

# %%
enrichment_gene_tf_corr, rank_gene_tf_corr = utils.cis_regulatory_ranking(
    gene_peak_conn_corr, peak_tf_binding,
    genes, peaks, tfs, n_samples=1000, random_seed=0
)

# %%
enrichment_gene_tf_corr.to_pickle(f"{PATH}/enrichment_gene_tf_corr.pkl.gz")
rank_gene_tf_corr.to_pickle(f"{PATH}/rank_gene_tf_corr.pkl.gz")
# enrichment_gene_tf_corr = pd.read_pickle(f"{PATH}/enrichment_gene_tf_corr.pkl.gz")
# rank_gene_tf_corr = pd.read_pickle(f"{PATH}/rank_gene_tf_corr.pkl.gz")

# %% [markdown]
# ## GLUE

# %%
enrichment_gene_tf_glue, rank_gene_tf_glue = utils.cis_regulatory_ranking(
    gene_peak_conn_glue, peak_tf_binding,
    genes, peaks, tfs, n_samples=1000, random_seed=0
)

# %%
enrichment_gene_tf_glue.to_pickle(f"{PATH}/enrichment_gene_tf_glue.pkl.gz")
rank_gene_tf_glue.to_pickle(f"{PATH}/rank_gene_tf_glue.pkl.gz")
# enrichment_gene_tf_glue = pd.read_pickle(f"{PATH}/enrichment_gene_tf_glue.pkl.gz")
# rank_gene_tf_glue = pd.read_pickle(f"{PATH}/rank_gene_tf_glue.pkl.gz")

# %% [markdown]
# # SCENIC: cisTarget pruning

# %%
ctx_annotation = pd.concat([
    pd.DataFrame({
        "#motif_id": tfs.index + "_atac",
        "gene_name": tfs.index
    }),
    pd.DataFrame({
        "#motif_id": tfs.index + "_flank",
        "gene_name": tfs.index
    })
]).assign(
    motif_similarity_qvalue=0.0,
    orthologous_identity=1.0,
    description="placeholder"
)
ctx_annotation.to_csv(f"{PATH}/ctx_annotation.tsv", sep="\t", index=False)

# %%
flank_feather = rank_flank_tf.T
flank_feather = flank_feather.loc[np.unique(flank_feather.index), np.unique(flank_feather.columns)].astype(np.int16)
flank_feather.index += "_flank"
flank_feather.index.name = "features"
flank_feather.columns.name = None
flank_feather = flank_feather.reset_index()
flank_feather.to_feather(f"{PATH}/flank_ctx_ranking.feather")

# %% tags=[]
# !pyscenic ctx {PATH}/scenic_grn.csv \
#     {PATH}/flank_ctx_ranking.feather \
#     --annotations_fname {PATH}/ctx_annotation.tsv \
#     --expression_mtx_fname {PATH}/rna.loom \
#     --output {PATH}/scenic_flank_reg.csv \
#     --rank_threshold 1500 \
#     --min_genes 6 \
#     --num_workers 20 \
#     --cell_id_attribute cells --gene_attribute genes 2> {PATH}/scenic_flank_reg.err

# %% tags=[]
flank_merged = pd.read_csv(f"{PATH}/scenic_flank_reg.csv", header=None, skiprows=3, usecols=[0, 8], names=["tf", "targets"])
flank_merged["targets"] = flank_merged["targets"].map(lambda x: set(i[0] for i in eval(x)))
flank_merged = flank_merged.groupby("tf").aggregate({"targets": lambda x: functools.reduce(set.union, x)})
flank_merged["n_targets"] = flank_merged["targets"].map(len)
flank_merged = flank_merged.sort_values("n_targets", ascending=False)
flank_merged

# %%
g = nx.DiGraph()
for tf, row in flank_merged.iterrows():
    for target in row["targets"]:
        g.add_edge(tf, target)
nx.set_node_attributes(g, "target", name="type")
for tf in flank_merged.index:
    g.nodes[tf]["type"] = "TF"
nx.write_graphml(g, f"{PATH}/flank_merged.graphml.gz")

# %% [markdown]
# ## Distance

# %%
dist_feather = rank_gene_tf_dist.T
dist_feather = dist_feather.loc[np.unique(dist_feather.index), np.unique(dist_feather.columns)].astype(np.int16)
dist_feather.index += "_atac"
dist_feather.index.name = "features"
dist_feather.columns.name = None
dist_feather = dist_feather.reset_index()
dist_feather.to_feather(f"{PATH}/dist_ctx_ranking.feather")

# %% tags=[]
# !pyscenic ctx {PATH}/scenic_grn.csv \
#     {PATH}/dist_ctx_ranking.feather {PATH}/flank_ctx_ranking.feather \
#     --annotations_fname {PATH}/ctx_annotation.tsv \
#     --expression_mtx_fname {PATH}/rna.loom \
#     --output {PATH}/scenic_dist_reg.csv \
#     --rank_threshold 1500 \
#     --min_genes 6 \
#     --num_workers 20 \
#     --cell_id_attribute cells --gene_attribute genes 2> {PATH}/scenic_dist_reg.err

# %% tags=[]
dist_merged = pd.read_csv(f"{PATH}/scenic_dist_reg.csv", header=None, skiprows=3, usecols=[0, 8], names=["tf", "targets"])
dist_merged["targets"] = dist_merged["targets"].map(lambda x: set(i[0] for i in eval(x)))
dist_merged = dist_merged.groupby("tf").aggregate({"targets": lambda x: functools.reduce(set.union, x)})
dist_merged["n_targets"] = dist_merged["targets"].map(len)
dist_merged = dist_merged.sort_values("n_targets", ascending=False)
dist_merged

# %%
g = nx.DiGraph()
for tf, row in dist_merged.iterrows():
    for target in row["targets"]:
        g.add_edge(tf, target)
nx.set_node_attributes(g, "target", name="type")
for tf in dist_merged.index:
    g.nodes[tf]["type"] = "TF"
nx.write_graphml(g, f"{PATH}/dist_merged.graphml.gz")

# %% [markdown]
# ## pcHi-C

# %%
pchic_feather = rank_gene_tf_pchic.T
pchic_feather = pchic_feather.loc[np.unique(pchic_feather.index), np.unique(pchic_feather.columns)].astype(np.int16)
pchic_feather.index += "_atac"
pchic_feather.index.name = "features"
pchic_feather.columns.name = None
pchic_feather = pchic_feather.reset_index()
pchic_feather.to_feather(f"{PATH}/pchic_ctx_ranking.feather")

# %% tags=[]
# !pyscenic ctx {PATH}/scenic_grn.csv \
#     {PATH}/pchic_ctx_ranking.feather {PATH}/flank_ctx_ranking.feather \
#     --annotations_fname {PATH}/ctx_annotation.tsv \
#     --expression_mtx_fname {PATH}/rna.loom \
#     --output {PATH}/scenic_pchic_reg.csv \
#     --rank_threshold 1500 \
#     --min_genes 6 \
#     --num_workers 20 \
#     --cell_id_attribute cells --gene_attribute genes 2> {PATH}/scenic_pchic_reg.err

# %% tags=[]
pchic_merged = pd.read_csv(f"{PATH}/scenic_pchic_reg.csv", header=None, skiprows=3, usecols=[0, 8], names=["tf", "targets"])
pchic_merged["targets"] = pchic_merged["targets"].map(lambda x: set(i[0] for i in eval(x)))
pchic_merged = pchic_merged.groupby("tf").aggregate({"targets": lambda x: functools.reduce(set.union, x)})
pchic_merged["n_targets"] = pchic_merged["targets"].map(len)
pchic_merged = pchic_merged.sort_values("n_targets", ascending=False)
pchic_merged

# %%
g = nx.DiGraph()
for tf, row in pchic_merged.iterrows():
    for target in row["targets"]:
        g.add_edge(tf, target)
nx.set_node_attributes(g, "target", name="type")
for tf in pchic_merged.index:
    g.nodes[tf]["type"] = "TF"
nx.write_graphml(g, f"{PATH}/pchic_merged.graphml.gz")

# %% [markdown]
# ## eQTL

# %%
eqtl_feather = rank_gene_tf_eqtl.T
eqtl_feather = eqtl_feather.loc[np.unique(eqtl_feather.index), np.unique(eqtl_feather.columns)].astype(np.int16)
eqtl_feather.index += "_atac"
eqtl_feather.index.name = "features"
eqtl_feather.columns.name = None
eqtl_feather = eqtl_feather.reset_index()
eqtl_feather.to_feather(f"{PATH}/eqtl_ctx_ranking.feather")

# %% tags=[]
# !pyscenic ctx {PATH}/scenic_grn.csv \
#     {PATH}/eqtl_ctx_ranking.feather {PATH}/flank_ctx_ranking.feather \
#     --annotations_fname {PATH}/ctx_annotation.tsv \
#     --expression_mtx_fname {PATH}/rna.loom \
#     --output {PATH}/scenic_eqtl_reg.csv \
#     --rank_threshold 1500 \
#     --min_genes 6 \
#     --num_workers 20 \
#     --cell_id_attribute cells --gene_attribute genes 2> {PATH}/scenic_eqtl_reg.err

# %% tags=[]
eqtl_merged = pd.read_csv(f"{PATH}/scenic_eqtl_reg.csv", header=None, skiprows=3, usecols=[0, 8], names=["tf", "targets"])
eqtl_merged["targets"] = eqtl_merged["targets"].map(lambda x: set(i[0] for i in eval(x)))
eqtl_merged = eqtl_merged.groupby("tf").aggregate({"targets": lambda x: functools.reduce(set.union, x)})
eqtl_merged["n_targets"] = eqtl_merged["targets"].map(len)
eqtl_merged = eqtl_merged.sort_values("n_targets", ascending=False)
eqtl_merged

# %%
g = nx.DiGraph()
for tf, row in eqtl_merged.iterrows():
    for target in row["targets"]:
        g.add_edge(tf, target)
nx.set_node_attributes(g, "target", name="type")
for tf in eqtl_merged.index:
    g.nodes[tf]["type"] = "TF"
nx.write_graphml(g, f"{PATH}/eqtl_merged.graphml.gz")

# %% [markdown]
# ## Correlation

# %%
corr_feather = rank_gene_tf_corr.T
corr_feather = corr_feather.loc[np.unique(corr_feather.index), np.unique(corr_feather.columns)].astype(np.int16)
corr_feather.index += "_atac"
corr_feather.index.name = "features"
corr_feather.columns.name = None
corr_feather = corr_feather.reset_index()
corr_feather.to_feather(f"{PATH}/corr_ctx_ranking.feather")

# %% tags=[]
# !pyscenic ctx {PATH}/scenic_grn.csv \
#     {PATH}/corr_ctx_ranking.feather {PATH}/flank_ctx_ranking.feather \
#     --annotations_fname {PATH}/ctx_annotation.tsv \
#     --expression_mtx_fname {PATH}/rna.loom \
#     --output {PATH}/scenic_corr_reg.csv \
#     --rank_threshold 1500 \
#     --min_genes 6 \
#     --num_workers 20 \
#     --cell_id_attribute cells --gene_attribute genes 2> {PATH}/scenic_corr_reg.err

# %% tags=[]
corr_merged = pd.read_csv(f"{PATH}/scenic_corr_reg.csv", header=None, skiprows=3, usecols=[0, 8], names=["tf", "targets"])
corr_merged["targets"] = corr_merged["targets"].map(lambda x: set(i[0] for i in eval(x)))
corr_merged = corr_merged.groupby("tf").aggregate({"targets": lambda x: functools.reduce(set.union, x)})
corr_merged["n_targets"] = corr_merged["targets"].map(len)
corr_merged = corr_merged.sort_values("n_targets", ascending=False)
corr_merged

# %%
g = nx.DiGraph()
for tf, row in corr_merged.iterrows():
    for target in row["targets"]:
        g.add_edge(tf, target)
nx.set_node_attributes(g, "target", name="type")
for tf in corr_merged.index:
    g.nodes[tf]["type"] = "TF"
nx.write_graphml(g, f"{PATH}/corr_merged.graphml.gz")

# %% [markdown]
# ## GLUE

# %%
glue_feather = rank_gene_tf_glue.T
glue_feather = glue_feather.loc[np.unique(glue_feather.index), np.unique(glue_feather.columns)].astype(np.int16)
glue_feather.index += "_atac"
glue_feather.index.name = "features"
glue_feather.columns.name = None
glue_feather = glue_feather.reset_index()
glue_feather.to_feather(f"{PATH}/glue_ctx_ranking.feather")

# %% tags=[]
# !pyscenic ctx {PATH}/scenic_grn.csv \
#     {PATH}/glue_ctx_ranking.feather {PATH}/flank_ctx_ranking.feather \
#     --annotations_fname {PATH}/ctx_annotation.tsv \
#     --expression_mtx_fname {PATH}/rna.loom \
#     --output {PATH}/scenic_glue_reg.csv \
#     --rank_threshold 1500 \
#     --min_genes 6 \
#     --num_workers 20 \
#     --cell_id_attribute cells --gene_attribute genes 2> {PATH}/scenic_glue_reg.err

# %% tags=[]
glue_merged = pd.read_csv(f"{PATH}/scenic_glue_reg.csv", header=None, skiprows=3, usecols=[0, 8], names=["tf", "targets"])
glue_merged["targets"] = glue_merged["targets"].map(lambda x: set(i[0] for i in eval(x)))
glue_merged = glue_merged.groupby("tf").aggregate({"targets": lambda x: functools.reduce(set.union, x)})
glue_merged["n_targets"] = glue_merged["targets"].map(len)
glue_merged = glue_merged.sort_values("n_targets", ascending=False)
glue_merged

# %%
g = nx.DiGraph()
for tf, row in glue_merged.iterrows():
    for target in row["targets"]:
        g.add_edge(tf, target)
nx.set_node_attributes(g, "target", name="type")
for tf in glue_merged.index:
    g.nodes[tf]["type"] = "TF"
nx.write_graphml(g, f"{PATH}/glue_merged.graphml.gz")

# %%
nx.to_pandas_edgelist(
    g, source="TF", target="Target gene"
).to_csv(f"{PATH}/glue_merged.csv", index=False)
