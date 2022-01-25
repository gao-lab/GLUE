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
from collections import defaultdict
from math import ceil

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
import scanpy as sc
import seaborn as sns
import scipy.stats
import statsmodels
import yaml
import matplotlib_venn
from matplotlib import rcParams
from sklearn.linear_model import LinearRegression
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm.notebook import tqdm

import scglue

# %%
scglue.plot.set_publication_params()
rcParams["figure.figsize"] = (4, 4)

PATH = "e02_comparison"
os.makedirs(PATH, exist_ok=True)

# %% [markdown]
# # Read data

# %%
rna = anndata.read_h5ad("s01_preprocessing/rna.h5ad")
met = anndata.read_h5ad("s01_preprocessing/met.h5ad")
atac = anndata.read_h5ad("s01_preprocessing/atac2rna.h5ad")

# %%
rna_latent = pd.read_csv(
    "e01_inmf/rna_latent.csv", header=None, index_col=0
).loc[rna.obs_names].to_numpy()
atac_latent = pd.read_csv(
    "e01_inmf/atac_latent.csv", header=None, index_col=0
).loc[atac.obs_names].to_numpy()
met_latent = pd.read_csv(
    "e01_inmf/met_latent.csv", header=None, index_col=0
).loc[met.obs_names].to_numpy()

# %%
rna.obsm["X_inmf"] = np.ascontiguousarray(rna_latent, dtype=np.float32)
atac.obsm["X_inmf"] = np.ascontiguousarray(atac_latent, dtype=np.float32)
met.obsm["X_inmf"] = np.ascontiguousarray(met_latent, dtype=np.float32)

# %%
combined = anndata.concat([rna, atac, met])
combined.uns["domain_colors"] = list(sns.color_palette(n_colors=3).as_hex())

# %% [markdown]
# # Visualization

# %%
sc.pp.neighbors(combined, use_rep="X_inmf", metric="cosine")
sc.tl.umap(combined)

# %%
rna.obsm["X_inmf_umap"] = combined[rna.obs_names].obsm["X_umap"]
atac.obsm["X_inmf_umap"] = combined[atac.obs_names].obsm["X_umap"]
met.obsm["X_inmf_umap"] = combined[met.obs_names].obsm["X_umap"]

# %%
fig, ax = plt.subplots()
ax.scatter(
    x=rna.obsm["X_inmf_umap"][:, 0], y=rna.obsm["X_inmf_umap"][:, 1],
    label="scRNA-seq", s=0.5, c=combined.uns["domain_colors"][0],
    edgecolor=None, rasterized=True
)
ax.scatter(
    x=atac.obsm["X_inmf_umap"][:, 0], y=atac.obsm["X_inmf_umap"][:, 1],
    label="scATAC-seq", s=1.0, c=combined.uns["domain_colors"][1],
    edgecolor=None, rasterized=True
)
ax.scatter(
    x=met.obsm["X_inmf_umap"][:, 0], y=met.obsm["X_inmf_umap"][:, 1],
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
fig.savefig(f"{PATH}/combined_domain.pdf")

# %%
fig = sc.pl.embedding(rna, "X_inmf_umap", color="cell_type", title="scRNA-seq cell type", return_fig=True)
fig.axes[0].set_xlabel("UMAP1")
fig.axes[0].set_ylabel("UMAP2")
fig.savefig(f"{PATH}/rna_ct.pdf")

# %%
fig = sc.pl.embedding(atac, "X_inmf_umap", color="cell_type", title="scATAC-seq cell type", return_fig=True)
fig.axes[0].set_xlabel("UMAP1")
fig.axes[0].set_ylabel("UMAP2")
fig.savefig(f"{PATH}/atac_ct.pdf")

# %%
fig = sc.pl.embedding(met, "X_inmf_umap", color="cell_type", title="snmC-seq cell type", return_fig=True)
fig.axes[0].set_xlabel("UMAP1")
fig.axes[0].set_ylabel("UMAP2")
fig.savefig(f"{PATH}/met_ct.pdf")

# %% [markdown]
# # Regulation contributions

# %%
rna_agg, met_agg, atac_agg = scglue.data.get_metacells(
    rna, met, atac, use_rep="X_inmf", n_meta=200, common=True, agg_kwargs=[
        {"X_agg": "sum"},
        {"X_agg": "mean"},
        {"X_agg": "sum"}
    ]
)

# %%
mCH_agg = met_agg[:, [item.endswith("_mCH") for item in met_agg.var_names]]
mCG_agg = met_agg[:, [item.endswith("_mCG") for item in met_agg.var_names]]
mCH_agg.var_names = mCH_agg.var_names.str.replace("_mCH", "")
mCG_agg.var_names = mCG_agg.var_names.str.replace("_mCG", "")

# %%
sc.pp.normalize_total(rna_agg)
sc.pp.log1p(rna_agg)

# sc.pp.normalize_total(mCH_agg)
sc.pp.log1p(mCH_agg)

# sc.pp.normalize_total(mCG_agg)
sc.pp.log1p(mCG_agg)

sc.pp.normalize_total(atac_agg)
sc.pp.log1p(atac_agg)

# %%
rna_agg_stat = pd.DataFrame({
    "mean": rna_agg.X.mean(axis=0).A1,
    "std": np.sqrt(scglue.num.col_var(rna_agg.X))
}, index=rna_agg.var_names)
mCH_agg_stat = pd.DataFrame({
    "mean": mCH_agg.X.mean(axis=0),
    "std": mCH_agg.X.std(axis=0)
}, index=mCH_agg.var_names)
mCG_agg_stat = pd.DataFrame({
    "mean": mCG_agg.X.mean(axis=0),
    "std": mCG_agg.X.std(axis=0)
}, index=mCG_agg.var_names)
atac_agg_stat = pd.DataFrame({
    "mean": atac_agg.X.mean(axis=0).A1,
    "std": np.sqrt(scglue.num.col_var(atac_agg.X))
}, index=atac_agg.var_names)

# %%
rna_agg_stat["std_lowess"] = lowess(rna_agg_stat["std"], rna_agg_stat["mean"], frac=0.3, return_sorted=False)
rna_agg_stat["std_remain"] = rna_agg_stat["std"] - rna_agg_stat["std_lowess"]

mCH_agg_stat["std_lowess"] = lowess(mCH_agg_stat["std"], mCH_agg_stat["mean"], frac=0.3, return_sorted=False)
mCH_agg_stat["std_remain"] = mCH_agg_stat["std"] - mCH_agg_stat["std_lowess"]

mCG_agg_stat["std_lowess"] = lowess(mCG_agg_stat["std"], mCG_agg_stat["mean"], frac=0.3, return_sorted=False)
mCG_agg_stat["std_remain"] = mCG_agg_stat["std"] - mCG_agg_stat["std_lowess"]

atac_agg_stat["std_lowess"] = lowess(atac_agg_stat["std"], atac_agg_stat["mean"], frac=0.3, return_sorted=False)
atac_agg_stat["std_remain"] = atac_agg_stat["std"] - atac_agg_stat["std_lowess"]

stats = [rna_agg_stat, mCH_agg_stat, mCG_agg_stat, atac_agg_stat]

# %%
fig, axes = plt.subplots(figsize=(16, 3), ncols=4, gridspec_kw=dict(wspace=0.5))
for ax, stat in zip(axes, stats):
    ax = sns.scatterplot(x="mean", y="std", data=stat, edgecolor=None, s=3, ax=ax)
    ax = sns.scatterplot(x="mean", y="std_lowess", data=stat, edgecolor=None, s=3, ax=ax)

# %%
mean_cutoffs = [0.7, 0.1, 0.1, 0.5]
std_cutoffs = [-0.02, -0.02, -0.02, -0.02]

# %%
fig, axes = plt.subplots(figsize=(16, 3), ncols=4, gridspec_kw=dict(wspace=0.5))
for ax, stat, mean_cutoff, std_cutoff in zip(axes, stats, mean_cutoffs, std_cutoffs):
    ax = sns.scatterplot(x="mean", y="std_remain", data=stat, edgecolor=None, s=3, ax=ax)
    ax.axvline(x=mean_cutoff, c="darkred", ls="--")
    ax.axhline(y=std_cutoff, c="darkred", ls="--")

# %%
rna_agg_use = rna_agg[:, np.logical_and(
    rna_agg_stat["mean"] >= mean_cutoffs[0],
    rna_agg_stat["std_remain"] >= std_cutoffs[0]
)]
mCH_agg_use = mCH_agg[:, np.logical_and(
    mCH_agg_stat["mean"] >= mean_cutoffs[1],
    mCH_agg_stat["std_remain"] >= std_cutoffs[1]
)]
mCG_agg_use = mCG_agg[:, np.logical_and(
    mCG_agg_stat["mean"] >= mean_cutoffs[2],
    mCG_agg_stat["std_remain"] >= std_cutoffs[2]
)]
atac_agg_use = atac_agg[:, np.logical_and(
    atac_agg_stat["mean"] >= mean_cutoffs[3],
    atac_agg_stat["std_remain"] >= std_cutoffs[3]
)]

# %%
common_genes = list(set(
    rna_agg_use.var_names
).intersection(
    mCH_agg_use.var_names
).intersection(
    mCG_agg_use.var_names
).intersection(
    atac_agg_use.var_names
))
len(common_genes)

# %%
rna_agg_use = rna_agg_use[:, common_genes].copy()
mCH_agg_use = mCH_agg_use[:, common_genes].copy()
mCG_agg_use = mCG_agg_use[:, common_genes].copy()
atac_agg_use = atac_agg_use[:, common_genes].copy()

# %%
rna_X = rna_agg_use.X.toarray()
mCH_X = mCH_agg_use.X
mCG_X = mCG_agg_use.X
atac_X = atac_agg_use.X.toarray()

# %%
corr = []
for i in tqdm(range(rna_X.shape[1])):
    corr.append([
        scipy.stats.spearmanr(rna_X[:, i], mCH_X[:, i]).correlation,
        scipy.stats.spearmanr(rna_X[:, i], mCG_X[:, i]).correlation,
        scipy.stats.spearmanr(rna_X[:, i], atac_X[:, i]).correlation,
    ])
corr = pd.DataFrame(corr, index=common_genes, columns=["mCH", "mCG", "ATAC"])
corr.head()

# %%
corr.mean(axis=0)


# %%
def offdiag_func(x, y, ax=None, **kwargs):
    ax = ax or plt.gca()
    ax.axvline(x=0, c="darkred", ls="--")
    ax.axhline(y=0, c="darkred", ls="--")

g = sns.pairplot(
    corr, diag_kind="kde", height=2,
    plot_kws=dict(s=3, edgecolor=None, alpha=0.5, rasterized=True)
).map_offdiag(offdiag_func)
g.savefig(f"{PATH}/corr_cmp.pdf")

# %%
gene_stat = rna_agg_stat.loc[common_genes, :].assign(
    gene_length=np.log10(
        rna.var.loc[common_genes, "chromEnd"] -
        rna.var.loc[common_genes, "chromStart"]
    )
)
gene_stat.head()

# %%
gene_stat_corr = pd.DataFrame(
    scglue.num.spr_mat(
        gene_stat.loc[:, ["gene_length", "mean", "std_remain"]], corr
    ), index=["Length", "Expr mean", "Expr variability"], columns=corr.columns
).abs()
gene_stat_corr.index.name = "Gene stat"
gene_stat_corr = gene_stat_corr.reset_index().melt(
    id_vars=["Gene stat"], var_name="Omics layer", value_name="Association"
)
gene_stat_corr

# %%
ax = sns.lineplot(
    x="Omics layer", y="Association", hue="Gene stat",
    data=gene_stat_corr, lw=2, legend=False
)
ax = sns.scatterplot(
    x="Omics layer", y="Association", hue="Gene stat",
    data=gene_stat_corr, edgecolor=None, ax=ax
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(title="Gene stat", loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False)
ax.get_figure().savefig(f"{PATH}/gene_stat_corr.pdf")

# %%
rsquare = defaultdict(list)

for i in range(rna_X.shape[1]):
    y_ = rna_X[:, i]

    X_ = np.stack([mCH_X[:, i], mCG_X[:, i], atac_X[:, i]], axis=1)
    lm = LinearRegression().fit(X_, y_)
    rsquare["Combined"].append(lm.score(X_, y_))

    X_ = np.expand_dims(mCH_X[:, i], axis=1)
    lm = LinearRegression().fit(X_, y_)
    rsquare["mCH"].append(lm.score(X_, y_))

    X_ = np.expand_dims(mCG_X[:, i], axis=1)
    lm = LinearRegression().fit(X_, y_)
    rsquare["mCG"].append(lm.score(X_, y_))

    X_ = np.expand_dims(atac_X[:, i], axis=1)
    lm = LinearRegression().fit(X_, y_)
    rsquare["ATAC"].append(lm.score(X_, y_))

rsquare = pd.DataFrame(rsquare, index=common_genes)
rsquare_melt = rsquare.melt(var_name="Omics layer", value_name="rsquare")

# %%
ax = sns.boxplot(
    x="Omics layer", y="rsquare", data=rsquare_melt,
    saturation=1.0, width=0.6, showmeans=True,
    meanprops=dict(marker="^", markerfacecolor="white", markeredgecolor="black"),
    boxprops=dict(edgecolor="black"), medianprops=dict(color="black"),
    whiskerprops=dict(color="black"), capprops=dict(color="black"),
    flierprops=dict(marker=".", markerfacecolor="black", markeredgecolor="none", markersize=3),
)
ax.set_ylabel("Gene expression $R^2$")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_figure().savefig(f"{PATH}/rsquare.pdf")

# %%
rsquare.mean(axis=0)
