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

import anndata
import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats
import seaborn as sns
import statsmodels.api as sm
import yaml
from matplotlib import rcParams
from sklearn.linear_model import LinearRegression
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm.notebook import tqdm

import scglue

# %%
scglue.plot.set_publication_params()
rcParams["figure.figsize"] = (4, 4)

PATH = "s04_reg_contrib"
os.makedirs(PATH, exist_ok=True)

# %% [markdown]
# # Read data

# %%
rna = anndata.read_h5ad("s02_glue/rna.h5ad")
met = anndata.read_h5ad("s02_glue/met.h5ad")
atac = anndata.read_h5ad("s02_glue/atac2rna.h5ad")
combined = anndata.read_h5ad("s02_glue/combined.h5ad")

# %%
rna.obs["common_cell_type"] = anndata.read_h5ad("s03_markers/rna_filtered.h5ad", backed="r").obs["common_cell_type"]
met.obs["common_cell_type"] = anndata.read_h5ad("s03_markers/met_filtered.h5ad", backed="r").obs["common_cell_type"]
atac.obs["common_cell_type"] = anndata.read_h5ad("s03_markers/atac_filtered.h5ad", backed="r").obs["common_cell_type"]

# %%
with open("manual_colors.yaml", "r") as f:
    MANUAL_COLORS = yaml.load(f, Loader=yaml.FullLoader)

# %% [markdown]
# # Clustering metacells

# %%
rna_agg, met_agg, atac_agg = scglue.data.get_metacells(
    rna, met, atac, use_rep="X_glue", n_meta=200, common=True, agg_kwargs=[
        {"X_agg": "sum", "obs_agg": {"common_cell_type": "majority"}},
        {"X_agg": "mean", "obs_agg": {"common_cell_type": "majority"}},
        {"X_agg": "sum", "obs_agg": {"common_cell_type": "majority"}}
    ]
)

# %%
mCH_agg = met_agg[:, [item.endswith("_mCH") for item in met_agg.var_names]]
mCG_agg = met_agg[:, [item.endswith("_mCG") for item in met_agg.var_names]]
mCH_agg.var_names = mCH_agg.var_names.str.replace("_mCH", "")
mCG_agg.var_names = mCG_agg.var_names.str.replace("_mCG", "")

# %% [markdown]
# # Normalization and filtering

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

# %% [markdown]
# # Global

# %% [markdown]
# ## Correlation

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

# %% tags=[]
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

# %% [markdown]
# ## R square

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

# %% [markdown]
# # Per cell type

# %%
common_cell_types = pd.DataFrame({
    "rna": rna_agg_use.obs["common_cell_type"],
    "mCH": mCH_agg_use.obs["common_cell_type"],
    "mCG": mCG_agg_use.obs["common_cell_type"],
    "atac": atac_agg_use.obs["common_cell_type"]
})
common_cell_types["n_ct"] = common_cell_types.apply(lambda x: len(set(x)), axis=1)
common_cell_types.head()

# %%
consistent_pseudocells = common_cell_types.query("n_ct == 1").index
rna_agg_use = rna_agg_use[consistent_pseudocells, :]
mCH_agg_use = mCH_agg_use[consistent_pseudocells, :]
mCG_agg_use = mCG_agg_use[consistent_pseudocells, :]
atac_agg_use = atac_agg_use[consistent_pseudocells, :]

# %%
rna_X = rna_agg_use.X.toarray()
mCH_X = mCH_agg_use.X
mCG_X = mCG_agg_use.X
atac_X = atac_agg_use.X.toarray()

# %%
ct_sizes = rna_agg_use.obs["common_cell_type"].value_counts()
ct_sizes

# %%
min_size = 10
used_cts = ct_sizes.index[ct_sizes > min_size].to_numpy()
used_cts

# %% [markdown]
# ## R square

# %%
rsquare = {}
n_subsample = 10
rs = np.random.RandomState(0)

for ct in used_cts:
    ct_idx_all = np.where(rna_agg_use.obs["common_cell_type"].to_numpy() == ct)[0]
    rsquare_ct_list = []

    for _ in tqdm(range(n_subsample), desc=ct):
        ct_idx = rs.choice(ct_idx_all, min_size, replace=False)
        rsquare_ct = defaultdict(list)

        for i in range(rna_X.shape[1]):
            y_ = rna_X[ct_idx, i]

            X_ = np.expand_dims(mCH_X[ct_idx, i], axis=1)
            lm = LinearRegression().fit(X_, y_)
            rsquare_ct["mCH"].append(lm.score(X_, y_))

            X_ = np.expand_dims(mCG_X[ct_idx, i], axis=1)
            lm = LinearRegression().fit(X_, y_)
            rsquare_ct["mCG"].append(lm.score(X_, y_))

            X_ = np.expand_dims(atac_X[ct_idx, i], axis=1)
            lm = LinearRegression().fit(X_, y_)
            rsquare_ct["ATAC"].append(lm.score(X_, y_))

        rsquare_ct_list.append(rsquare_ct)
    
    rsquare[ct] = {
        k: np.stack([rsquare_ct_list[i][k] for i in range(n_subsample)]).mean(axis=0)
        for k in ("mCH", "mCG", "ATAC")
    }

rsquare = pd.concat({
    ct: pd.DataFrame(d, index=common_genes)
    for ct, d in rsquare.items()
})
rsquare.index.names = ["Cell type", "Gene"]
rsquare.reset_index(inplace=True)
rsquare["Cell type"] = pd.Categorical(
    rsquare["Cell type"],
    categories=["mL2/3", "mL4", "mL5-1", "mDL-2", "mL6-2"],
    ordered=True
)
rsquare_melt = rsquare.melt(
    id_vars=["Cell type", "Gene"],
    var_name="Omics layer", value_name="rsquare"
)

# %%
coefs = {}
pvals = {}
for k in ("mCH", "mCG", "ATAC"):
    regress_data = rsquare_melt.query(f"`Omics layer` == '{k}'")
    X = regress_data["Cell type"].cat.codes
    X = sm.add_constant(X)
    y = regress_data["rsquare"]
    model = sm.OLS(y, X)
    results = model.fit()
    coefs[k] = results.params.loc[0]
    pvals[k] = results.pvalues.loc[0]

# %%
fig, ax = plt.subplots(figsize=(6, 5))
ax = sns.boxplot(
    x="Omics layer", y="rsquare", hue="Cell type", data=rsquare_melt,
    saturation=1.0, width=0.8, showmeans=True,
    meanprops=dict(marker="^", markerfacecolor="white", markeredgecolor="black"),
    boxprops=dict(edgecolor="black"), medianprops=dict(color="black"),
    whiskerprops=dict(color="black"), capprops=dict(color="black"),
    flierprops=dict(marker=".", markerfacecolor="black", markeredgecolor="none", markersize=3),
    palette=MANUAL_COLORS, ax=ax
)
text_kws = dict(
    size=12, ha="center", va="center",
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="lightgrey")
)
ax.text(0.0, 0.95, f"$\\beta$ = {coefs['mCH']:.2e}\n$P$ = {pvals['mCH']:.2e}", **text_kws)
ax.text(1.0, 0.95, f"$\\beta$ = {coefs['mCG']:.2e}\n$P$ = {pvals['mCG']:.2e}", **text_kws)
ax.text(2.0, 0.95, f"$\\beta$ = {coefs['ATAC']:.2e}\n$P$ = {pvals['ATAC']:.2e}", **text_kws)
ax.set_ylabel("Gene expression $R^2$")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.05, 0.5), title="Cell type")
fig.savefig(f"{PATH}/rsquare_ct.pdf")

# %% [markdown]
# # Higher mean expression cutoff

# %%
mean_cutoffs = [2.0, 0.4, 0.4, 2.0]
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
    plot_kws=dict(s=3, edgecolor=None, alpha=0.8, rasterized=True)
).map_offdiag(offdiag_func)
g.savefig(f"{PATH}/corr_cmp_himean.pdf")

# %%
gene_stat = rna_agg_stat.loc[common_genes, :].assign(
    gene_length=np.log10(
        rna.var.loc[common_genes, "chromEnd"] -
        rna.var.loc[common_genes, "chromStart"]
    )
)
gene_stat.head()

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
ax.get_figure().savefig(f"{PATH}/rsquare_himean.pdf")

# %%
rsquare.mean(axis=0)
