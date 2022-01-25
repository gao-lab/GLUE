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
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import scipy.stats
import seaborn as sns
from matplotlib import rcParams
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, cdist

import scglue

# %%
scglue.plot.set_publication_params()
rcParams["figure.figsize"] = (7, 7)

PATH = "s03_unsupervised_balancing"
os.makedirs(PATH, exist_ok=True)

# %% [markdown]
# # Read data

# %%
rna = anndata.read_h5ad("s01_preprocessing/rna.h5ad")
atac = anndata.read_h5ad("s01_preprocessing/atac.h5ad")

# %%
rna_agg = anndata.read_h5ad("s02_glue_pretrain/rna_agg.h5ad")
atac_agg = anndata.read_h5ad("s02_glue_pretrain/atac_agg.h5ad")

# %% [markdown]
# # Clustering

# %% [markdown]
# We need another level of clustering because:
# * The metacell level clustering must retain enough sample size and not distort the distribution of aggregated profiles too much. So it inherently requires a lot of clusters with small cluster size.
# * For cluster matching, larger clusters make more sense.

# %%
sc.pp.neighbors(rna_agg, n_pcs=rna_agg.obsm["X_glue"].shape[1], use_rep="X_glue", metric="cosine")
sc.tl.leiden(rna_agg, resolution=2, random_state=0)  # Resolution higher than the default in `scglue.data.estimate_balancing_weight`
rna_agg.obs["leiden"].cat.rename_categories(lambda x: f"rna-leiden-{x}", inplace=True)

# %%
sc.pp.neighbors(atac_agg, n_pcs=atac_agg.obsm["X_glue"].shape[1], use_rep="X_glue", metric="cosine")
sc.tl.leiden(atac_agg, resolution=2, random_state=0)  # Resolution higher than the default in `scglue.data.estimate_balancing_weight`
atac_agg.obs["leiden"].cat.rename_categories(lambda x: f"atac-leiden-{x}", inplace=True)

# %% [markdown]
# # Visualization

# %%
combined_agg = anndata.AnnData(
    obs=pd.concat([rna_agg.obs, atac_agg.obs], join="inner"),
    obsm={
        "X_glue": np.concatenate([rna_agg.obsm["X_glue"], atac_agg.obsm["X_glue"]]),
        "X_glue_umap": np.concatenate([rna_agg.obsm["X_glue_umap"], atac_agg.obsm["X_glue_umap"]])
    }
)

# %%
fig = sc.pl.embedding(
    combined_agg, "X_glue_umap", color="leiden", return_fig=True,
    legend_loc="on data", legend_fontsize=4, legend_fontoutline=0.5
)
fig.axes[0].set_xlabel("UMAP1")
fig.axes[0].set_ylabel("UMAP2")
fig.savefig(f"{PATH}/combined_leiden.pdf")

# %% [markdown]
# # Cross-domain heatmap

# %%
rna_agg.obs["n_metacells"] = 1
rna_leiden = scglue.data.aggregate_obs(
    rna_agg, by="leiden", X_agg=None,
    obs_agg={
        "domain": "majority", "cell_type": "majority",
        "n_metacells": "sum", "n_cells": "sum"
    },
    obsm_agg={"X_glue": "mean", "X_glue_umap": "mean"}
)

# %%
atac_agg.obs["n_metacells"] = 1
atac_leiden = scglue.data.aggregate_obs(
    atac_agg, by="leiden", X_agg=None,
    obs_agg={
        "domain": "majority", "cell_type": "majority",
        "n_metacells": "sum", "n_cells": "sum"
    },
    obsm_agg={"X_glue": "mean", "X_glue_umap": "mean"}
)

# %%
u1 = rna_leiden.obsm["X_glue"]
u2 = atac_leiden.obsm["X_glue"]
nm1 = rna_leiden.obs["n_metacells"].to_numpy()
nm2 = atac_leiden.obs["n_metacells"].to_numpy()
nc1 = rna_leiden.obs["n_cells"].to_numpy()
nc2 = atac_leiden.obs["n_cells"].to_numpy()

link1 = linkage(pdist(u1, metric="cosine"), method="average")
link2 = linkage(pdist(u2, metric="cosine"), method="average")

cosine = 1 - cdist(u2, u1, metric="cosine")
cosine[cosine < 0.5] = np.nan  # Only keep highly-correlated pairs
cosine = cosine ** 4  # Increase contrast

# %% tags=[]
heatmap_df = pd.DataFrame(
    cosine,
    index=atac_leiden.obs_names,
    columns=rna_leiden.obs_names
)
g = sns.clustermap(
    heatmap_df, row_linkage=link2, col_linkage=link1,
    cmap="bwr", center=0, xticklabels=1, yticklabels=1,
    figsize=(14, 9)
)
g.fig.axes[2].xaxis.set_tick_params(labelsize=10)
g.fig.axes[2].yaxis.set_tick_params(labelsize=10)
g.fig.savefig(f"{PATH}/leiden_heatmap.pdf")

# %% [markdown]
# # Compute unsupervised balancing

# %% [markdown]
# **NOTE:** We'd also want to squeeze those non-shared cell types to 0. In which case MNN might be able to help.

# %%
cosine[np.isnan(cosine)] = 0

# %% tags=[]
nm1_balancing = pd.Series(
    cosine.sum(axis=0) / nm1,
    index=rna_leiden.obs_names
)
nc1_balancing = pd.Series(
    cosine.sum(axis=0) / nc1,
    index=rna_leiden.obs_names
)

# %%
rna_agg.obs["nm_balancing"] = nm1_balancing.loc[rna_agg.obs["leiden"]].to_numpy()
rna_agg.obs["nc_balancing"] = nc1_balancing.loc[rna_agg.obs["leiden"]].to_numpy()

# %% tags=[]
nm2_balancing = pd.Series(
    cosine.sum(axis=1) / nm2,
    index=atac_leiden.obs_names
)
nc2_balancing = pd.Series(
    cosine.sum(axis=1) / nc2,
    index=atac_leiden.obs_names
)

# %%
atac_agg.obs["nm_balancing"] = nm2_balancing.loc[atac_agg.obs["leiden"]].to_numpy()
atac_agg.obs["nc_balancing"] = nc2_balancing.loc[atac_agg.obs["leiden"]].to_numpy()

# %% [markdown]
# # Compare balanced with original

# %%
rna_agg.obs["nc_balancing_"] = rna_agg.obs["nc_balancing"] * rna_agg.obs["n_cells"]
rna_df = rna_agg.obs.groupby("cell_type").sum()
rna_df = rna_df / rna_df.sum(axis=0)
rna_df

# %%
atac_agg.obs["nc_balancing_"] = atac_agg.obs["nc_balancing"] * atac_agg.obs["n_cells"]
atac_df = atac_agg.obs.groupby("cell_type").sum()
atac_df = atac_df / atac_df.sum(axis=0)
atac_df

# %% [markdown]
# ## Fraction of metacells

# %%
df = pd.concat([
    pd.DataFrame({"scRNA-seq": rna_df["n_metacells"], "scATAC-seq": atac_df["n_metacells"], "Fraction": "Unbalanced"}),
    pd.DataFrame({"scRNA-seq": rna_df["nm_balancing"], "scATAC-seq": atac_df["nm_balancing"], "Fraction": "Balanced"})
]).fillna(0)

# %%
rcParams["figure.figsize"] = (4, 4)
ax = sns.scatterplot(x="scRNA-seq", y="scATAC-seq", hue="Fraction", data=df)
ax.axline((0, 0), (1, 1), c="grey", zorder=0, linestyle="--")
ax.set_xlim(-0.01, 0.35)
ax.set_ylim(-0.01, 0.35)
ax.get_figure().savefig(f"{PATH}/leiden_metacell_fraction_cmp.pdf") 

# %%
ax.set_xlim(-0.005, 0.10)
ax.set_ylim(-0.005, 0.10)
ax.get_figure().savefig(f"{PATH}/leiden_metacell_fraction_cmp_zoomin.pdf")
ax.get_figure()

# %%
scipy.stats.pearsonr(
    df.query("Fraction == 'Balanced'")["scRNA-seq"],
    df.query("Fraction == 'Balanced'")["scATAC-seq"]
)

# %% [markdown]
# ## Fraction of cells

# %%
df = pd.concat([
    pd.DataFrame({"scRNA-seq": rna_df["n_cells"], "scATAC-seq": atac_df["n_cells"], "Fraction": "Unbalanced"}),
    pd.DataFrame({"scRNA-seq": rna_df["nc_balancing_"], "scATAC-seq": atac_df["nc_balancing_"], "Fraction": "Balanced"})
]).fillna(0)

# %%
ax = sns.scatterplot(x="scRNA-seq", y="scATAC-seq", hue="Fraction", data=df)
ax.axline((0, 0), (1, 1), c="grey", zorder=0, linestyle="--")
ax.set_xlim(-0.01, 0.35)
ax.set_ylim(-0.01, 0.35)
ax.get_figure().savefig(f"{PATH}/leiden_cell_fraction_cmp.pdf")

# %%
ax.set_xlim(-0.005, 0.10)
ax.set_ylim(-0.005, 0.10)
ax.get_figure().savefig(f"{PATH}/leiden_cell_fraction_cmp_zoomin.pdf")
ax.get_figure()

# %%
scipy.stats.pearsonr(
    df.query("Fraction == 'Balanced'")["scRNA-seq"],
    df.query("Fraction == 'Balanced'")["scATAC-seq"]
)

# %% [markdown]
# # Propagate to unaggregated data

# %%
rna.obs["nc_balancing"] = rna_agg.obs["nc_balancing"].loc[
    rna.obs["metacell"].to_numpy()
].to_numpy()

# %%
atac.obs["nc_balancing"] = atac_agg.obs["nc_balancing"].loc[
    atac.obs["metacell"].to_numpy()
].to_numpy()

# %% [markdown]
# # Save results

# %%
rna_agg.write(f"{PATH}/rna_agg_balanced.h5ad", compression="gzip")
atac_agg.write(f"{PATH}/atac_agg_balanced.h5ad", compression="gzip")

# %%
rna.write(f"{PATH}/rna_balanced.h5ad", compression="gzip")
atac.write(f"{PATH}/atac_balanced.h5ad", compression="gzip")
