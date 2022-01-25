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
import statsmodels
import yaml
import matplotlib_venn
from matplotlib import rcParams

import scglue

# %%
scglue.plot.set_publication_params()
rcParams["figure.figsize"] = (4, 4)

PATH = "s03_markers"
os.makedirs(PATH, exist_ok=True)

# %%
# %load_ext rpy2.ipython

# %% language="R"
# suppressPackageStartupMessages({
#     source(".Rprofile")
#     library(dplyr)
#     library(SuperExactTest)
# })

# %% [markdown]
# # Read data

# %%
rna = anndata.read_h5ad("s02_glue/rna.h5ad")
met = anndata.read_h5ad("s02_glue/met2rna.h5ad")
atac = anndata.read_h5ad("s02_glue/atac2rna.h5ad")
combined = anndata.read_h5ad("s02_glue/combined.h5ad")

# %%
with open("manual_colors.yaml", "r") as f:
    MANUAL_COLORS = yaml.load(f, Loader=yaml.FullLoader)

# %% [markdown]
# # Transfer labels

# %% tags=[]
met.obs["common_cell_type"] = met.obs["cell_type"].replace({
    "mNdnf-1": "mNdnf",
    "mNdnf-2": "mNdnf",
    "mSst-1": "mSst",
    "mSst-2": "mSst"
})
met.obs["common_cell_type"] = pd.Categorical(met.obs["common_cell_type"], categories=[
    "mL2/3", "mL4", "mL5-1", "mDL-1", "mDL-2", "mL5-2",
    "mL6-1", "mL6-2", "mDL-3", "mIn-1", "mVip",
    "mNdnf", "mPv", "mSst"
])
met.uns["common_cell_type_colors"] = [MANUAL_COLORS[item] for item in met.obs["common_cell_type"].cat.categories]

# %%
fig = sc.pl.embedding(
    met, "X_glue_umap", color="common_cell_type",
    title="snmC-seq common cell type", return_fig=True
)
fig.axes[0].set_xlabel("UMAP1")
fig.axes[0].set_ylabel("UMAP2")
fig.savefig(f"{PATH}/met_cmct.pdf")

# %%
link_cutoff = ceil(met.shape[0] * 0.001)
link_color_map = defaultdict(lambda: "#CCCCCC")
link_color_map.update({
    ("mNdnf-1", "mNdnf"): MANUAL_COLORS["CGE"],
    ("mNdnf-2", "mNdnf"): MANUAL_COLORS["CGE"],
    ("mVip", "mVip"): MANUAL_COLORS["CGE"],
    ("mSst-1", "mSst"): MANUAL_COLORS["MGE"],
    ("mSst-2", "mSst"): MANUAL_COLORS["MGE"],
    ("mPv", "mPv"): MANUAL_COLORS["MGE"],
    ("mDL-3", "mDL-3"): MANUAL_COLORS["Claustrum"]
})
fig = scglue.plot.sankey(
    met.obs["cell_type"], met.obs["common_cell_type"],
    title="snmC-seq cell type mapping",
    left_color=lambda x: MANUAL_COLORS[x],
    right_color=lambda x: MANUAL_COLORS[x],
    link_color=lambda x: "rgba(0.9,0.9,0.9,0.2)" if x["value"] <= link_cutoff \
        else link_color_map[(x["left"], x["right"])]
)
pio.write_image(fig, f"{PATH}/met_sankey.png", scale=10)

# %%
scglue.data.transfer_labels(met, rna, "common_cell_type", use_rep="X_glue", metric="cosine")
rna.uns["common_cell_type_colors"] = met.uns["common_cell_type_colors"]

# %%
fig = sc.pl.embedding(
    rna, "X_glue_umap", color="common_cell_type",
    title="scRNA-seq common cell type", return_fig=True
)
fig.axes[0].set_xlabel("UMAP1")
fig.axes[0].set_ylabel("UMAP2")
fig.savefig(f"{PATH}/rna_cmct.pdf")

# %%
link_cutoff = ceil(rna.shape[0] * 0.001)
link_color_map = defaultdict(lambda: "#CCCCCC")
link_color_map.update({
    ("CGE", "mNdnf"): MANUAL_COLORS["CGE"],
    ("CGE", "mVip"): MANUAL_COLORS["CGE"],
    ("MGE", "mSst"): MANUAL_COLORS["MGE"],
    ("MGE", "mPv"): MANUAL_COLORS["MGE"],
    ("Claustrum", "mDL-3"): MANUAL_COLORS["Claustrum"]
})
fig = scglue.plot.sankey(
    rna.obs["cell_type"], rna.obs["common_cell_type"],
    title="scRNA-seq cell type mapping",
    left_color=lambda x: MANUAL_COLORS[x],
    right_color=lambda x: MANUAL_COLORS[x],
    link_color=lambda x: "rgba(0.9,0.9,0.9,0.2)" if x["value"] <= link_cutoff \
        else link_color_map[(x["left"], x["right"])]
)
pio.write_image(fig, f"{PATH}/rna_sankey.png", scale=10)

# %%
scglue.data.transfer_labels(met, atac, "common_cell_type", use_rep="X_glue", metric="cosine")
atac.uns["common_cell_type_colors"] = met.uns["common_cell_type_colors"]

# %%
fig = sc.pl.embedding(
    atac, "X_glue_umap", color="common_cell_type",
    title="scATAC-seq common cell type", return_fig=True
)
fig.axes[0].set_xlabel("UMAP1")
fig.axes[0].set_ylabel("UMAP2")
fig.savefig(f"{PATH}/atac_cmct.pdf")

# %%
link_cutoff = ceil(atac.shape[0] * 0.001)
link_color_map = defaultdict(lambda: "#CCCCCC")
link_color_map.update({
    ("Vip", "mNdnf"): MANUAL_COLORS["CGE"],
    ("Vip", "mVip"): MANUAL_COLORS["CGE"],
    ("Sst", "mSst"): MANUAL_COLORS["MGE"],
    ("Pvalb", "mPv"): MANUAL_COLORS["MGE"],
    ("L6 IT", "mDL-3"): MANUAL_COLORS["Claustrum"]
})
fig = scglue.plot.sankey(
    atac.obs["cell_type"], atac.obs["common_cell_type"],
    title="scATAC-seq cell type mapping",
    left_color=lambda x: MANUAL_COLORS[x],
    right_color=lambda x: MANUAL_COLORS[x],
    link_color=lambda x: "rgba(0.9,0.9,0.9,0.2)" if x["value"] <= link_cutoff \
        else link_color_map[(x["left"], x["right"])]
)
pio.write_image(fig, f"{PATH}/atac_sankey.png", scale=10)

# %% [markdown]
# # Marker identification

# %% [markdown]
# ## Normalization and filtering

# %%
sc.pp.normalize_total(rna)
sc.pp.log1p(rna)

# sc.pp.normalize_total(met)
sc.pp.log1p(met)

sc.pp.normalize_total(atac)
sc.pp.log1p(atac)

# %%
rna = rna[:, rna.X.sum(axis=0).A1 != 0]
met = met[:, met.X.sum(axis=0) != 0]
atac = atac[:, atac.X.sum(axis=0).A1 != 0]

# %%
common_genes = list(set(rna.var_names).intersection(met.var_names).intersection(atac.var_names))
len(common_genes)

# %%
rna = rna[:, common_genes].copy()
met = met[:, common_genes].copy()
atac = atac[:, common_genes].copy()

# %%
rna.write(f"{PATH}/rna_filtered.h5ad", compression="gzip")
met.write(f"{PATH}/met_filtered.h5ad", compression="gzip")
atac.write(f"{PATH}/atac_filtered.h5ad", compression="gzip")

# %% [markdown]
# ## Differential expression

# %%
sc.tl.rank_genes_groups(rna, "common_cell_type", method="wilcoxon")

# %%
rna_marker_df = scglue.data.extract_rank_genes_groups(
    rna, rna.obs["common_cell_type"].cat.categories,
    filter_by="pvals_adj < 0.05 & logfoldchanges > 0",
    sort_by="logfoldchanges", ascending=False
).assign(domain="scRNA-seq")
rna_marker_df.head()

# %%
sc.tl.rank_genes_groups(met, "common_cell_type", method="wilcoxon")

# %%
met_marker_df = scglue.data.extract_rank_genes_groups(
    met, met.obs["common_cell_type"].cat.categories,
    filter_by="pvals_adj < 0.05 & logfoldchanges < 0",
    sort_by="logfoldchanges", ascending=True
).assign(domain="snmC-seq")
met_marker_df.head()

# %%
sc.tl.rank_genes_groups(atac, "common_cell_type", method="wilcoxon")

# %%
atac_marker_df = scglue.data.extract_rank_genes_groups(
    atac, atac.obs["common_cell_type"].cat.categories,
    filter_by="pvals_adj < 0.05 & logfoldchanges > 0",
    sort_by="logfoldchanges", ascending=False
).assign(domain="scATAC-seq")
atac_marker_df.head()

# %% [markdown]
# ## Significance of intersection

# %%
combined_marker_df = pd.concat([rna_marker_df, met_marker_df, atac_marker_df])
combined_marker_df["domain"] = pd.Categorical(combined_marker_df["domain"], categories=["scRNA-seq", "snmC-seq", "scATAC-seq"])
combined_marker_df["group_domain"] = pd.Categorical(
    combined_marker_df["group"].astype(str) + "_" + combined_marker_df["domain"].astype(str),
    categories=[
        f"{g}_{d}"
        for g in combined_marker_df["group"].cat.categories
        for d in combined_marker_df["domain"].cat.categories
    ]
)

# %% magic_args="-i combined_marker_df -i common_genes -o res" language="R"
# combined_marker_sets <- split(combined_marker_df$names, combined_marker_df$group_domain)
# empty_sets <- setdiff(levels(combined_marker_df$group_domain), names(combined_marker_sets))
# for (empty_set in empty_sets) {
#     combined_marker_sets[[empty_set]] <- character()
# }  # Add empty marker sets for completeness
# res <- supertest(combined_marker_sets, n=length(common_genes), degree=3)  # Test all 3-set intersections
# res <- summary(res)$Table
# rownames(res) <- NULL

# %%
intersections = pd.DataFrame.from_records(
    res["Intersections"].str.split(" & ").map(lambda x: {
        item.split("_")[1]: item.split("_")[0]
        for item in x
    }), index=res.index
)
res = res.join(intersections).dropna(
    subset=intersections.columns
)  # Dropped are those involving intersection within the same domains
res["FDR"] = statsmodels.stats.multitest.fdrcorrection(res["P.value"])[1]
res["-log10 P.value"] = -np.log10(res["P.value"])
res["-log10 FDR"] = -np.log10(res["FDR"])
res = res.sort_values("FDR")
res.to_csv(f"{PATH}/marker_intersection_test.csv", index=False)

# %%
res_samect = res.query("`scRNA-seq` == `snmC-seq` == `scATAC-seq`")
res_samect

# %% tags=[]
os.makedirs(f"{PATH}/venn", exist_ok=True)
for g, fdr in zip(res_samect["scRNA-seq"], res_samect["FDR"]):
    current_df = combined_marker_df.query(f"group == '{g}'")
    fig, ax = plt.subplots(figsize=(4, 4))
    sets, set_labels, set_colors = [], [], []
    for d, c in zip(combined.obs["domain"].cat.categories, combined.uns["domain_colors"]):
        s = set(current_df.query(f"domain == '{d}'")["names"])
        if len(s):
            sets.append(s)
            set_labels.append(d)
            set_colors.append(c)
    if len(sets) <= 1:
        continue
    venn = getattr(matplotlib_venn, f"venn{len(sets)}")(
        sets, set_labels=set_labels, set_colors=set_colors, alpha=0.75, ax=ax
    )
    for item in venn.set_labels:
        item.set_fontsize(15)
    ax.set_title(f"{g} (FDR = {fdr:.3e})")
    fig.savefig(f"{PATH}/venn/{g.replace('/', '&')}.pdf")
    plt.close()  # Too long, do not show in here

# %%
fig, ax = plt.subplots(figsize=(6, 4))
ax = sns.barplot(
    x="-log10 FDR", y="scRNA-seq", data=res_samect, 
    palette=MANUAL_COLORS, saturation=1.0, ax=ax
)
ax.set_ylabel("Cell type")
ax.axvline(x=2, c="black", ls="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.savefig(f"{PATH}/marker_intersection_logfdr.pdf")

# %% [markdown]
# ## Select consensus markers

# %%
consensus_marker_sets = {
    g: set(elements.split(", "))
    for elements, g in zip(res_samect["Elements"], res_samect["scRNA-seq"])
}
mask = [
    name in consensus_marker_sets[g]
    for name, g in zip(combined_marker_df["names"], combined_marker_df["group"])
]
consensus_marker_df = combined_marker_df.loc[mask].copy()
consensus_marker_df.head()

# %%
consensus_marker_df["scores"] = consensus_marker_df["scores"].abs()
consensus_marker_df["logfoldchanges"] = consensus_marker_df["logfoldchanges"].abs()
consensus_marker_df = consensus_marker_df.groupby(["names", "group"]).mean().reset_index().dropna()
consensus_marker_df = consensus_marker_df.sort_values(["group", "logfoldchanges"], ascending=[True, False])
consensus_marker_df.head()

# %% [markdown]
# # Visualization

# %% [markdown]
# ## Heatmap

# %%
selected_marker_df = consensus_marker_df.groupby("group").head(n=5).copy()
selected_marker_df["group"].cat.remove_unused_categories(inplace=True)

# %%
fig = sc.pl.matrixplot(
    rna[rna.obs["common_cell_type"] != "mIn-1"],
    selected_marker_df["names"], "common_cell_type",
    dendrogram=False, standard_scale="var", cmap="viridis",
    var_group_positions=[(
        selected_marker_df["group"].searchsorted(g),
        selected_marker_df["group"].searchsorted(g, side="right") - 1
    ) for g in selected_marker_df["group"].cat.categories],
    var_group_labels=selected_marker_df["group"].cat.categories,
    var_group_rotation=45,
    return_fig=True
)
fig.legend(title="Mean expression\nin group")
fig.savefig(f"{PATH}/rna_cmct_consensus_heatmap.pdf")

# %%
fig = sc.pl.matrixplot(
    met[met.obs["common_cell_type"] != "mIn-1"],
    selected_marker_df["names"], "common_cell_type",
    dendrogram=False, standard_scale="var", cmap="viridis_r",
    var_group_positions=[(
        selected_marker_df["group"].searchsorted(g),
        selected_marker_df["group"].searchsorted(g, side="right") - 1
    ) for g in selected_marker_df["group"].cat.categories],
    var_group_labels=selected_marker_df["group"].cat.categories,
    var_group_rotation=45,
    return_fig=True
)
fig.legend(title="Mean methylation\nin group")
fig.savefig(f"{PATH}/met_cmct_consensus_heatmap.pdf")

# %%
fig = sc.pl.matrixplot(
    atac[atac.obs["common_cell_type"] != "mIn-1"],
    selected_marker_df["names"], "common_cell_type",
    dendrogram=False, standard_scale="var", cmap="viridis",
    var_group_positions=[(
        selected_marker_df["group"].searchsorted(g),
        selected_marker_df["group"].searchsorted(g, side="right") - 1
    ) for g in selected_marker_df["group"].cat.categories],
    var_group_labels=selected_marker_df["group"].cat.categories,
    var_group_rotation=45,
    return_fig=True
)
fig.legend(title="Mean accessibility\nin group")
fig.savefig(f"{PATH}/atac_cmct_consensus_heatmap.pdf")

# %% [markdown]
# ## Feature plot

# %%
selected_marker_df = consensus_marker_df.groupby("group").head(n=1).copy()
selected_marker_df["group"].cat.remove_unused_categories(inplace=True)
selected_marker_df

# %%
rcParams["figure.figsize"] = (3, 3)

# %%
fig = sc.pl.embedding(rna, "X_glue_umap", color=selected_marker_df["names"], return_fig=True)
for i, ax in enumerate(fig.axes):
    if i % 2 == 0:
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
fig.savefig(f"{PATH}/rna_cmct_consensus_featureplot.pdf")

# %%
fig = sc.pl.embedding(met, "X_glue_umap", color=selected_marker_df["names"], return_fig=True)
for i, ax in enumerate(fig.axes):
    if i % 2 == 0:
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
fig.savefig(f"{PATH}/met_cmct_consensus_featureplot.pdf")

# %%
fig = sc.pl.embedding(atac, "X_glue_umap", color=selected_marker_df["names"], return_fig=True)
for i, ax in enumerate(fig.axes):
    if i % 2 == 0:
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
fig.savefig(f"{PATH}/atac_cmct_consensus_featureplot.pdf")
