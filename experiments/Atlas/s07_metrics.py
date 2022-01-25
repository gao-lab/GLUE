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
import scanpy as sc
import seaborn as sns
import yaml
from matplotlib import rcParams

import scglue
import scglue.metrics

# %%
scglue.plot.set_publication_params()
rcParams["figure.figsize"] = (4, 4)

PATH = "s07_metrics"
os.makedirs(PATH, exist_ok=True)

# %% [markdown]
# # Read data

# %%
rna = anndata.read_h5ad("s04_glue_final/full/rna.h5ad", backed="r")
atac = anndata.read_h5ad("s04_glue_final/full/atac.h5ad", backed="r")

# %%
atac_transferred = anndata.read_h5ad("s06_sankey/atac_transferred.h5ad")
atac.obs["NNLS"] = atac_transferred.obs["NNLS"]
atac.obs["GLUE"] = atac_transferred.obs["GLUE"]
atac.obs["iNMF"] = atac_transferred.obs["iNMF"]

# %% [markdown]
# # Label transfer matchers

# %%
exact_match_set = {(item, item) for item in atac.obs["NNLS"].cat.categories}

# %%
relaxed_match_set = exact_match_set.copy()
for item in atac.obs["NNLS"].cat.categories:
    if "?" in item:
        relaxed_match_set.add((item, item.replace("?", "")))
    if "/" in item:
        for split in item.split("/"):
            relaxed_match_set.add((item, split))
relaxed_match_set = relaxed_match_set.union({
    ("Syncytiotrophoblast and villous cytotrophoblasts?", "Syncytiotrophoblasts and villous cytotrophoblasts"),
    ("Thymocytes", "Lymphoid cells"),
    ("Astrocytes", "Excitatory neurons")
})


# %%
def exact_transfer_accuracy(nnls, pred):
    match = np.array([(i, j) in exact_match_set for i, j in zip(nnls, pred)])
    return np.sum(match) / match.size


def relaxed_transfer_accuracy(nnls, pred):
    match = np.array([(i, j) in relaxed_match_set for i, j in zip(nnls, pred)])
    mask = ~nnls.str.contains("unknown", case=False)
    return np.sum(np.logical_and(match, mask)) / mask.sum()


# %% [markdown]
# # Compute metrics

# %%
atac.obs["Organ"] = atac.obs["tissue"].str.capitalize()
assert set(atac.obs["Organ"]) == set(rna.obs["Organ"])
organs = set(rna.obs["Organ"]).union(["All"])

# %%
combined_cell_type = np.concatenate([
    rna.obs["cell_type"].to_numpy(),
    atac.obs["cell_type"].to_numpy()
])

# %%
combined_domain = np.concatenate([
    rna.obs["domain"].to_numpy(),
    atac.obs["domain"].to_numpy()
])

# %%
combined_organ = np.concatenate([
    rna.obs["Organ"].to_numpy(),
    atac.obs["Organ"].to_numpy()
])

# %%
rna_uni = rna.obsm["X_pca"]
atac_uni = atac.obsm["X_lsi"]
combined_uni = np.concatenate([rna_uni, atac_uni])

# %% [markdown]
# ## GLUE

# %%
rna_latent = rna.obsm["X_glue"]
atac_latent = atac.obsm["X_glue"]
combined_latent = np.concatenate([rna_latent, atac_latent])

# %%
np.random.seed(0)
glue_metrics = []
for organ in organs:
    print(f"Dealing with {organ}...")

    idx = np.arange(atac.shape[0]) if organ == "All" else np.where(atac.obs["Organ"] == organ)[0]
    metrics = {
        "Label transfer accuracy": exact_transfer_accuracy(atac.obs["NNLS"].iloc[idx], atac.obs["GLUE"].iloc[idx]),
        "Label transfer accuracy (relaxed)": relaxed_transfer_accuracy(atac.obs["NNLS"].iloc[idx], atac.obs["GLUE"].iloc[idx]),
    }

    idx = np.arange(combined_organ.size) if organ == "All" else np.where(combined_organ == organ)[0]
    idx = np.random.choice(idx, min(10000, idx.size), replace=False)
    metrics.update({
        "Mean average precision": scglue.metrics.mean_average_precision(combined_latent[idx], combined_cell_type[idx]),
        "Average silhouette width (cell type)": scglue.metrics.avg_silhouette_width(combined_latent[idx], combined_cell_type[idx]),
        "Neighbor conservation": scglue.metrics.neighbor_conservation(combined_latent[idx], combined_uni[idx], combined_domain[idx]),
        "Seurat alignment score": scglue.metrics.seurat_alignment_score(combined_latent[idx], combined_domain[idx], random_state=0),
        "Average silhouette width (batch)":scglue.metrics.avg_silhouette_width_batch(combined_latent[idx], combined_domain[idx], combined_cell_type[idx]),
        "Graph connectivity": scglue.metrics.graph_connectivity(combined_latent[idx], combined_cell_type[idx])
    })
    glue_metrics.append(pd.Series(metrics, name=organ))
glue_metrics = pd.DataFrame(glue_metrics)
glue_metrics.index.name = "Organ"
glue_metrics

# %% [markdown]
# ## iNMF

# %%
rna_latent = pd.read_csv(
    "e02_inmf/rna_latent.csv", header=None, index_col=0
).loc[rna.obs_names].to_numpy()
atac_latent = pd.read_csv(
    "e02_inmf/atac_latent.csv", header=None, index_col=0
).loc[atac.obs_names].to_numpy()

# %%
rna_latent = np.ascontiguousarray(rna_latent, dtype=np.float32)
atac_latent = np.ascontiguousarray(atac_latent, dtype=np.float32)
combined_latent = np.concatenate([rna_latent, atac_latent])

# %%
np.random.seed(0)
inmf_metrics = []
for organ in organs:
    print(f"Dealing with {organ}...")

    idx = np.arange(atac.shape[0]) if organ == "All" else np.where(atac.obs["Organ"] == organ)[0]
    metrics = {
        "Label transfer accuracy": exact_transfer_accuracy(atac.obs["NNLS"].iloc[idx], atac.obs["iNMF"].iloc[idx]),
        "Label transfer accuracy (relaxed)": relaxed_transfer_accuracy(atac.obs["NNLS"].iloc[idx], atac.obs["iNMF"].iloc[idx]),
    }

    idx = np.arange(combined_organ.size) if organ == "All" else np.where(combined_organ == organ)[0]
    idx = np.random.choice(idx, min(10000, idx.size), replace=False)
    metrics.update({
        "Mean average precision": scglue.metrics.mean_average_precision(combined_latent[idx], combined_cell_type[idx]),
        "Average silhouette width (cell type)": scglue.metrics.avg_silhouette_width(combined_latent[idx], combined_cell_type[idx]),
        "Neighbor conservation": scglue.metrics.neighbor_conservation(combined_latent[idx], combined_uni[idx], combined_domain[idx]),
        "Seurat alignment score": scglue.metrics.seurat_alignment_score(combined_latent[idx], combined_domain[idx], random_state=0),
        "Average silhouette width (batch)":scglue.metrics.avg_silhouette_width_batch(combined_latent[idx], combined_domain[idx], combined_cell_type[idx]),
        "Graph connectivity": scglue.metrics.graph_connectivity(combined_latent[idx], combined_cell_type[idx])
    })
    inmf_metrics.append(pd.Series(metrics, name=organ))
inmf_metrics = pd.DataFrame(inmf_metrics)
inmf_metrics.index.name = "Organ"
inmf_metrics

# %% [markdown]
# # Plotting

# %%
df = pd.concat([
    glue_metrics.assign(Method="GLUE"),
    inmf_metrics.assign(Method="Online iNMF")
]).reset_index().melt(id_vars=["Organ", "Method"], var_name="Metric", value_name="Value")
df["Organ"] = pd.Categorical(df["Organ"], categories=["All"] + sorted(set(df["Organ"]).difference(["All"])))
df

# %%
with open("../../evaluation/config/display.yaml", "r") as f:
    palette = yaml.load(f, Loader=yaml.Loader)["palette"]

# %%
g = sns.FacetGrid(
    df, col="Organ", col_wrap=4
).map(
    sns.barplot, "Value", "Metric", "Method",
    order=[
        "Label transfer accuracy",
        "Label transfer accuracy (relaxed)",
        "Mean average precision",
        "Average silhouette width (cell type)",
        "Neighbor conservation",
        "Seurat alignment score",
        "Average silhouette width (batch)",
        "Graph connectivity"
    ],
    hue_order=["Online iNMF", "GLUE"],
    palette=palette
)
for ax in g.axes:
    yticklabels = ax.get_yticklabels()
    if not yticklabels:
        continue
    for i in (0, 1):
        yticklabels[i].set_color("darkred")
    for i in (2, 3, 4):
        yticklabels[i].set_color("darkgreen")
    for i in (5, 6, 7):
        yticklabels[i].set_color("darkblue")
g.add_legend(title="Method")
g.savefig(f"{PATH}/metrics.pdf")
