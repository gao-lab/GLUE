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
import networkx as nx
import scanpy as sc
from matplotlib import rcParams
from networkx.algorithms.bipartite import biadjacency_matrix

import scglue

# %%
scglue.plot.set_publication_params()
rcParams["figure.figsize"] = (3, 3)

PATH = "s05_verify"
os.makedirs(PATH, exist_ok=True)

# %% [markdown]
# # Read data

# %%
rna = anndata.read_h5ad("../../data/dataset/Cao-2020.h5ad")
atac = anndata.read_h5ad("../../data/dataset/Domcke-2020.h5ad")

# %%
rna_aligned = anndata.read_h5ad("s04_glue_final/full/rna.h5ad", backed="r")
atac_aligned = anndata.read_h5ad("s04_glue_final/full/atac.h5ad", backed="r")
rna.obsm["X_glue_umap"] = rna_aligned.obsm["X_glue_umap"]
atac.obsm["X_glue_umap"] = atac_aligned.obsm["X_glue_umap"]
del rna_aligned, atac_aligned

# %%
prior = nx.read_graphml("s01_preprocessing/full.graphml.gz")
biadj = biadjacency_matrix(prior, atac.var_names, rna.var_names)

# %%
atac = anndata.AnnData(
    X=atac.X @ biadj, obs=atac.obs, var=rna.var,
    obsm={"X_glue_umap": atac.obsm["X_glue_umap"]}
)

# %% [markdown]
# # Normalization

# %%
sc.pp.normalize_total(rna)
sc.pp.normalize_total(atac)

# %%
sc.pp.log1p(rna)
sc.pp.log1p(atac)

# %%
fig = sc.pl.embedding(rna, "X_glue_umap", color="Organ", return_fig=True)
fig.savefig(f"{PATH}/rna_organ.pdf")

# %%
fig = sc.pl.embedding(atac, "X_glue_umap", color="tissue", return_fig=True)
fig.savefig(f"{PATH}/atac_tissue.pdf")

# %% [markdown]
# # Cerebrum

# %%
rna_cerebrum = rna[rna.obs["Organ"] == "Cerebrum", :]
atac_cerebrum = atac[atac.obs["tissue"] == "cerebrum", :]

# %%
os.makedirs(f"{PATH}/cerebrum", exist_ok=True)

# %% [markdown]
# ## Neural progenitor markers

# %% [markdown]
# Likely radial glial cells

# %%
MARKERS = [
    "PAX6", "SOX2", "HES1", "VIM",
    "HOPX", "LIFR", "PDGFD", "GLI3"
]

# %%
fig = sc.pl.embedding(
    rna_cerebrum, "X_glue_umap", gene_symbols="gene_name", color=MARKERS,
    vmin=0, vmax="p99.9", return_fig=True
)
for i, ax in enumerate(fig.axes):
    if i % 2 == 0:
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
fig.savefig(f"{PATH}/cerebrum/rna_neural_progenitor.pdf")

# %%
fig = sc.pl.embedding(
    atac_cerebrum, "X_glue_umap", gene_symbols="gene_name", color=MARKERS,
    vmin=0, vmax="p99.9", return_fig=True
)
for i, ax in enumerate(fig.axes):
    if i % 2 == 0:
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
fig.savefig(f"{PATH}/cerebrum/atac_neural_progenitor.pdf")

# %% [markdown]
# ## Excitatory neurons markers

# %%
MARKERS = [
    "EOMES", "SOX5", "RUNX1T1", "EPHA3",
    "FAM19A1", "LMO3", "SATB2", "UNC5D"
]

# %%
fig = sc.pl.embedding(
    rna_cerebrum, "X_glue_umap", gene_symbols="gene_name", color=MARKERS,
    vmin=0, vmax="p99.9", return_fig=True
)
for i, ax in enumerate(fig.axes):
    if i % 2 == 0:
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
fig.savefig(f"{PATH}/cerebrum/rna_excitatory_neuron.pdf")

# %%
fig = sc.pl.embedding(
    atac_cerebrum, "X_glue_umap", gene_symbols="gene_name", color=MARKERS,
    vmin=0, vmax="p99.9", return_fig=True
)
for i, ax in enumerate(fig.axes):
    if i % 2 == 0:
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
fig.savefig(f"{PATH}/cerebrum/atac_excitatory_neuron.pdf")

# %% [markdown]
# ## Astrocyte markers

# %%
MARKERS = [
    "SLC1A3", "AQP4", "PAX3", "GFAP",
    "RFX4", "PTN", "MMD2", "ATP1A2"
]

# %%
fig = sc.pl.embedding(
    rna_cerebrum, "X_glue_umap", gene_symbols="gene_name", color=MARKERS,
    vmin=0, vmax="p99.9", return_fig=True
)
for i, ax in enumerate(fig.axes):
    if i % 2 == 0:
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
fig.savefig(f"{PATH}/cerebrum/rna_astrocyte.pdf")

# %%
fig = sc.pl.embedding(
    atac_cerebrum, "X_glue_umap", gene_symbols="gene_name", color=MARKERS,
    vmin=0, vmax="p99.9", return_fig=True
)
for i, ax in enumerate(fig.axes):
    if i % 2 == 0:
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
fig.savefig(f"{PATH}/cerebrum/atac_astrocyte.pdf")

# %% [markdown]
# ## Oligodendrocyte markers

# %%
MARKERS = [
    "SOX10", "LHFPL3", "PCDH15", "SCN1A",
    "ZNF365", "PDGFRA", "BRINP3", "PDE4B"
]

# %%
fig = sc.pl.embedding(
    rna_cerebrum, "X_glue_umap", gene_symbols="gene_name", color=MARKERS,
    vmin=0, vmax="p99.9", return_fig=True
)
for i, ax in enumerate(fig.axes):
    if i % 2 == 0:
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
fig.savefig(f"{PATH}/cerebrum/rna_oligodendrocyte.pdf")

# %%
fig = sc.pl.embedding(
    atac_cerebrum, "X_glue_umap", gene_symbols="gene_name", color=MARKERS,
    vmin=0, vmax="p99.9", return_fig=True
)
for i, ax in enumerate(fig.axes):
    if i % 2 == 0:
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
fig.savefig(f"{PATH}/cerebrum/atac_oligodendrocyte.pdf")
