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
import faiss
import numpy as np
import pandas as pd
import plotly.io as pio
import scanpy as sc
from matplotlib import rcParams

import scglue

# %%
scglue.plot.set_publication_params()
rcParams["figure.figsize"] = (8, 8)

PATH = "s06_sankey"
os.makedirs(PATH, exist_ok=True)

# %% [markdown]
# # Read data

# %%
rna = anndata.read_h5ad("s04_glue_final/full/rna.h5ad", backed="r")
atac = anndata.read_h5ad("s04_glue_final/full/atac.h5ad", backed="r")

# %%
atac.obs["NNLS"] = atac.obs["cell_type"]

# %% [markdown]
# # Transfer labels

# %% [markdown]
# ## GLUE

# %%
rna_latent = rna.obsm["X_glue"]
atac_latent = atac.obsm["X_glue"]

# %%
rna_latent = rna_latent / np.linalg.norm(rna_latent, axis=1, keepdims=True)
atac_latent = atac_latent / np.linalg.norm(atac_latent, axis=1, keepdims=True)

# %%
np.random.seed(0)
quantizer = faiss.IndexFlatIP(rna_latent.shape[1])
n_voronoi = round(np.sqrt(rna_latent.shape[0]))
index = faiss.IndexIVFFlat(quantizer, rna_latent.shape[1], n_voronoi, faiss.METRIC_INNER_PRODUCT)
index.train(rna_latent[np.random.choice(rna_latent.shape[0], 50 * n_voronoi, replace=False)])
index.add(rna_latent)

# %%
nnd, nni = index.search(atac_latent, 50)

# %%
hits = rna.obs["cell_type"].to_numpy()[nni]

# %%
pred = pd.crosstab(
    np.repeat(atac.obs_names, nni.shape[1]), hits.ravel()
).idxmax(axis=1).loc[atac.obs_names]
pred = pd.Categorical(pred, categories=rna.obs["cell_type"].cat.categories)
atac.obs["GLUE"] = pred

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

# %%
rna_latent = rna_latent / np.linalg.norm(rna_latent, axis=1, keepdims=True)
atac_latent = atac_latent / np.linalg.norm(atac_latent, axis=1, keepdims=True)

# %%
np.random.seed(0)
quantizer = faiss.IndexFlatIP(rna_latent.shape[1])
n_voronoi = round(np.sqrt(rna_latent.shape[0]))
index = faiss.IndexIVFFlat(quantizer, rna_latent.shape[1], n_voronoi, faiss.METRIC_INNER_PRODUCT)
index.train(rna_latent[np.random.choice(rna_latent.shape[0], 50 * n_voronoi, replace=False)])
index.add(rna_latent)

# %%
nnd, nni = index.search(atac_latent, 50)

# %%
hits = rna.obs["cell_type"].to_numpy()[nni]

# %%
pred = pd.crosstab(
    np.repeat(atac.obs_names, nni.shape[1]), hits.ravel()
).idxmax(axis=1).loc[atac.obs_names]
pred = pd.Categorical(pred, categories=rna.obs["cell_type"].cat.categories)
atac.obs["iNMF"] = pred

# %% [markdown]
# ## Save results

# %%
atac.write(f"{PATH}/atac_transferred.h5ad", compression="gzip")
# atac = anndata.read_h5ad(f"{PATH}/atac_transferred.h5ad")

# %% [markdown]
# # Sankey

# %%
COLOR_MAP = {
    k: v for k, v in
    zip(atac.obs["cell_type"].cat.categories, atac.uns["cell_type_colors"])
}
link_cutoff = ceil(atac.shape[0] * 0.001)
link_color_map = defaultdict(lambda: "#CCCCCC")
link_color_map.update({
    ("Astrocytes", "Excitatory neurons"): COLOR_MAP["Excitatory neurons"],
    ("Astrocytes/Oligodendrocytes", "Astrocytes"): COLOR_MAP["Astrocytes"],
    ("Astrocytes/Oligodendrocytes", "Oligodendrocytes"): COLOR_MAP["Oligodendrocytes"]
})

# %%
fig = scglue.plot.sankey(
    atac.obs["NNLS"],
    atac.obs["GLUE"],
    title="NNLS vs GLUE transferred labels",
    left_color=lambda x: COLOR_MAP[x],
    right_color=lambda x: COLOR_MAP[x],
    link_color=lambda x: "rgba(0.9,0.9,0.9,0.2)" if x["value"] <= link_cutoff \
        else link_color_map[(x["left"], x["right"])],
    width=700, height=1400, font_size=14
)
pio.write_image(fig, f"{PATH}/glue_sankey.png", scale=10)

# %%
fig = scglue.plot.sankey(
    atac.obs["NNLS"],
    atac.obs["iNMF"],
    title="NNLS vs iNMF transferred labels",
    left_color=lambda x: COLOR_MAP[x],
    right_color=lambda x: COLOR_MAP[x],
    link_color=lambda x: "rgba(0.9,0.9,0.9,0.2)" if x["value"] <= link_cutoff \
        else link_color_map[(x["left"], x["right"])],
    width=700, height=1400, font_size=14
)
pio.write_image(fig, f"{PATH}/imnf_sankey.png", scale=10)
