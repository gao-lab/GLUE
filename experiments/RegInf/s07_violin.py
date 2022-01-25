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
import matplotlib.pyplot as plt
import scanpy as sc
from matplotlib import rcParams

import scglue

# %%
scglue.plot.set_publication_params()
rcParams["figure.figsize"] = (6, 4)

PATH = "s07_violin"
os.makedirs(PATH, exist_ok=True)

# %%
rna = anndata.read_h5ad("s01_preprocessing/rna.h5ad")
sc.pp.normalize_total(rna)
sc.pp.log1p(rna)

# %%
fig, ax = plt.subplots()
sc.pl.violin(rna, "NCF2", groupby="cell_type", rotation=70, show=False, ax=ax)
for item in ax.collections[10:]:
    item.set_rasterized(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.savefig(f"{PATH}/NCF2.pdf")

# %%
fig, ax = plt.subplots()
sc.pl.violin(rna, "SPI1", groupby="cell_type", rotation=70, show=False, ax=ax)
for item in ax.collections[10:]:
    item.set_rasterized(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.savefig(f"{PATH}/SPI1.pdf")

# %%
fig, ax = plt.subplots()
sc.pl.violin(rna, "CD83", groupby="cell_type", rotation=70, show=False, ax=ax)
for item in ax.collections[10:]:
    item.set_rasterized(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.savefig(f"{PATH}/CD83.pdf")

# %%
fig, ax = plt.subplots()
sc.pl.violin(rna, "BCL11A", groupby="cell_type", rotation=70, show=False, ax=ax)
for item in ax.collections[10:]:
    item.set_rasterized(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.savefig(f"{PATH}/BCL11A.pdf")

# %%
fig, ax = plt.subplots()
sc.pl.violin(rna, "PAX5", groupby="cell_type", rotation=70, show=False, ax=ax)
for item in ax.collections[10:]:
    item.set_rasterized(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.savefig(f"{PATH}/PAX5.pdf")

# %%
fig, ax = plt.subplots()
sc.pl.violin(rna, "RELB", groupby="cell_type", rotation=70, show=False, ax=ax)
for item in ax.collections[10:]:
    item.set_rasterized(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.savefig(f"{PATH}/RELB.pdf")
