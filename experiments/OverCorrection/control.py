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
from pathlib import Path

import anndata
import networkx as nx
import scanpy as sc
from matplotlib import rcParams

import scglue

# %%
scglue.plot.set_publication_params()
rcParams["figure.figsize"] = (4, 4)
DATASET = os.environ.get("DATASET", "Chen-2019")
path = Path(f"control/{DATASET}+{DATASET}")
path.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Read data

# %%
rna = anndata.read_h5ad(f"../../evaluation/results/raw/{DATASET}/original/rna.h5ad")
atac = anndata.read_h5ad(f"../../evaluation/results/raw/{DATASET}/original/atac.h5ad")

# %%
subgraph = nx.read_graphml(f"../../evaluation/results/raw/{DATASET}/original/gene_region:combined-extend_range:0-corrupt_rate:0.0-corrupt_seed:0/sub.graphml.gz")

# %% [markdown]
# # Preprocessing

# %%
rna.layers["counts"] = rna.X.copy()
sc.pp.normalize_total(rna)
sc.pp.log1p(rna)
sc.pp.scale(rna, max_value=10)
sc.tl.pca(rna, n_comps=100, svd_solver="auto")
rna.X = rna.layers["counts"]
del rna.layers["counts"]

# %%
scglue.data.lsi(
    atac, n_components=100,
    use_highly_variable=False, n_iter=15
)

# %% [markdown]
# # GLUE

# %% [markdown]
# ## Load evaluation model

# %%
glue = scglue.models.load_model(f"../../evaluation/results/raw/{DATASET}/original/gene_region:combined-extend_range:0-corrupt_rate:0.0-corrupt_seed:0/GLUE/dim:50-alt_dim:100-hidden_depth:2-hidden_dim:256-dropout:0.2-lam_graph:0.02-lam_align:0.05-neg_samples:10/seed:0/final.dill")

# %% [markdown]
# ## Visualization

# %%
rna.obsm["X_glue"] = glue.encode_data("rna", rna)
atac.obsm["X_glue"] = glue.encode_data("atac", atac)

# %%
combined = anndata.concat([rna, atac])
sc.pp.neighbors(combined, use_rep="X_glue", metric="cosine")
sc.tl.umap(combined)

# %%
fig = sc.pl.umap(combined, color="domain", return_fig=True)
fig.axes[0].set_title("Combined omics layer")
fig.savefig(path / "combined_domain.pdf")

# %%
fig = sc.pl.umap(combined, color="cell_type", return_fig=True)
fig.axes[0].set_title("Combined cell type")
fig.savefig(path / "combined_cell_type.pdf")

# %%
rna.obsm["X_glue_umap"] = combined[:rna.n_obs].obsm["X_umap"]
atac.obsm["X_glue_umap"] = combined[rna.n_obs:].obsm["X_umap"]

# %%
fig = sc.pl.embedding(rna, "X_glue_umap", color="cell_type", return_fig=True)
fig.axes[0].set_xlabel("UMAP1")
fig.axes[0].set_ylabel("UMAP2")
fig.axes[0].set_title("RNA cell type")
fig.savefig(path / "rna_cell_type.pdf")

# %%
fig = sc.pl.embedding(atac, "X_glue_umap", color="cell_type", return_fig=True)
fig.axes[0].set_xlabel("UMAP1")
fig.axes[0].set_ylabel("UMAP2")
fig.axes[0].set_title("ATAC cell type")
fig.savefig(path / "atac_cell_type.pdf")

# %% [markdown]
# # Integration consistency

# %%
consistency_df = scglue.models.integration_consistency(glue, {"rna": rna, "atac": atac}, subgraph)
consistency_df.to_csv(path / "integration_consistency.csv", index=False)
