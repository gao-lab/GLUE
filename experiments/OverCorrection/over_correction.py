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
import itertools
import os
from pathlib import Path

import anndata
import numpy as np
import scanpy as sc
from matplotlib import rcParams

import scglue

# %%
scglue.plot.set_publication_params()
rcParams["figure.figsize"] = (4, 4)
DATASET1 = os.environ.get("DATASET1", "Chen-2019")
DATASET2 = os.environ.get("DATASET2", "Ma-2020")
path = Path(f"over_correction/{DATASET1}+{DATASET2}")
path.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Read data

# %%
rna = anndata.read_h5ad(f"../../data/dataset/{DATASET1}-RNA.h5ad")
atac = anndata.read_h5ad(f"../../data/dataset/{DATASET2}-ATAC.h5ad")

# %%
graph = scglue.genomics.rna_anchored_prior_graph(rna, atac, propagate_highly_variable=True)

# %%
subgraph = graph.subgraph(set(itertools.chain(
    rna.var.query("highly_variable").index,
    atac.var.query("highly_variable").index
)))

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
# ## Train model

# %%
scglue.models.configure_dataset(
    rna, "NB", use_highly_variable=True,
    use_rep="X_pca"
)
scglue.models.configure_dataset(
    atac, "NB", use_highly_variable=True,
    use_rep="X_lsi"
)

# %%
vertices = sorted(subgraph.nodes)
glue = scglue.models.SCGLUEModel(
    {"rna": rna, "atac": atac}, vertices,
    random_seed=0
)
glue.compile()
glue.fit(
    {"rna": rna, "atac": atac},
    subgraph, edge_weight="weight", edge_sign="sign",
    align_burnin=np.inf, safe_burnin=False,
    directory=path / "pretrain"
)
glue.save(path / "pretrain" / "final.dill")

# %%
rna.obsm["X_glue"] = glue.encode_data("rna", rna)
atac.obsm["X_glue"] = glue.encode_data("atac", atac)
scglue.data.estimate_balancing_weight(
    rna, atac, use_rep="X_glue"
)

# %%
scglue.models.configure_dataset(
    rna, "NB", use_highly_variable=True,
    use_rep="X_pca",
    use_dsc_weight="balancing_weight"
)
scglue.models.configure_dataset(
    atac, "NB", use_highly_variable=True,
    use_rep="X_lsi",
    use_dsc_weight="balancing_weight"
)

# %%
glue = scglue.models.SCGLUEModel(
    {"rna": rna, "atac": atac}, vertices,
    random_seed=0
)
glue.adopt_pretrained_model(scglue.models.load_model(
    path / "pretrain" / "final.dill"
))
glue.compile()
glue.fit(
    {"rna": rna, "atac": atac},
    subgraph, edge_weight="weight", edge_sign="sign",
    directory=path / "fine-tune"
)
glue.save(path / "fine-tune" / "final.dill")

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
