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
import pathlib

import anndata
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib import rcParams

import scglue
import scglue.metrics

# %%
scglue.plot.set_publication_params()
rcParams["figure.figsize"] = (4, 4)

# %%
data_path = pathlib.Path("../../evaluation/results/raw/Muto-2021/original")
output_path = pathlib.Path("results/Muto-2021")
output_path.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Read data

# %%
rna = anndata.read_h5ad(data_path / "rna.h5ad")
atac = anndata.read_h5ad(data_path / "atac.h5ad")
graph = nx.read_graphml(data_path / "gene_region:combined-extend_range:0-corrupt_rate:0.0-corrupt_seed:0" / "sub.graphml.gz")

# %% [markdown]
# # Original result

# %%
rna.obsm["X_umap"] = pd.read_csv(
    data_path / "gene_region:combined-extend_range:0-corrupt_rate:0.0-corrupt_seed:0" / "GLUE" /
    "dim:50-alt_dim:100-hidden_depth:2-hidden_dim:256-dropout:0.2-lam_graph:0.02-lam_align:0.05-neg_samples:10" /
    "seed:4" / "rna_umap.csv", header=None, index_col=0
).to_numpy()
atac.obsm["X_umap"] = pd.read_csv(
    data_path / "gene_region:combined-extend_range:0-corrupt_rate:0.0-corrupt_seed:0" / "GLUE" /
    "dim:50-alt_dim:100-hidden_depth:2-hidden_dim:256-dropout:0.2-lam_graph:0.02-lam_align:0.05-neg_samples:10" /
    "seed:4" / "atac_umap.csv", header=None, index_col=0
).to_numpy()

# %%
combined = anndata.concat([rna, atac])

# %% tags=[]
combined_domain_sas = scglue.metrics.seurat_alignment_score(
    combined.obsm["X_umap"],
    combined.obs["domain"].to_numpy(),
    random_state=0
)

# %%
fig = sc.pl.umap(combined, color="domain", return_fig=True)
fig.axes[0].set_title(f"Combined omics layer (without BC)\nSeurat alignment score = {combined_domain_sas:.3f}")
fig.savefig(output_path / "combined_original_domain.pdf")

# %%
combined_batch_sas = scglue.metrics.seurat_alignment_score(
    combined.obsm["X_umap"],
    combined.obs["batch"].to_numpy(),
    random_state=0
)

# %%
fig = sc.pl.umap(combined, color="batch", return_fig=True)
fig.axes[0].set_title(f"Combined batch (without BC)\nSeurat alignment score = {combined_batch_sas:.3f}")
fig.savefig(output_path / "combined_original_batch.pdf")

# %%
combined_ct_map = scglue.metrics.mean_average_precision(
    combined.obsm["X_umap"],
    combined.obs["cell_type"].to_numpy()
)

# %%
fig = sc.pl.umap(combined, color="cell_type", return_fig=True)
fig.axes[0].set_title(f"Combined cell type (without BC)\nMean average precision = {combined_ct_map:.3f}")
fig.savefig(output_path / "combined_original_cell_type.pdf")

# %% tags=[]
rna_batch_sas = scglue.metrics.seurat_alignment_score(
    rna.obsm["X_umap"],
    rna.obs["batch"].to_numpy(),
    random_state=0
)

# %%
fig = sc.pl.umap(rna, color="batch", return_fig=True)
fig.axes[0].set_title(f"RNA batch (without BC)\nSeurat alignment score = {rna_batch_sas:.3f}")
fig.savefig(output_path / "rna_original_batch.pdf")

# %%
rna_ct_map = scglue.metrics.mean_average_precision(
    rna.obsm["X_umap"],
    rna.obs["cell_type"].to_numpy()
)

# %%
fig = sc.pl.umap(rna, color="cell_type", return_fig=True)
fig.axes[0].set_title(f"RNA cell type (without BC)\nMean average precision = {rna_ct_map:.3f}")
fig.savefig(output_path / "rna_original_cell_type.pdf")

# %%
atac_batch_sas = scglue.metrics.seurat_alignment_score(
    atac.obsm["X_umap"],
    atac.obs["batch"].to_numpy(),
    random_state=0
)

# %%
fig = sc.pl.umap(atac, color="batch", return_fig=True)
fig.axes[0].set_title(f"ATAC batch (without BC)\nSeurat alignment score = {atac_batch_sas:.3f}")
fig.savefig(output_path / "atac_original_batch.pdf")

# %%
atac_ct_map = scglue.metrics.mean_average_precision(
    atac.obsm["X_umap"],
    atac.obs["cell_type"].to_numpy()
)

# %%
fig = sc.pl.umap(atac, color="cell_type", return_fig=True)
fig.axes[0].set_title(f"ATAC cell type (without BC)\nMean average precision = {atac_ct_map:.3f}")
fig.savefig(output_path / "atac_original_cell_type.pdf")

# %% [markdown]
# # Batch effect removal

# %%
atac.var["highly_variable"] = [graph.has_node(item) for item in atac.var_names]

# %%
rna.layers["counts"] = rna.X.copy()
sc.pp.normalize_total(rna)
sc.pp.log1p(rna)
sc.pp.scale(rna, max_value=10)
sc.tl.pca(rna, n_comps=100, svd_solver="auto")

# %%
scglue.data.lsi(
    atac, n_components=100,
    use_highly_variable=False, n_iter=15
)

# %%
scglue.models.configure_dataset(
    rna, "NB", use_layer="counts", use_highly_variable=True,
    use_rep="X_pca", use_batch="batch"
)
scglue.models.configure_dataset(
    atac, "NB", use_highly_variable=True,
    use_rep="X_lsi", use_batch="batch"
)

# %%
vertices = sorted(graph.nodes)
glue = scglue.models.SCGLUEModel(
    {"rna": rna, "atac": atac}, vertices,
    random_seed=4
)
glue.compile()
glue.fit(
    {"rna": rna, "atac": atac},
    graph, edge_weight="weight", edge_sign="sign",
    align_burnin=np.inf, safe_burnin=False,
    directory=output_path / "pretrain"
)
glue.save(output_path / "pretrain" / "final.dill")

# %%
rna.obsm["X_glue"] = glue.encode_data("rna", rna)
atac.obsm["X_glue"] = glue.encode_data("atac", atac)
scglue.data.estimate_balancing_weight(
    rna, atac, use_rep="X_glue", use_batch="batch"
)

# %%
scglue.models.configure_dataset(
    rna, "NB", use_layer="counts", use_highly_variable=True,
    use_rep="X_pca", use_batch="batch",
    use_dsc_weight="balancing_weight"
)
scglue.models.configure_dataset(
    atac, "NB", use_highly_variable=True,
    use_rep="X_lsi", use_batch="batch",
    use_dsc_weight="balancing_weight"
)

# %%
glue = scglue.models.SCGLUEModel(
    {"rna": rna, "atac": atac}, vertices,
    shared_batches=True, random_seed=4
)
glue.adopt_pretrained_model(scglue.models.load_model(
    output_path / "pretrain" / "final.dill"
))
glue.compile()
glue.fit(
    {"rna": rna, "atac": atac},
    graph, edge_weight="weight", edge_sign="sign",
    directory=output_path / "fine-tune"
)
glue.save(output_path / "fine-tune" / "final.dill")

# %%
rna.obsm["X_glue"] = glue.encode_data("rna", rna)
atac.obsm["X_glue"] = glue.encode_data("atac", atac)

# %%
combined = anndata.concat([rna, atac])
sc.pp.neighbors(combined, use_rep="X_glue", metric="cosine")
sc.tl.umap(combined)

# %%
combined_domain_sas = scglue.metrics.seurat_alignment_score(
    combined.obsm["X_umap"],
    combined.obs["domain"].to_numpy(),
    random_state=0
)

# %%
fig = sc.pl.umap(combined, color="domain", return_fig=True)
fig.axes[0].set_title(f"Combined omics layer (with BC)\nSeurat alignment score = {combined_domain_sas:.3f}")
fig.savefig(output_path / "combined_corrected_domain.pdf")

# %%
combined_batch_sas = scglue.metrics.seurat_alignment_score(
    combined.obsm["X_umap"],
    combined.obs["batch"].to_numpy(),
    random_state=0
)

# %%
fig = sc.pl.umap(combined, color="batch", return_fig=True)
fig.axes[0].set_title(f"Combined batch (with BC)\nSeurat alignment score = {combined_batch_sas:.3f}")
fig.savefig(output_path / "combined_corrected_batch.pdf")

# %%
combined_ct_map = scglue.metrics.mean_average_precision(
    combined.obsm["X_umap"],
    combined.obs["cell_type"].to_numpy()
)

# %%
fig = sc.pl.umap(combined, color="cell_type", return_fig=True)
fig.axes[0].set_title(f"Combined cell type (with BC)\nMean average precision = {combined_ct_map:.3f}")
fig.savefig(output_path / "combined_corrected_cell_type.pdf")

# %%
rna.obsm["X_glue_umap"] = combined[:rna.n_obs].obsm["X_umap"]
atac.obsm["X_glue_umap"] = combined[rna.n_obs:].obsm["X_umap"]

# %%
rna_batch_sas = scglue.metrics.seurat_alignment_score(
    rna.obsm["X_glue_umap"],
    rna.obs["batch"].to_numpy(),
    random_state=0
)

# %%
fig = sc.pl.embedding(rna, "X_glue_umap", color="batch", return_fig=True)
fig.axes[0].set_xlabel("UMAP1")
fig.axes[0].set_ylabel("UMAP2")
fig.axes[0].set_title(f"RNA batch (with BC)\nSeurat alignment score = {rna_batch_sas:.3f}")
fig.savefig(output_path / "rna_corrected_batch.pdf")

# %%
rna_ct_map = scglue.metrics.mean_average_precision(
    rna.obsm["X_glue_umap"],
    rna.obs["cell_type"].to_numpy()
)

# %%
fig = sc.pl.embedding(rna, "X_glue_umap", color="cell_type", return_fig=True)
fig.axes[0].set_xlabel("UMAP1")
fig.axes[0].set_ylabel("UMAP2")
fig.axes[0].set_title(f"RNA cell type (with BC)\nMean average precision = {rna_ct_map:.3f}")
fig.savefig(output_path / "rna_corrected_cell_type.pdf")

# %%
atac_batch_sas = scglue.metrics.seurat_alignment_score(
    atac.obsm["X_glue_umap"],
    atac.obs["batch"].to_numpy(),
    random_state=0
)

# %%
fig = sc.pl.embedding(atac, "X_glue_umap", color="batch", return_fig=True)
fig.axes[0].set_xlabel("UMAP1")
fig.axes[0].set_ylabel("UMAP2")
fig.axes[0].set_title(f"ATAC batch (with BC)\nSeurat alignment score={atac_batch_sas:.3f}")
fig.savefig(output_path / "atac_corrected_batch.pdf")

# %%
atac_ct_map = scglue.metrics.mean_average_precision(
    atac.obsm["X_glue_umap"],
    atac.obs["cell_type"].to_numpy()
)

# %%
fig = sc.pl.embedding(atac, "X_glue_umap", color="cell_type", return_fig=True)
fig.axes[0].set_xlabel("UMAP1")
fig.axes[0].set_ylabel("UMAP2")
fig.axes[0].set_title(f"ATAC cell type (with BC)\nMean average precision={atac_ct_map:.3f}")
fig.savefig(output_path / "atac_corrected_cell_type.pdf")
