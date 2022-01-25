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
import re

import anndata
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib import rcParams

import scglue

# %%
scglue.plot.set_publication_params()
rcParams["figure.figsize"] = (4, 4)

PATH = "s02_glue"
os.makedirs(PATH, exist_ok=True)

# %%
PRIOR = os.environ.get("PRIOR", "d")
SEED = int(os.environ.get("SEED", "0"))

# %% [markdown]
# # Read data

# %%
rna = anndata.read_h5ad("s01_preprocessing/rna.h5ad")
atac = anndata.read_h5ad("s01_preprocessing/atac.h5ad")

# %%
prior = nx.read_graphml(
    f"s08_corrupt/{PRIOR}_prior.graphml.gz" if "corrupted" in PRIOR else
    f"s01_preprocessing/{PRIOR}_prior.graphml.gz"
)

# %% [markdown]
# # Assign highly variable genes

# %%
clean_prior = re.sub(r"^.*corrupted_", "", PRIOR)

# %%
rna.var["highly_variable"] = rna.var[f"{clean_prior}_highly_variable"]
atac.var["highly_variable"] = atac.var[f"{clean_prior}_highly_variable"]
rna.var["highly_variable"].sum(), atac.var["highly_variable"].sum()

# %% [markdown]
# # Train model

# %%
scglue.models.configure_dataset(rna, "NB", use_highly_variable=True, use_rep="X_pca")
scglue.models.configure_dataset(atac, "NB", use_highly_variable=True, use_rep="X_lsi")

# %% tags=[]
glue = scglue.models.SCGLUEModel(
    {"rna": rna, "atac": atac}, sorted(prior.nodes),
    random_seed=SEED
)
glue.compile()
glue.fit(
    {"rna": rna, "atac": atac},
    prior, edge_weight="weight", edge_sign="sign",
    align_burnin=np.inf, safe_burnin=False,
    directory=f"{PATH}/prior:{PRIOR}/seed:{SEED}/pretrain"
)
glue.save(f"{PATH}/prior:{PRIOR}/seed:{SEED}/pretrain/final.dill")

# %%
rna.obsm["X_glue"] = glue.encode_data("rna", rna)
atac.obsm["X_glue"] = glue.encode_data("atac", atac)
scglue.data.estimate_balancing_weight(
    rna, atac, use_rep="X_glue"
)

# %%
scglue.models.configure_dataset(rna, "NB", use_highly_variable=True, use_rep="X_pca", use_dsc_weight="balancing_weight")
scglue.models.configure_dataset(atac, "NB", use_highly_variable=True, use_rep="X_lsi", use_dsc_weight="balancing_weight")

# %%
glue = scglue.models.SCGLUEModel(
    {"rna": rna, "atac": atac}, sorted(prior.nodes),
    random_seed=SEED
)
glue.adopt_pretrained_model(scglue.models.load_model(
    f"{PATH}/prior:{PRIOR}/seed:{SEED}/pretrain/final.dill"
))
glue.compile()
glue.fit(
    {"rna": rna, "atac": atac},
    prior, edge_weight="weight", edge_sign="sign",
    directory=f"{PATH}/prior:{PRIOR}/seed:{SEED}/fine-tune"
)
glue.save(f"{PATH}/prior:{PRIOR}/seed:{SEED}/fine-tune/final.dill")

# %% [markdown]
# # Embeddings

# %% [markdown]
# ## Cell embeddings

# %%
rna.obsm["X_glue"] = glue.encode_data("rna", rna)
atac.obsm["X_glue"] = glue.encode_data("atac", atac)

# %%
combined = anndata.AnnData(
    obs=pd.concat([rna.obs, atac.obs], join="inner"),
    obsm={"X_glue": np.concatenate([rna.obsm["X_glue"], atac.obsm["X_glue"]])}
)

# %%
sc.pp.neighbors(combined, n_pcs=50, use_rep="X_glue", metric="cosine")
sc.tl.umap(combined)

# %%
fig = sc.pl.umap(combined, color="cell_type", title="Cell type", return_fig=True)
fig.savefig(f"{PATH}/prior:{PRIOR}/seed:{SEED}/combined_glue_ct.pdf")

# %%
fig = sc.pl.umap(combined, color="domain", title="Omics layer", return_fig=True)
fig.savefig(f"{PATH}/prior:{PRIOR}/seed:{SEED}/combined_glue_domain.pdf")

# %%
rna.write(f"{PATH}/prior:{PRIOR}/seed:{SEED}/rna_glue.h5ad", compression="gzip")
atac.write(f"{PATH}/prior:{PRIOR}/seed:{SEED}/atac_glue.h5ad", compression="gzip")
combined.write(f"{PATH}/prior:{PRIOR}/seed:{SEED}/combined_glue.h5ad", compression="gzip")

# %% [markdown]
# ## Feature embeddings

# %%
feature_embeddings = pd.DataFrame(
    glue.encode_graph(prior, "weight", "sign"),
    index=glue.vertices
)
feature_embeddings.iloc[:5, :5]

# %%
feature_embeddings.to_csv(f"{PATH}/prior:{PRIOR}/seed:{SEED}/feature_embeddings.csv", index=True, header=False)
