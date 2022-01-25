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
import functools
import itertools
import os

import anndata
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib import rcParams
from networkx.algorithms.bipartite import biadjacency_matrix

import scglue

# %%
scglue.plot.set_publication_params()
rcParams["figure.figsize"] = (4, 4)

PATH = "s01_preprocessing"
os.makedirs(PATH, exist_ok=True)

# %% [markdown]
# # Read data

# %%
rna = anndata.read_h5ad("../../data/dataset/10x-Multiome-Pbmc10k-RNA.h5ad")
rna

# %%
atac = anndata.read_h5ad("../../data/dataset/10x-Multiome-Pbmc10k-ATAC.h5ad")
atac

# %%
rna.obs["cell_type"].cat.categories

# %%
used_cts = {
    "CD4 Naive", "CD4 TCM", "CD4 TEM", "CD8 Naive", "CD8 TEM_1", "CD8 TEM_2",
    "CD14 Mono", "CD16 Mono", "Memory B", "Naive B"
}  # To match cell types covered in PC Hi-C
used_chroms = {f"chr{x}" for x in range(1, 23)}.union({"chrX"})

# %%
rna = rna[
    [item in used_cts for item in rna.obs["cell_type"]],
    [item in used_chroms for item in rna.var["chrom"]]
]
sc.pp.filter_genes(rna, min_counts=1)
rna.obs_names += "-RNA"
rna

# %%
atac = atac[
    [item in used_cts for item in atac.obs["cell_type"]],
    [item in used_chroms for item in atac.var["chrom"]]
]
sc.pp.filter_genes(atac, min_counts=1)
atac.obs_names += "-ATAC"
atac

# %%
genes = scglue.genomics.Bed(rna.var.assign(name=rna.var_names))
peaks = scglue.genomics.Bed(atac.var.assign(name=atac.var_names))
tss = genes.strand_specific_start_site()
promoters = tss.expand(2000, 0)

# %% [markdown]
# # RNA

# %%
rna.layers["counts"] = rna.X.copy()
sc.pp.highly_variable_genes(rna, n_top_genes=6000, flavor="seurat_v3")
sc.pp.normalize_total(rna)
sc.pp.log1p(rna)
sc.pp.scale(rna, max_value=10)
sc.tl.pca(rna, n_comps=100, use_highly_variable=True, svd_solver="auto")
sc.pp.neighbors(rna, n_pcs=100, metric="cosine")
sc.tl.umap(rna)

# %%
rna.X = rna.layers["counts"]
del rna.layers["counts"]

# %%
fig = sc.pl.umap(rna, color="cell_type", title="scRNA-seq cell type", return_fig=True)
fig.savefig(f"{PATH}/rna_ct.pdf")

# %% [markdown]
# # ATAC

# %%
scglue.data.lsi(atac, n_components=100, use_highly_variable=False, n_iter=15)
sc.pp.neighbors(atac, n_pcs=100, use_rep="X_lsi", metric="cosine")
sc.tl.umap(atac)

# %%
fig = sc.pl.umap(atac, color="cell_type", title="scATAC-seq cell type", return_fig=True)
fig.savefig(f"{PATH}/atac_ct.pdf")

# %% [markdown]
# # Build graph

# %% [markdown]
# ## Overlap

# %%
overlap_graph = scglue.genomics.window_graph(
    genes.expand(2000, 0), peaks, 0,
    attr_fn=lambda l, r, d: {
        "weight": 1.0,
        "type": "overlap"
    }
)
overlap_graph = nx.DiGraph(overlap_graph)
overlap_graph.number_of_edges()

# %% [markdown]
# ## Genomic distance

# %%
dist_graph = scglue.genomics.window_graph(
    promoters, peaks, 150000,
    attr_fn=lambda l, r, d: {
        "dist": abs(d),
        "weight": scglue.genomics.dist_power_decay(abs(d)),
        "type": "dist"
    }
)
dist_graph = nx.DiGraph(dist_graph)
dist_graph.number_of_edges()

# %% [markdown]
# ## pcHi-C

# %%
frags = pd.read_table(
    "../../data/hic/Javierre-2016/Human_hg38/Digest_Human_HindIII.rmap",
    header=None, names=["chrom", "chromStart", "chromEnd", "name"],
    dtype={"name": str}
)
frags["chromStart"] -= 1  # Originally 1-based, convert to 0-based as in BED
frags.index = frags["name"]
frags.head(n=3)

# %%
baits = pd.read_table(
    "../../data/hic/Javierre-2016/Human_hg38/Digest_Human_HindIII_baits_e75_ID.baitmap",
    header=None, names=["chrom", "chromStart", "chromEnd", "name", "targets"],
    usecols=["name"], dtype={"name": str}
)["name"].to_numpy()
baits = scglue.genomics.Bed(frags.loc[baits, :])

# %%
used_cts = ["Mon", "nCD4", "tCD4", "aCD4", "naCD4", "nCD8", "tCD8", "nB", "tB"]
bait_oe = pd.read_table(
    "../../data/hic/Javierre-2016/PCHiC_peak_matrix_cutoff5.tsv",
    usecols=["baitID", "oeID"] + used_cts, dtype={"baitID": str, "oeID": str}
)
bait_oe = bait_oe.loc[bait_oe.loc[:, used_cts].to_numpy().max(axis=1) > 5, ["baitID", "oeID"]]
bait_oe.shape

# %%
frags_set, baits_set = set(frags["name"]), set(baits["name"])
bait_oe = bait_oe.loc[[
    i in baits_set and j in frags_set
    for i, j in zip(bait_oe["baitID"], bait_oe["oeID"])
], :]  # Some frags might be missing if liftover is used
bait_oe.shape

# %%
bait_oe = pd.concat([bait_oe, pd.DataFrame({"baitID": baits.index, "oeID": baits.index})])  # Add same-frag links
bait_oe = nx.from_pandas_edgelist(bait_oe, source="baitID", target="oeID", create_using=nx.DiGraph)
oes = scglue.genomics.Bed(frags.loc[np.unique([e[1] for e in bait_oe.edges]), :])

# %%
gene_bait = scglue.genomics.window_graph(promoters, baits, 1000)
oe_peak = scglue.genomics.window_graph(oes, peaks, 1000)

# %%
pchic_graph = (
    biadjacency_matrix(gene_bait, genes.index, baits.index, weight=None) @
    biadjacency_matrix(bait_oe, baits.index, oes.index, weight=None) @
    biadjacency_matrix(oe_peak, oes.index, peaks.index, weight=None)
).tocoo()
pchic_graph.eliminate_zeros()
pchic_graph.data = np.minimum(pchic_graph.data, 1.0)
pchic_graph = nx.DiGraph([
    (genes.index[i], peaks.index[j], {"weight": k, "type": "pchic"})
    for i, j, k in zip(pchic_graph.row, pchic_graph.col, pchic_graph.data)
])
pchic_graph.number_of_edges()

# %%
rna.var["in_pchic"] = biadjacency_matrix(gene_bait, genes.index).sum(axis=1).A1 != 0
rna.var["in_pchic"].sum()

# %%
pchic_links = nx.to_pandas_edgelist(
    bait_oe, source="baitID", target="oeID"
).query(
    "baitID != oeID"
).merge(
    frags, how="left", left_on="baitID", right_index=True
).merge(
    frags, how="left", left_on="oeID", right_index=True
).merge(
    nx.to_pandas_edgelist(gene_bait, source="gene", target="baitID"), how="left", on="baitID"
).dropna(subset=["gene"]).assign(score=1).loc[:, [
    "chrom_x", "chromStart_x", "chromEnd_x",
    "chrom_y", "chromStart_y", "chromEnd_y",
    "score", "gene"
]]
pchic_links = pchic_links.query("chrom_x == chrom_y")
pchic_links.to_csv(f"{PATH}/pchic.annotated_links", sep="\t", index=False, header=False)

# %% [markdown]
# ## eQTL

# %%
gene_id_mapping = {ens: name for ens, name in zip(rna.var["gene_ids"], rna.var_names)}

# %%
eqtl = scglue.genomics.read_bed("../../data/eqtl/GTEx-v8/bed/Whole_Blood.v8.signif_variant_gene_pairs.bed.gz")
eqtl["name"] = eqtl["name"].map(scglue.genomics.ens_trim_version).map(gene_id_mapping)
eqtl = scglue.genomics.Bed(eqtl.df.dropna(subset=["name"]))

# %%
eqtl_graph = scglue.genomics.window_graph(
    eqtl, peaks, 0, left_sorted=True,
    attr_fn=lambda l, r, d: {
        "weight": 1.0,
        "type": "eqtl"
    }
)
eqtl_graph = nx.DiGraph(eqtl_graph)
eqtl_graph.number_of_edges()

# %%
eqtl_genes = pd.read_table(
    "../../data/eqtl/GTEx-v8/GTEx_Analysis_v8_eQTL/Whole_Blood.v8.egenes.txt.gz", usecols=["gene_id"]
)["gene_id"].map(scglue.genomics.ens_trim_version).map(gene_id_mapping).dropna()
eqtl_genes = set(eqtl_genes)
rna.var["in_eqtl"] = [item in eqtl_genes for item in rna.var_names]
rna.var["in_eqtl"].sum()

# %%
eqtl_links = eqtl.df.iloc[:, :4].merge(tss.df.iloc[:, :4], how="left", on="name").assign(score=1)
eqtl_links = eqtl_links.query("chrom_x == chrom_y")
eqtl_links["name"] = eqtl_links.pop("name")
eqtl_links.to_csv(f"{PATH}/eqtl.annotated_links", sep="\t", index=False, header=False)

# %% [markdown]
# # Update highly variable genes

# %%
rna.var["o_highly_variable"] = rna.var["highly_variable"]
rna.var["o_highly_variable"].sum()

# %%
rna.var["in_cicero"] = biadjacency_matrix(
    scglue.genomics.window_graph(promoters, peaks, 0),
    genes.index
).sum(axis=1).A1 > 0
rna.var["in_cicero"].sum()

# %%
rna.var["d_highly_variable"] = functools.reduce(np.logical_and, [
    rna.var["highly_variable"],
    rna.var["in_pchic"],
    rna.var["in_eqtl"],
    rna.var["in_cicero"]
])
rna.var["d_highly_variable"].sum()

# %%
rna.var["dcq_highly_variable"] = rna.var["highly_variable"]
rna.var["dcq_highly_variable"].sum()

# %% [markdown]
# # Combine graphs into priors

# %% [markdown]
# ## Overlap

# %%
o_prior = overlap_graph.copy()

# %%
hvg_reachable = scglue.graph.reachable_vertices(o_prior, rna.var.query("o_highly_variable").index)

# %%
atac.var["o_highly_variable"] = [item in hvg_reachable for item in atac.var_names]
atac.var["o_highly_variable"].sum()

# %%
o_prior = scglue.graph.compose_multigraph(o_prior, o_prior.reverse())
for item in itertools.chain(atac.var_names, rna.var_names):
    o_prior.add_edge(item, item, weight=1.0, type="self-loop")
nx.set_edge_attributes(o_prior, 1, "sign")

# %%
o_prior = o_prior.subgraph(hvg_reachable)

# %% [markdown]
# ## Genomic distance

# %%
d_prior = dist_graph.copy()

# %%
hvg_reachable = scglue.graph.reachable_vertices(d_prior, rna.var.query("d_highly_variable").index)

# %%
atac.var["d_highly_variable"] = [item in hvg_reachable for item in atac.var_names]
atac.var["d_highly_variable"].sum()

# %%
d_prior = scglue.graph.compose_multigraph(d_prior, d_prior.reverse())
for item in itertools.chain(atac.var_names, rna.var_names):
    d_prior.add_edge(item, item, weight=1.0, type="self-loop")
nx.set_edge_attributes(d_prior, 1, "sign")

# %%
d_prior = d_prior.subgraph(hvg_reachable)

# %% [markdown]
# ## Genomic distance + pcHi-C + eQTL

# %%
dcq_prior = scglue.graph.compose_multigraph(dist_graph, pchic_graph, eqtl_graph)

# %%
hvg_reachable = scglue.graph.reachable_vertices(dcq_prior, rna.var.query("dcq_highly_variable").index)

# %%
atac.var["dcq_highly_variable"] = [item in hvg_reachable for item in atac.var_names]
atac.var["dcq_highly_variable"].sum()

# %%
dcq_prior = scglue.graph.compose_multigraph(dcq_prior, dcq_prior.reverse())
for item in itertools.chain(atac.var_names, rna.var_names):
    dcq_prior.add_edge(item, item, weight=1.0, type="self-loop")
nx.set_edge_attributes(dcq_prior, 1, "sign")

# %%
dcq_prior = dcq_prior.subgraph(hvg_reachable)

# %% [markdown]
# # Write data

# %%
rna.write(f"{PATH}/rna.h5ad", compression="gzip")
atac.write(f"{PATH}/atac.h5ad", compression="gzip")

# %%
nx.write_graphml(overlap_graph, f"{PATH}/overlap.graphml.gz")
nx.write_graphml(dist_graph, f"{PATH}/dist.graphml.gz")
nx.write_graphml(pchic_graph, f"{PATH}/pchic.graphml.gz")
nx.write_graphml(eqtl_graph, f"{PATH}/eqtl.graphml.gz")

# %%
nx.write_graphml(o_prior, f"{PATH}/o_prior.graphml.gz")
nx.write_graphml(d_prior, f"{PATH}/d_prior.graphml.gz")
nx.write_graphml(dcq_prior, f"{PATH}/dcq_prior.graphml.gz")
