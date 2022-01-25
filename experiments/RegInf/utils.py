import itertools

import faiss
import numpy as np
import pandas as pd
import scipy.sparse
import seaborn as sns

import scglue
from scglue.data import _metacell_corr, _metacell_regr


def get_metacells_paired(rna, atac, use_rep, n_meta=200):
    kmeans = faiss.Kmeans(rna.obsm[use_rep].shape[1], n_meta, gpu=False, seed=0)
    kmeans.train(rna.obsm[use_rep])
    _, rna.obs["metacell"] = kmeans.index.search(rna.obsm[use_rep], 1)
    atac.obs["metacell"] = rna.obs["metacell"].to_numpy()
    rna_agg = scglue.data.aggregate_obs(rna, "metacell")
    atac_agg = scglue.data.aggregate_obs(atac, "metacell")
    common_metacells = np.intersect1d(rna_agg.obs_names, atac_agg.obs_names)
    rna_agg = rna_agg[common_metacells, :].copy()
    atac_agg = atac_agg[common_metacells, :].copy()
    return rna_agg, atac_agg


def metacell_corr(rna, atac, use_rep, n_meta=200, skeleton=None, method="spr"):
    print("Clustering metacells...")
    rna_agg, atac_agg = get_metacells_paired(rna, atac, use_rep, n_meta=n_meta)
    print("Computing correlation...")
    return _metacell_corr(rna_agg, atac_agg, skeleton=skeleton, method=method)


def metacell_regr(rna, atac, use_rep, n_meta=200, skeleton=None, model="Lasso", **kwargs):
    print("Clustering metacells...")
    rna_agg, atac_agg = get_metacells_paired(rna, atac, use_rep, n_meta=n_meta)
    print("Computing regression...")
    return _metacell_regr(rna_agg, atac_agg, skeleton=skeleton, model=model, **kwargs)


def make_dist_bins(dist, bins):
    r"""
    ``bins`` are in KB
    """
    labels = [f"{bins[i]}-{bins[i+1]} kb" for i in range(len(bins) - 1)]
    bins = np.asarray(bins) * 1e3
    return pd.cut(dist, bins, labels=labels, include_lowest=True)


def boxplot(x=None, y=None, hue=None, data=None):
    r"""
    Box plot with marginal distributions
    """
    assert x in data and y in data and hue in data
    data = data.copy(deep=False)
    if not pd.api.types.is_categorical_dtype(data[x]):
        data[x] = data[x].astype("category")
    if not pd.api.types.is_categorical_dtype(data[hue]):
        data[hue] = data[hue].astype("category")
    data[y] = data[y].astype(float)

    g = sns.JointGrid(x=x, y=y, data=data, height=5)
    sns.boxplot(
        x=x, y=y, hue=hue, data=data,
        saturation=1.0, showmeans=True,
        meanprops=dict(marker="^", markerfacecolor="white", markeredgecolor="black"),
        boxprops=dict(edgecolor="black"), medianprops=dict(color="black"),
        whiskerprops=dict(color="black"), capprops=dict(color="black"),
        flierprops=dict(marker=".", markerfacecolor="black", markeredgecolor="none", markersize=3),
        ax=g.ax_joint
    )
    sns.kdeplot(
        y=y, hue=hue, data=data,
        common_norm=False, shade=True, legend=False, ax=g.ax_marg_y
    )
    data = data.groupby(x)[hue].value_counts(normalize=True).rename("frac").reset_index()
    bottom = np.zeros(data[x].cat.categories.size)
    for _, d in data.groupby(hue):
        g.ax_marg_x.bar(d[x], d["frac"], bottom=bottom, width=0.7, edgecolor="black")
        bottom += d["frac"]
    return g


def cis_regulatory_ranking(
        gene_peak_conn: pd.DataFrame, peak_tf_binding: pd.DataFrame,
        genes: pd.DataFrame, peaks: pd.DataFrame, tfs: pd.DataFrame,
        n_samples: int = 1000, random_seed: int = 0
) -> None:
    gene_peak_component = scipy.sparse.coo_matrix((
        np.ones(gene_peak_conn.shape[0], dtype=np.int16), (
            genes.index.get_indexer(gene_peak_conn["gene"]),
            peaks.index.get_indexer(gene_peak_conn["peak"]),
        )
    ), shape=(genes.index.size, peaks.index.size))
    peak_tf_component = scipy.sparse.coo_matrix((
        np.ones(peak_tf_binding.shape[0], dtype=np.int16), (
            peaks.index.get_indexer(peak_tf_binding["peak"]),
            tfs.index.get_indexer(peak_tf_binding["tf"]),
        )
    ), shape=(peaks.index.size, tfs.index.size))

    peak_lens = (peaks["chromEnd"] - peaks["chromStart"]).reset_index(drop=True)
    peak_bins = pd.qcut(peak_lens, 500, duplicates="drop")
    peak_bins_lut = peak_bins.index.groupby(peak_bins)

    rs = np.random.RandomState(random_seed)
    row, rand_col, data = [], [], []
    lil = gene_peak_component.tolil()
    for r, (c, d) in scglue.utils.smart_tqdm(enumerate(zip(lil.rows, lil.data)), total=len(lil.rows)):
        if not c:  # Empty row
            continue
        row.append(np.ones_like(c) * r)
        rand_col.append(np.stack([
            rs.choice(peak_bins_lut[peak_bins[c_]], n_samples, replace=True)
            for c_ in c
        ], axis=0))
        data.append(d)
    row = np.concatenate(row)
    rand_col = np.concatenate(rand_col)
    data = np.concatenate(data)

    observed_gene_tf = (gene_peak_component @ peak_tf_component).toarray()
    rand_gene_tf = np.empty((genes.shape[0], tfs.shape[0], rand_col.shape[1]), dtype=np.int16)
    for k in scglue.utils.smart_tqdm(range(rand_col.shape[1])):
        rand_gene_peak_component = scipy.sparse.coo_matrix((
            data, (row, rand_col[:, k])
        ), shape=(genes.index.size, peaks.index.size))
        rand_gene_tf[:, :, k] = (rand_gene_peak_component @ peak_tf_component).toarray()
    rand_gene_tf.sort(axis=2)

    enrichment_gene_tf = np.empty_like(observed_gene_tf)
    for i, j in itertools.product(range(enrichment_gene_tf.shape[0]), range(enrichment_gene_tf.shape[1])):
        if observed_gene_tf[i, j] == 0:
            enrichment_gene_tf[i, j] = 0
            continue
        enrichment_gene_tf[i, j] = np.searchsorted(rand_gene_tf[i, j, :], observed_gene_tf[i, j], side="right")

    enrichment_gene_tf = pd.DataFrame(
        enrichment_gene_tf / rand_gene_tf.shape[2],
        index=genes.index, columns=tfs.index
    )
    rank_gene_tf = pd.DataFrame(
        scipy.stats.rankdata(-enrichment_gene_tf, axis=0),
        index=genes.index, columns=tfs.index
    )
    return enrichment_gene_tf, rank_gene_tf
