r"""
Auxiliary functions for :class:`anndata.AnnData` objects
that are not covered in :mod:`scanpy`.
"""

import os
from typing import Callable, List, Mapping, Optional

import anndata
import numpy as np
import pandas as pd
import scipy.sparse
import sklearn.decomposition
import sklearn.feature_extraction.text
import sklearn.preprocessing
import sklearn.neighbors
import sklearn.utils.extmath

from . import genomics, num
from .utils import logged


def lsi(
        adata: anndata.AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs
) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)

    Parameters
    ----------
    adata
        Input dataset
    n_components
        Number of dimensions to use
    use_highly_variable
        Whether to use highly variable features only, stored in
        ``adata.var['highly_variable']``. By default uses them if they
        have been determined beforehand.
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.utils.extmath.randomized_svd`
    """
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = num.tfidf(adata_use.X)
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi


@logged
def get_gene_annotation(
        adata: anndata.AnnData, var_by: str = None,
        gtf: os.PathLike = None, gtf_by: str = None,
        by_func: Optional[Callable] = None
) -> None:
    r"""
    Get genomic annotation of genes by joining with a GTF file.

    Parameters
    ----------
    adata
        Input dataset
    var_by
        Specify a column in ``adata.var`` used to merge with GTF attributes,
        otherwise ``adata.var_names`` is used by default.
    gtf
        Path to the GTF file
    gtf_by
        Specify a field in the GTF attributes used to merge with ``adata.var``,
        e.g. "gene_id", "gene_name".
    by_func
        Specify an element-wise function used to transform merging fields,
        e.g. removing suffix in gene IDs.

    Note
    ----
    The genomic locations are converted to 0-based as specified
    in bed format rather than 1-based as specified in GTF format.
    """
    if gtf is None or gtf_by is None:
        raise ValueError("Arguments `gtf` and `gtf_by` must be specified!")
    var_by = adata.var_names if var_by is None else adata.var[var_by]
    gtf = genomics.Gtf.read_gtf(gtf).query("feature == 'gene'").split_attribute()
    if by_func:
        by_func = np.vectorize(by_func)
        var_by = by_func(var_by)
        gtf[gtf_by] = by_func(gtf[gtf_by])  # Safe inplace modification
    gtf = gtf.sort_values("seqname").drop_duplicates(
        subset=[gtf_by], keep="last"
    )  # Typically, scaffolds come first, chromosomes come last
    merge_df = pd.concat([
        pd.DataFrame(gtf.to_bed(name=gtf_by)),
        pd.DataFrame(gtf).drop(columns=genomics.Gtf.COLUMNS)  # Only use the splitted attributes
    ], axis=1).set_index(gtf_by).reindex(var_by).set_index(adata.var.index)
    adata.var = pd.concat([adata.var, merge_df], axis=1)


def aggregate_obs(
        adata: anndata.AnnData, by: str, X_agg: Optional[str] = "sum",
        obs_agg: Optional[Mapping[str, str]] = None,
        obsm_agg: Optional[Mapping[str, str]] = None,
        layers_agg: Optional[Mapping[str, str]] = None
) -> anndata.AnnData:
    r"""
    Aggregate obs in a given dataset by certain categories

    Parameters
    ----------
    adata
        Dataset to be aggregated
    by
        Specify a column in ``adata.obs`` used for aggregation,
        must be discrete.
    X_agg
        Aggregation function for ``adata.X``, must be one of
        ``{"sum", "mean", ``None``}``. Setting to ``None`` discards
        the ``adata.X`` matrix.
    obs_agg
        Aggregation methods for ``adata.obs``, indexed by obs columns,
        must be one of ``{"sum", "mean", "majority"}``, where ``"sum"``
        and ``"mean"`` are for continuous data, and ``"majority"`` is for
        discrete data. Fields not specified will be discarded.
    obsm_agg
        Aggregation methods for ``adata.obsm``, indexed by obsm keys,
        must be one of ``{"sum", "mean"}``. Fields not specified will be
        discarded.
    layers_agg
        Aggregation methods for ``adata.layers``, indexed by layer keys,
        must be one of ``{"sum", "mean"}``. Fields not specified will be
        discarded.

    Returns
    -------
    aggregated
        Aggregated dataset
    """
    obs_agg = obs_agg or {}
    obsm_agg = obsm_agg or {}
    layers_agg = layers_agg or {}

    by = adata.obs[by]
    agg_idx = pd.Index(by.cat.categories) \
        if pd.api.types.is_categorical_dtype(by) \
        else pd.Index(np.unique(by))
    agg_sum = scipy.sparse.coo_matrix((
        np.ones(adata.shape[0]), (
            agg_idx.get_indexer(by),
            np.arange(adata.shape[0])
        )
    )).tocsr()
    agg_mean = agg_sum.multiply(1 / agg_sum.sum(axis=1))

    agg_method = {
        "sum": lambda x: agg_sum @ x,
        "mean": lambda x: agg_mean @ x,
        "majority": lambda x: pd.crosstab(by, x).idxmax(axis=1).loc[agg_idx].to_numpy()
    }

    X = agg_method[X_agg](adata.X) if X_agg and adata.X is not None else None
    obs = pd.DataFrame({
        k: agg_method[v](adata.obs[k])
        for k, v in obs_agg.items()
    }, index=agg_idx.astype(str))
    obsm = {
        k: agg_method[v](adata.obsm[k])
        for k, v in obsm_agg.items()
    }
    layers = {
        k: agg_method[v](adata.layers[k])
        for k, v in layers_agg.items()
    }
    for c in obs:
        if pd.api.types.is_categorical_dtype(adata.obs[c]):
            obs[c] = pd.Categorical(obs[c], categories=adata.obs[c].cat.categories)
    return anndata.AnnData(
        X=X, obs=obs, var=adata.var,
        obsm=obsm, varm=adata.varm, layers=layers
    )


def transfer_labels(
        ref: anndata.AnnData, query: anndata.AnnData, field: str,
        n_neighbors: int = 5, use_rep: Optional[str] = None,
        key_added: Optional[str] = None, **kwargs
) -> None:
    r"""
    Transfer discrete labels from reference dataset to query dataset

    Parameters
    ----------
    ref
        Reference dataset
    query
        Query dataset
    field
        Field to be transferred in ``ref.obs`` (must be discrete)
    n_neighbors
        Number of nearest neighbors used for label transfer
    use_rep
        Data representation based on which to find nearest neighbors,
        by default uses ``{ref, query}.X``.
    key_added
        New ``query.obs`` key added for the transfered labels,
        by default the same as ``field``.
    **kwargs
        Additional keyword arguments are passed to
        :class:`sklearn.neighbors.NearestNeighbors`
    """
    ref_mat = ref.obsm[use_rep] if use_rep else ref.X
    query_mat = query.obsm[use_rep] if use_rep else query.X
    nn = sklearn.neighbors.NearestNeighbors(
        n_neighbors=n_neighbors, **kwargs
    ).fit(ref_mat)
    nni = nn.kneighbors(query_mat, return_distance=False)
    hits = ref.obs[field].to_numpy()[nni]
    pred = pd.crosstab(
        np.repeat(query.obs_names, n_neighbors), hits.ravel()
    ).idxmax(axis=1).loc[query.obs_names]
    if pd.api.types.is_categorical_dtype(ref.obs[field]):
        pred = pd.Categorical(pred, categories=ref.obs[field].cat.categories)
    query.obs[key_added or field] = pred


def extract_rank_genes_groups(
        adata: anndata.AnnData, groups: Optional[List[str]] = None,
        filter_by: str = "pvals_adj < 0.01", sort_by: str = "scores",
        ascending: str = False
) -> pd.DataFrame:
    r"""
    Extract result of :func:`scanpy.tl.rank_genes_groups` in the form of
    marker gene data frame for specific cell groups

    Parameters
    ----------
    adata
        Input dataset
    groups
        Target groups for which markers should be extracted,
        by default extract all groups.
    filter_by
        Marker filtering criteria (passed to :meth:`pandas.DataFrame.query`)
    sort_by
        Column used for sorting markers
    ascending
        Whether to sort in ascending order

    Returns
    -------
    marker_df
        Extracted marker data frame

    Note
    ----
    Markers shared by multiple groups will be assign to the group
    with highest score.
    """
    if "rank_genes_groups" not in adata.uns:
        raise ValueError("Please call `sc.tl.rank_genes_groups` first!")
    if groups is None:
        groups = adata.uns["rank_genes_groups"][sort_by].dtype.names
    df = pd.concat([
        pd.DataFrame({
            k: np.asarray(v[g])
            for k, v in adata.uns["rank_genes_groups"].items()
            if k != "params"
        }).assign(group=g)
        for g in groups
    ])
    df["group"] = pd.Categorical(df["group"], categories=groups)
    df = df.sort_values(
        sort_by, ascending=ascending
    ).drop_duplicates(
        subset=["names"], keep="first"
    ).sort_values(
        ["group", sort_by], ascending=[True, ascending]
    ).query(filter_by)
    df = df.reset_index(drop=True)
    return df
