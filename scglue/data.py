r"""
Auxiliary functions for :class:`anndata.AnnData` objects
that are not covered in :mod:`scanpy`.
"""

import os
from collections import defaultdict
from itertools import chain
from typing import Callable, List, Mapping, Optional

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
import scipy.stats
import sklearn.cluster
import sklearn.decomposition
import sklearn.feature_extraction.text
import sklearn.linear_model
import sklearn.neighbors
import sklearn.preprocessing
import sklearn.utils.extmath
from anndata import AnnData
from networkx.algorithms.bipartite import biadjacency_matrix
from sklearn.preprocessing import normalize
from sparse import COO
from tqdm.auto import tqdm

from . import genomics, num
from .typehint import Kws
from .utils import logged


def count_prep(adata: AnnData) -> None:
    r"""
    Standard preprocessing of count-based dataset with
    total count normalization and log-transformation

    Parameters
    ----------
    adata
        Dataset to be preprocessed
    """
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)


def lsi(
        adata: AnnData, n_components: int = 20,
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
    if "random_state" not in kwargs:
        kwargs["random_state"] = 0  # Keep deterministic as the default behavior
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = num.tfidf(adata_use.X)
    X_norm = normalize(X, norm="l1")
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi


@logged
def get_gene_annotation(
        adata: AnnData, var_by: str = None,
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
    if gtf is None:
        raise ValueError("Missing required argument `gtf`!")
    if gtf_by is None:
        raise ValueError("Missing required argument `gtf_by`!")
    var_by = adata.var_names if var_by is None else adata.var[var_by]
    gtf = genomics.read_gtf(gtf).query("feature == 'gene'").split_attribute()
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
    adata.var = adata.var.assign(**merge_df)


def aggregate_obs(
        adata: AnnData, by: str, X_agg: Optional[str] = "sum",
        obs_agg: Optional[Mapping[str, str]] = None,
        obsm_agg: Optional[Mapping[str, str]] = None,
        layers_agg: Optional[Mapping[str, str]] = None
) -> AnnData:
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
    return AnnData(
        X=X, obs=obs, var=adata.var,
        obsm=obsm, varm=adata.varm, layers=layers,
        dtype=None if X is None else X.dtype
    )


def transfer_labels(
        ref: AnnData, query: AnnData, field: str,
        n_neighbors: int = 30, use_rep: Optional[str] = None,
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

    Note
    ----
    First, nearest neighbors between reference and query cells are searched and
    weighted by Jaccard index of SNN (shared nearest neighbors). The Jaccard
    indices are then normalized per query cell to form a mapping matrix. To
    obtain predictions for query cells, we multiply the above mapping matrix to
    the one-hot matrix of reference labels. The category with the highest score
    is taken as the final prediction, while its score is interpreted as
    transfer confidence (stored as "{key_added}_confidence" in ``query.obs``).
    """
    xrep = ref.obsm[use_rep] if use_rep else ref.X
    yrep = query.obsm[use_rep] if use_rep else query.X
    xnn = sklearn.neighbors.NearestNeighbors(
        n_neighbors=n_neighbors, **kwargs
    ).fit(xrep)
    ynn = sklearn.neighbors.NearestNeighbors(
        n_neighbors=n_neighbors, **kwargs
    ).fit(yrep)
    xx = xnn.kneighbors_graph(xrep)
    xy = ynn.kneighbors_graph(xrep)
    yx = xnn.kneighbors_graph(yrep)
    yy = ynn.kneighbors_graph(yrep)
    jaccard = (xx @ yx.T) + (xy @ yy.T)
    jaccard.data /= 4 * n_neighbors - jaccard.data
    normalized_jaccard = jaccard.multiply(1 / jaccard.sum(axis=0))
    onehot = sklearn.preprocessing.OneHotEncoder()
    xtab = onehot.fit_transform(ref.obs[[field]])
    ytab = normalized_jaccard.T @ xtab
    pred = pd.Series(
        onehot.categories_[0][ytab.argmax(axis=1).A1],
        index=query.obs_names, dtype=ref.obs[field].dtype
    )
    conf = pd.Series(
        ytab.max(axis=1).toarray().ravel(),
        index=query.obs_names
    )
    key_added = key_added or field
    query.obs[key_added] = pred
    query.obs[key_added + "_confidence"] = conf


def extract_rank_genes_groups(
        adata: AnnData, groups: Optional[List[str]] = None,
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


def bedmap2anndata(
        bedmap: os.PathLike, var_col: int = 3, obs_col: int = 6
) -> AnnData:
    r"""
    Convert bedmap result to :class:`anndata.AnnData` object

    Parameters
    ----------
    bedmap
        Path to bedmap result
    var_col
        Variable column (0-based)
    obs_col
        Observation column (0-based)

    Returns
    -------
    adata
        Converted :class:`anndata.AnnData` object

    Note
    ----
    Similar to ``rliger::makeFeatureMatrix``,
    but more automated and memory efficient.
    """
    bedmap = pd.read_table(bedmap, sep="\t", header=None, usecols=[var_col, obs_col])
    var_names = pd.Index(sorted(set(bedmap[var_col])))
    bedmap = bedmap.dropna()
    var_pool = bedmap[var_col]
    obs_pool = bedmap[obs_col].str.split(";")
    obs_names = pd.Index(sorted(set(chain.from_iterable(obs_pool))))
    X = scipy.sparse.lil_matrix((var_names.size, obs_names.size))  # Transposed
    for obs, var in tqdm(zip(obs_pool, var_pool), total=bedmap.shape[0], desc="bedmap2anndata"):
        row = obs_names.get_indexer(obs)
        col = var_names.get_loc(var)
        X.rows[col] += row.tolist()
        X.data[col] += [1] * row.size
    X = X.tocsc().T  # Transpose back
    X.sum_duplicates()
    return AnnData(
        X=X, obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(index=var_names),
        dtype=X.dtype
    )


@logged
def estimate_balancing_weight(
        *adatas: AnnData, use_rep: str = None, use_batch: Optional[str] = None,
        resolution: float = 1.0, cutoff: float = 0.5, power: float = 4.0,
        key_added: str = "balancing_weight"
) -> None:
    r"""
    Estimate balancing weights in an unsupervised manner

    Parameters
    ----------
    *adatas
        Datasets to be balanced
    use_rep
        Data representation based on which to match clusters
    use_batch
        Estimate balancing per batch
        (batch keys and categories must match across all datasets)
    resolution
        Leiden clustering resolution
    cutoff
        Cosine similarity cutoff
    power
        Cosine similarity power (for increasing contrast)
    key_added
        New ``obs`` key added for the balancing weight

    Note
    ----
    While the joint similarity array would have a size of :math:`K^n`
    (where :math:`K` is the average number of clusters per dataset,
    and :math:`n` is the number of datasets), a sparse implementation
    was used, so the scalability regarding dataset number should be good.
    """
    if use_batch:  # Recurse per batch
        estimate_balancing_weight.logger.info("Splitting batches...")
        adatas_per_batch = defaultdict(list)
        for adata in adatas:
            groupby = adata.obs.groupby(use_batch, dropna=False)
            for b, idx in groupby.indices.items():
                adata_sub = adata[idx]
                adatas_per_batch[b].append(AnnData(
                    obs=adata_sub.obs,
                    obsm={use_rep: adata_sub.obsm[use_rep]}
                ))
        if len(set(len(items) for items in adatas_per_batch.values())) != 1:
            raise ValueError("Batches must match across datasets!")
        for b, items in adatas_per_batch.items():
            estimate_balancing_weight.logger.info("Processing batch %s...", b)
            estimate_balancing_weight(
                *items, use_rep=use_rep, use_batch=None,
                resolution=resolution, cutoff=cutoff,
                power=power, key_added=key_added
            )
        estimate_balancing_weight.logger.info("Collating batches...")
        collates = [
            pd.concat([item.obs[key_added] for item in items])
            for items in zip(*adatas_per_batch.values())
        ]
        for adata, collate in zip(adatas, collates):
            adata.obs[key_added] = collate.loc[adata.obs_names]
        return

    if use_rep is None:
        raise ValueError("Missing required argument `use_rep`!")
    adatas_ = [
        AnnData(
            obs=adata.obs.copy(deep=False).assign(n=1),
            obsm={use_rep: adata.obsm[use_rep]}
        ) for adata in adatas
    ]  # Avoid unwanted updates to the input objects

    estimate_balancing_weight.logger.info("Clustering cells...")
    for adata_ in adatas_:
        sc.pp.neighbors(
            adata_, n_pcs=adata_.obsm[use_rep].shape[1],
            use_rep=use_rep, metric="cosine"
        )
        sc.tl.leiden(adata_, resolution=resolution)

    leidens = [
        aggregate_obs(
            adata, by="leiden", X_agg=None,
            obs_agg={"n": "sum"}, obsm_agg={use_rep: "mean"}
        ) for adata in adatas_
    ]
    us = [normalize(leiden.obsm[use_rep], norm="l2") for leiden in leidens]
    ns = [leiden.obs["n"] for leiden in leidens]

    estimate_balancing_weight.logger.info("Matching clusters...")
    cosines = []
    for i, ui in enumerate(us):
        for j, uj in enumerate(us[i + 1:], start=i + 1):
            cosine = ui @ uj.T
            cosine[cosine < cutoff] = 0
            cosine = COO.from_numpy(cosine)
            cosine = np.power(cosine, power)
            key = tuple(
                slice(None) if k in (i, j) else np.newaxis
                for k in range(len(us))
            )  # To align axes
            cosines.append(cosine[key])
    joint_cosine = num.prod(cosines)
    estimate_balancing_weight.logger.info(
        "Matching array shape = %s...", str(joint_cosine.shape)
    )

    estimate_balancing_weight.logger.info("Estimating balancing weight...")
    for i, (adata, adata_, leiden, n) in enumerate(zip(adatas, adatas_, leidens, ns)):
        balancing = joint_cosine.sum(axis=tuple(
            k for k in range(joint_cosine.ndim) if k != i
        )).todense() / n
        balancing = pd.Series(balancing, index=leiden.obs_names)
        balancing = balancing.loc[adata_.obs["leiden"]].to_numpy()
        balancing /= balancing.sum() / balancing.size
        adata.obs[key_added] = balancing


@logged
def get_metacells(
        *adatas: AnnData, use_rep: str = None, n_meta: int = None,
        common: bool = True, seed: int = 0,
        agg_kws: Optional[List[Kws]] = None
) -> List[AnnData]:
    r"""
    Aggregate datasets into metacells

    Parameters
    ----------
    *adatas
        Datasets to be correlated
    use_rep
        Data representation based on which to cluster meta-cells
    n_meta
        Number of metacells to use
    common
        Whether to return only metacells common to all datasets
    seed
        Random seed for k-Means clustering
    agg_kws
        Keyword arguments per dataset passed to :func:`aggregate_obs`

    Returns
    -------
    adatas
        A list of AnnData objects containing the metacells

    Note
    ----
    When a single dataset is provided, the metacells are clustered
    with the dataset itself.
    When multiple datasets are provided, the metacells are clustered
    jointly with all datasets.
    """
    if use_rep is None:
        raise ValueError("Missing required argument `use_rep`!")
    if n_meta is None:
        raise ValueError("Missing required argument `n_meta`!")
    adatas = [
        AnnData(
            X=adata.X,
            obs=adata.obs.set_index(adata.obs_names + f"-{i}"), var=adata.var,
            obsm=adata.obsm, varm=adata.varm, layers=adata.layers,
            dtype=None if adata.X is None else adata.X.dtype
        ) for i, adata in enumerate(adatas)
    ]  # Avoid unwanted updates to the input objects

    get_metacells.logger.info("Clustering metacells...")
    combined = ad.concat(adatas)
    try:
        import faiss
        kmeans = faiss.Kmeans(
            combined.obsm[use_rep].shape[1], n_meta,
            gpu=False, seed=seed
        )
        kmeans.train(combined.obsm[use_rep])
        _, combined.obs["metacell"] = kmeans.index.search(combined.obsm[use_rep], 1)
    except ImportError:
        get_metacells.logger.warning(
            "`faiss` is not installed, using `sklearn` instead... "
            "This might be slow with a large number of cells. "
            "Consider installing `faiss` following the guide from "
            "https://github.com/facebookresearch/faiss/blob/main/INSTALL.md"
        )
        kmeans = sklearn.cluster.KMeans(n_clusters=n_meta, random_state=seed)
        combined.obs["metacell"] = kmeans.fit_predict(combined.obsm[use_rep])
    for adata in adatas:
        adata.obs["metacell"] = combined[adata.obs_names].obs["metacell"]

    get_metacells.logger.info("Aggregating metacells...")
    agg_kws = agg_kws or [{}] * len(adatas)
    if not len(agg_kws) == len(adatas):
        raise ValueError("Length of `agg_kws` must match the number of datasets!")
    adatas = [
        aggregate_obs(adata, "metacell", **kws)
        for adata, kws in zip(adatas, agg_kws)
    ]
    if common:
        common_metacells = list(set.intersection(*(
            set(adata.obs_names) for adata in adatas
        )))
        if len(common_metacells) == 0:
            raise RuntimeError("No common metacells found!")
        return [adata[common_metacells].copy() for adata in adatas]
    return adatas


def _metacell_corr(
        *adatas: AnnData, skeleton: nx.Graph = None, method: str = "spr",
        prep_fns: Optional[List[Optional[Callable[[AnnData], None]]]] = None
) -> nx.Graph:
    if skeleton is None:
        raise ValueError("Missing required argument `skeleton`!")
    if set.intersection(*(set(adata.var_names) for adata in adatas)):
        raise ValueError("Overlapping features are currently not supported!")
    prep_fns = prep_fns or [None] * len(adatas)
    if not len(prep_fns) == len(adatas):
        raise ValueError("Length of `prep_fns` must match the number of datasets!")
    for adata, prep_fn in zip(adatas, prep_fns):
        if prep_fn:
            prep_fn(adata)
    adata = ad.concat(adatas, axis=1)
    edgelist = nx.to_pandas_edgelist(skeleton)
    source = adata.var_names.get_indexer(edgelist["source"])
    target = adata.var_names.get_indexer(edgelist["target"])
    X = num.densify(adata.X.T)
    if method == "spr":
        X = np.array([scipy.stats.rankdata(x) for x in X])
    elif method != "pcc":
        raise ValueError(f"Unrecognized method: {method}!")
    mean = X.mean(axis=1)
    meansq = np.square(X).mean(axis=1)
    std = np.sqrt(meansq - np.square(mean))
    edgelist["corr"] = np.array([
        ((X[s] * X[t]).mean() - mean[s] * mean[t]) / (std[s] * std[t])
        for s, t in zip(source, target)
    ])
    return nx.from_pandas_edgelist(edgelist, edge_attr=True, create_using=type(skeleton))


@logged
def metacell_corr(
        *adatas: AnnData, skeleton: nx.Graph = None, method: str = "spr",
        agg_fns: Optional[List[str]] = None,
        prep_fns: Optional[List[Optional[Callable[[AnnData], None]]]] = None,
        **kwargs
) -> nx.Graph:
    r"""
    Metacell based correlation

    Parameters
    ----------
    *adatas
        Datasets to be correlated
    skeleton
        Skeleton graph determining which pair of features to correlate
    method
        Correlation method, must be one of {"pcc", "spr"}
    agg_fns
        Aggregation functions used to obtain metacells for each dataset,
        must be one of {"sum", "mean"}
    prep_fns
        Preprocessing functions to be applied to metacells for each dataset,
        ``None`` indicates no preprocessing
    **kwargs
        Additional keyword arguments are passed to :func:`get_metacells`

    Returns
    -------
    corr
        A skeleton-based graph containing correlation
        as edge attribute "corr"

    Note
    ----
    All aggregation, preprocessing and correlation apply to ``adata.X``.
    """
    adatas = get_metacells(*adatas, **kwargs, agg_kws=[
        dict(X_agg=agg_fn) for agg_fn in agg_fns
    ] if agg_fns else None)
    metacell_corr.logger.info(
        "Computing correlation on %d common metacells...",
        adatas[0].shape[0]
    )
    return _metacell_corr(
        *adatas, skeleton=skeleton, method=method, prep_fns=prep_fns
    )


def _metacell_regr(
        *adatas: AnnData, skeleton: nx.DiGraph = None,
        model: str = "Lasso", **kwargs
) -> nx.DiGraph:
    if skeleton is None:
        raise ValueError("Missing required argument `skeleton`!")
    for adata in adatas:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
    if set.intersection(*(set(adata.var_names) for adata in adatas)):
        raise ValueError("Overlapping features are currently not supported!")
    adata = ad.concat(adatas, axis=1)

    targets = [node for node, in_degree in skeleton.in_degree() if in_degree]
    biadj = biadjacency_matrix(
        skeleton, adata.var_names, targets, weight=None
    ).astype(bool).T.tocsr()
    X = num.densify(adata.X)
    Y = num.densify(adata[:, targets].X.T)
    coef = []
    model = getattr(sklearn.linear_model, model)
    for target, y, mask in tqdm(zip(targets, Y, biadj), total=len(targets), desc="metacell_regr"):
        X_ = X[:, mask.indices]
        lm = model(**kwargs).fit(X_, y)
        coef.append(pd.DataFrame({
            "source": adata.var_names[mask.indices],
            "target": target,
            "regr": lm.coef_
        }))
    coef = pd.concat(coef)
    return nx.from_pandas_edgelist(coef, edge_attr=True, create_using=type(skeleton))


@logged
def metacell_regr(
        *adatas: AnnData, use_rep: str = None, n_meta: int = None,
        skeleton: nx.DiGraph = None, model: str = "Lasso", **kwargs
) -> nx.DiGraph:
    r"""
    Metacell-based regression

    Parameters
    ----------
    *adatas
        Datasets to be correlated, where ``.X`` are raw counts
        (indexed by modality name)
    use_rep
        Data representation based on which to cluster meta-cells
    n_meta
        Number of metacells to use
    skeleton
        Skeleton graph determining which pair of features to correlate
    model
        Regression model (should be a class name under
        :mod:`sklearn.linear_model`)
    **kwargs
        Additional keyword arguments are passed to the regression model

    Returns
    -------
    regr
        A skeleton-based graph containing regression weights
        as edge attribute "regr"
    """
    for adata in adatas:
        if not num.all_counts(adata.X):
            raise ValueError("``.X`` must contain raw counts!")
    adatas = get_metacells(*adatas, use_rep=use_rep, n_meta=n_meta, common=True)
    metacell_regr.logger.info(
        "Computing regression on %d common metacells...",
        adatas[0].shape[0]
    )
    return _metacell_regr(*adatas, skeleton=skeleton, model=model, **kwargs)
