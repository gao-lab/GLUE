r"""
Model diagnostics
"""

from typing import Mapping

import h5py
import networkx as nx
import pandas as pd
from anndata import AnnData
from anndata._core.sparse_dataset import SparseDataset

from ..data import count_prep, metacell_corr
from ..utils import config, logged
from .scglue import SCGLUEModel


@logged
def integration_consistency(
        model: SCGLUEModel, adatas: Mapping[str, AnnData],
        graph: nx.Graph, **kwargs
) -> pd.DataFrame:
    r"""
    Integration consistency score, defined as the consistency between
    aligned-space meta-cell correlation and the guidance graph

    Parameters
    ----------
    model
        Integration model to be evaluated
    adatas
        Datasets (indexed by modality name)
    graph
        Guidance graph
    **kwargs
        Additional keyword arguments are passed to
        :func:`scglue.data.metacell_corr`

    Returns
    -------
    consistency_df
        Consistency score at different numbers of meta cells
    """
    for adata in adatas.values():
        if isinstance(adata.X, (h5py.Dataset, SparseDataset)):
            raise RuntimeError("Backed data is not currently supported!")
    logger = integration_consistency.logger

    adatas = {
        k: AnnData(
            X=adata.X, obs=adata.obs, var=adata.var,
            obsm=adata.obsm.copy(), layers=adata.layers,
            uns=adata.uns, dtype=adata.X.dtype
        ) for k, adata in adatas.items()
    }  # Avoid unwanted updates to the input objects
    for k, adata in adatas.items():
        adata.obsm["X_glue"] = model.encode_data(k, adata)

    for k, adata in adatas.items():
        use_layer = adata.uns[config.ANNDATA_KEY]["use_layer"]
        if use_layer:
            logger.info("Using layer \"%s\" for modality \"%s\"", use_layer, k)
            adata.X = adata.layers[use_layer]

    if "agg_fns" not in kwargs:
        agg_fns = []
        for k, adata in adatas.items():
            if adata.uns[config.ANNDATA_KEY]["prob_model"] in ("NB", "ZINB"):
                logger.info("Selecting aggregation \"sum\" for modality \"%s\"", k)
                agg_fns.append("sum")
            else:
                logger.info("Selecting aggregation \"mean\" for modality \"%s\"", k)
                agg_fns.append("mean")
        kwargs["agg_fns"] = agg_fns

    if "prep_fns" not in kwargs:
        prep_fns = []
        for k, adata in adatas.items():
            if adata.uns[config.ANNDATA_KEY]["prob_model"] in ("NB", "ZINB"):
                logger.info("Selecting log-norm preprocessing for modality \"%s\"", k)
                prep_fns.append(count_prep)
            else:
                logger.info("Selecting no preprocessing for modality \"%s\"", k)
                prep_fns.append(None)
        kwargs["prep_fns"] = prep_fns

    n_metas, consistencies = [], []
    for n_meta in (10, 20, 50, 100, 200):
        if n_meta > min(adata.shape[0] for adata in adatas.values()):
            continue
        corr = metacell_corr(
            *adatas.values(), skeleton=graph,
            use_rep="X_glue", n_meta=n_meta, **kwargs
        )
        corr = corr.edge_subgraph(e for e in corr.edges if e[0] != e[1])  # Exclude self-loops
        edgelist = nx.to_pandas_edgelist(corr)
        n_metas.append(n_meta)
        consistencies.append((
            edgelist["sign"] * edgelist["weight"] * edgelist["corr"]
        ).sum() / edgelist["weight"].sum())
    return pd.DataFrame({
        "n_meta": n_metas,
        "consistency": consistencies
    })
