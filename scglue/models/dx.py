r"""
Model diagnostics
"""

from typing import Mapping, Optional

import h5py
import networkx as nx
import pandas as pd
from anndata import AnnData
from anndata._core.sparse_dataset import SparseDataset

from ..data import metacell_corr
from .scglue import SCGLUEModel


def integration_consistency(
        model: SCGLUEModel, adatas: Mapping[str, AnnData], graph: nx.Graph,
        edge_weight: str = "weight", edge_sign: str = "sign",
        count_layers: Optional[Mapping[str, str]] = None
) -> pd.DataFrame:
    r"""
    Integration consistency score, defined as the consistency between
    aligned-space meta-cell correlation and the prior graph

    Parameters
    ----------
    model
        Integration model to be evaluated
    adatas
        Datasets (indexed by domain name)
    graph
        Prior graph
    edge_weight
        Key of edge attribute for edge weight
    edge_sign
        Key of edge attribute for edge sign
    count_layers
        Dataset layers that contain raw counts (indexed by domain name).
        If not provided, it is assumed that ``.X`` is the raw counts.

    Returns
    -------
    consistency_df
        Consistency score at different numbers of meta cells
    """
    for adata in adatas.values():
        if isinstance(adata.X, (h5py.Dataset, SparseDataset)):
            raise RuntimeError("Backed data is not currently supported!")
    adatas = {
        k: AnnData(
            X=adata.X, obs=adata.obs, var=adata.var,
            obsm=adata.obsm.copy(), layers=adata.layers
        ) for k, adata in adatas.items()
    }  # Avoid unwanted updates to the input objects
    for k, adata in adatas.items():
        adata.obsm["X_glue"] = model.encode_data(k, adata)

    count_layers = count_layers or {}
    for k, count_layer in count_layers.items():
        adatas[k].X = adatas[k].layers[count_layer]

    n_metas, consistencies = [], []
    for n_meta in (10, 20, 50, 100, 200):
        if n_meta > min(adata.shape[0] for adata in adatas.values()):
            continue
        corr = metacell_corr(
            *adatas.values(), use_rep="X_glue",
            n_meta=n_meta, skeleton=graph
        )
        corr = corr.edge_subgraph(e for e in corr.edges if e[0] != e[1])  # Exclude self-loops
        edgelist = nx.to_pandas_edgelist(corr)
        n_metas.append(n_meta)
        consistencies.append((
            edgelist[edge_sign] * edgelist[edge_weight] * edgelist["corr"]
        ).sum() / edgelist[edge_weight].sum())
    return pd.DataFrame({
        "n_meta": n_metas,
        "consistency": consistencies
    })
