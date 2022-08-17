r"""
Tests for the :mod:`scglue.models.nn` module
"""

# pylint: disable=wildcard-import, unused-wildcard-import, redefined-outer-name

import numpy as np
import pytest

import scglue
import scglue.models.data

from ..fixtures import *
from ..utils import cmp_arrays


def test_dataset():
    scglue.config.FORCE_TERMINATE_WORKER_PATIENCE = 1
    dataset = scglue.models.data.Dataset()
    dataset.prepare_shuffle(num_workers=1)
    assert dataset.has_workers
    dataset.prepare_shuffle(num_workers=2)
    dataset.clean()
    scglue.config.FORCE_TERMINATE_WORKER_PATIENCE = 60


def test_array_dataset(rna, atac):
    dataset = scglue.models.data.ArrayDataset(rna.X, atac.X)
    assert dataset.size == max(dataset.sizes)
    _ = dataset.random_split([0.6, 0.4])
    with pytest.raises(ValueError):
        _ = dataset.random_split([0.2, 0.3])
    with pytest.raises(ValueError):
        _ = dataset.random_split([0.0, 1.0])
    with pytest.raises(ValueError):
        _ = scglue.models.data.ArrayDataset(np.empty(0))


@pytest.mark.parametrize("use_obs_names", [True, False])
def test_anndataset(rna, atac, use_obs_names):
    scglue.models.configure_dataset(rna, "NB", use_highly_variable=False, use_layer="arange", use_obs_names=use_obs_names)
    scglue.models.configure_dataset(atac, "NB", use_highly_variable=False, use_layer="arange", use_obs_names=use_obs_names)
    with pytest.raises(ValueError):
        dataset = scglue.models.data.AnnDataset(
            [rna[[], :], atac],
            [rna.uns[scglue.config.ANNDATA_KEY], atac.uns[scglue.config.ANNDATA_KEY]],
            mode="train", getitem_size=5
        )
    with pytest.raises(ValueError):
        dataset = scglue.models.data.AnnDataset(
            [rna, atac],
            [rna.uns[scglue.config.ANNDATA_KEY]],
            mode="train", getitem_size=5
        )
    with pytest.raises(ValueError):
        dataset = scglue.models.data.AnnDataset(
            [rna, atac],
            [rna.uns[scglue.config.ANNDATA_KEY], atac.uns[scglue.config.ANNDATA_KEY]],
            mode="xxx", getitem_size=5
        )
    dataset = scglue.models.data.AnnDataset(
        [rna, atac],
        [rna.uns[scglue.config.ANNDATA_KEY], atac.uns[scglue.config.ANNDATA_KEY]],
        getitem_size=5
    )
    with pytest.raises(ValueError):
        dataset.random_split([-0.2, 1.2])
    with pytest.raises(ValueError):
        dataset.random_split([0.2, 0.3])
    dataset.prepare_shuffle()
    dataset.shuffle()
    for i in range(len(dataset)):
        x1, x2, *_, pmsk = dataset[i]
        x1, x2, pmsk = x1.numpy(), x2.numpy(), pmsk.numpy()
        paired = np.logical_and(pmsk[:, 0], pmsk[:, 1])
        if not use_obs_names:
            assert not paired.any()
        elif paired.any():
            cmp_arrays(x1[paired, 0], x2[paired, 0])
    dataset.clean()


def test_graph_dataset(graph):
    nx.set_edge_attributes(graph, 1, "sign")
    vertices = pd.Index(graph.nodes)
    _ = scglue.models.data.GraphDataset(graph, vertices)
    _ = scglue.models.data.GraphDataset(graph, vertices, weighted_sampling=False)
    _ = scglue.models.data.GraphDataset(graph, vertices, deemphasize_loops=False)
    vertices_ = vertices[:3]
    with pytest.raises(ValueError):
        _ = scglue.models.data.GraphDataset(graph, vertices_)
    graph_ = graph.copy()
    graph_.edges["a", "a", 0]["weight"] = -0.5
    with pytest.raises(ValueError):
        _ = scglue.models.data.GraphDataset(graph_, vertices)
    graph_ = graph.copy()
    graph_.edges["a", "a", 0]["sign"] = 0.3
    with pytest.raises(ValueError):
        _ = scglue.models.data.GraphDataset(graph_, vertices)


def test_parallel_dataloader():
    pdl = scglue.models.data.ParallelDataLoader(range(3), range(5), cycle_flags=[False, False])
    for i, _ in enumerate(pdl):
        pass
    assert i == 2  # pylint: disable=undefined-loop-variable
    pdl = scglue.models.data.ParallelDataLoader(range(3), range(5), cycle_flags=[True, False])
    for i, _ in enumerate(pdl):
        pass
    assert i == 4  # pylint: disable=undefined-loop-variable

    with pytest.raises(ValueError):
        _ = scglue.models.data.ParallelDataLoader(range(3), range(5), cycle_flags=[False])
