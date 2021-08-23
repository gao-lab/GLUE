r"""
Tests for the :mod:`scglue.models.nn` module
"""

# pylint: disable=wildcard-import, unused-wildcard-import, redefined-outer-name

import numpy as np
import pytest

import scglue
import scglue.models.nn

from ..fixtures import *


def test_dataset():
    scglue.config.FORCE_TERMINATE_WORKER_PATIENCE = 1
    dataset = scglue.models.nn.Dataset()
    dataset.prepare_shuffle(num_workers=1)
    assert dataset.has_workers
    dataset.prepare_shuffle(num_workers=2)
    dataset.clean()
    scglue.config.FORCE_TERMINATE_WORKER_PATIENCE = 60


def test_array_dataset(rna, atac):
    dataset = scglue.models.nn.ArrayDataset(rna.X, atac.X, grouping=[0, 1])
    _ = dataset.random_split([0.6, 0.4])
    with pytest.raises(ValueError):
        _ = dataset.random_split([0.2, 0.3])
    with pytest.raises(ValueError):
        _ = dataset.random_split([0.0, 1.0])
    with pytest.raises(ValueError):
        _ = scglue.models.nn.ArrayDataset(rna.X, atac.X, grouping=[0])
    with pytest.raises(ValueError):
        _ = scglue.models.nn.ArrayDataset(rna.X, atac.X, grouping=[0, 0])
    with pytest.raises(ValueError):
        _ = scglue.models.nn.ArrayDataset(np.empty(0))


def test_graph_dataset(eidx, ewt, esgn):
    _ = scglue.models.nn.GraphDataset(eidx, ewt, esgn)
    _ = scglue.models.nn.GraphDataset(eidx, ewt, esgn, weighted_sampling=False)
    _ = scglue.models.nn.GraphDataset(eidx, ewt, esgn, deemphasize_loops=False)
    with pytest.raises(ValueError):
        _ = scglue.models.nn.GraphDataset(eidx[0], ewt, esgn)
    with pytest.raises(ValueError):
        _ = scglue.models.nn.GraphDataset(eidx, ewt[:2], esgn)
    eidx_ = eidx.copy()
    eidx_[0, 0] = -1
    with pytest.raises(ValueError):
        _ = scglue.models.nn.GraphDataset(eidx_, ewt, esgn)
    ewt_ = ewt.copy()
    ewt_[0] = -0.5
    with pytest.raises(ValueError):
        _ = scglue.models.nn.GraphDataset(eidx, ewt_, esgn)
    esgn_ = esgn.copy()
    esgn_[0] = 0.3
    with pytest.raises(ValueError):
        _ = scglue.models.nn.GraphDataset(eidx, ewt, esgn_)


def test_parallel_dataloader():
    pdl = scglue.models.nn.ParallelDataLoader(range(3), range(5), cycle_flags=[False, False])
    for i, _ in enumerate(pdl):
        pass
    assert i == 2  # pylint: disable=undefined-loop-variable
    pdl = scglue.models.nn.ParallelDataLoader(range(3), range(5), cycle_flags=[True, False])
    for i, _ in enumerate(pdl):
        pass
    assert i == 4  # pylint: disable=undefined-loop-variable

    with pytest.raises(ValueError):
        _ = scglue.models.nn.ParallelDataLoader(range(3), range(5), cycle_flags=[False])
