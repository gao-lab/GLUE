r"""
Tests for the :mod:`scglue.metrics` module
"""

# pylint: disable=redefined-outer-name, wildcard-import, unused-wildcard-import


import numpy as np
import pytest

import scglue.metrics

from .fixtures import *


def test_mean_average_precision(rna_pp):
    mean_average_precision = scglue.metrics.mean_average_precision(
        rna_pp.obsm["X_pca"],
        rna_pp.obs["ct"].to_numpy().ravel()
    )
    assert 0 <= mean_average_precision <= 1


def test_seurat_alignment_score(atac_pp):
    seurat_alignment_score = scglue.metrics.seurat_alignment_score(
        atac_pp.obsm["X_lsi"],
        atac_pp.obs["ct"].to_numpy().ravel()
    )
    assert 0 <= seurat_alignment_score <= 1


def test_foscttm(rna_pp, atac_pp):
    foscttm_x, foscttm_y = scglue.metrics.foscttm(
        rna_pp.obsm["X_pca"], rna_pp.obsm["X_pca"]
    )
    assert np.all(foscttm_x == 0)
    assert np.all(foscttm_y == 0)
    foscttm_x, foscttm_y = scglue.metrics.foscttm(
        rna_pp.obsm["X_pca"][:20], atac_pp.obsm["X_lsi"][:20]
    )
    assert 0 < foscttm_x.mean() <= 1
    assert 0 < foscttm_y.mean() <= 1

    with pytest.raises(ValueError):
        foscttm_x, foscttm_y = scglue.metrics.foscttm(
            rna_pp.obsm["X_pca"], atac_pp.obsm["X_lsi"]
        )
