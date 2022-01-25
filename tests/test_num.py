r"""
Tests for the :mod:`scglue.num` module
"""

# pylint: disable=redefined-outer-name, wildcard-import, unused-wildcard-import

import numpy as np
import pytest

import scglue

from .fixtures import *


def test_sigmoid():
    assert scglue.num.sigmoid(0) == 0.5
    assert 0 < scglue.num.sigmoid(-1) < 0.5 < scglue.num.sigmoid(1) < 1


def test_col_var(mat, spmat):
    _ = scglue.num.col_var(mat)
    _ = scglue.num.col_var(spmat)
    _ = scglue.num.col_var(mat[:10, :2], spmat[:10, :][:, :2])
    with pytest.raises(ValueError):
        _ = scglue.num.col_var(mat, spmat)


def test_col_pcc(mat, spmat):
    pcc = scglue.num.col_pcc(mat, mat)
    assert np.allclose(pcc, 1)
    pcc = scglue.num.col_pcc(spmat, spmat)
    assert np.allclose(pcc, 1)


def test_col_spr(mat, spmat):
    spr = scglue.num.col_spr(mat, mat)
    assert np.allclose(spr, 1)
    spr = scglue.num.col_spr(spmat, spmat)
    assert np.allclose(spr, 1)


def test_cov_mat(mat, spmat):
    _ = scglue.num.cov_mat(mat)
    _ = scglue.num.cov_mat(spmat)
    _ = scglue.num.cov_mat(mat[:10], spmat[:10])
    with pytest.raises(ValueError):
        _ = scglue.num.cov_mat(mat, spmat)


def test_pcc_mat(mat, spmat):
    pcc = scglue.num.pcc_mat(mat)
    pcc = scglue.num.pcc_mat(mat, mat)
    assert np.allclose(np.diag(pcc), 1)
    assert not np.allclose(pcc, 1)
    pcc = scglue.num.pcc_mat(spmat)
    pcc = scglue.num.pcc_mat(spmat, spmat)
    assert np.allclose(np.diag(pcc), 1)
    assert not np.allclose(pcc, 1)


def test_spr_mat(mat, spmat):
    spr = scglue.num.spr_mat(mat)
    spr = scglue.num.spr_mat(mat, mat)
    assert np.allclose(np.diag(spr), 1)
    assert not np.allclose(spr, 1)
    spr = scglue.num.spr_mat(spmat)
    spr = scglue.num.spr_mat(spmat, spmat)
    assert np.allclose(np.diag(spr), 1)
    assert not np.allclose(spr, 1)


def test_tfidf(spmat):
    assert np.allclose(
        scglue.num.tfidf(spmat).toarray(),
        scglue.num.tfidf(spmat.toarray())
    )


def test_vertex_degrees(eidx, ewt):
    degrees = scglue.num.vertex_degrees(eidx, ewt, direction="in")
    assert np.allclose(degrees, np.array([1.0, 0.4, 0.8]))
    degrees = scglue.num.vertex_degrees(eidx, ewt, direction="out")
    assert np.allclose(degrees, np.array([1.0, 0.5, 0.7]))
    degrees = scglue.num.vertex_degrees(eidx, ewt, direction="both")
    assert np.allclose(degrees, np.array([1.0, 0.5, 0.8]))


def test_normalize_edges(eidx, ewt):
    enorm = scglue.num.normalize_edges(eidx, ewt, method="in")
    assert np.allclose(enorm, np.array([
        1.0 / 1.0, 0.4 / 0.4,
        0.7 / 0.8, 0.1 / 0.8
    ]))
    enorm = scglue.num.normalize_edges(eidx, ewt, method="out")
    assert np.allclose(enorm, np.array([
        1.0 / 1.0, 0.4 / 0.5,
        0.7 / 0.7, 0.1 / 0.5
    ]))
    enorm = scglue.num.normalize_edges(eidx, ewt, method="sym")
    assert np.allclose(enorm, np.array([
        1.0 / np.sqrt(1.0 * 1.0), 0.4 / np.sqrt(0.4 * 0.5),
        0.7 / np.sqrt(0.8 * 0.7), 0.1 / np.sqrt(0.8 * 0.5)
    ]))
    enorm = scglue.num.normalize_edges(eidx, ewt, method="keepvar")
    assert np.allclose(enorm, np.array([
        1.0 / np.sqrt(1.0), 0.4 / np.sqrt(0.4),
        0.7 / np.sqrt(0.8), 0.1 / np.sqrt(0.8)
    ]))
