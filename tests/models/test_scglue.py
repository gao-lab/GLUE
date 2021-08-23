r"""
Tests for the :mod:`scglue.models.scglue` module
"""

# pylint: disable=redefined-outer-name, wildcard-import, unused-wildcard-import

import warnings

import pytest

import scglue

from ..fixtures import *
from ..utils import cmp_arrays


@pytest.mark.parametrize("rna_prob", ["NB"])
@pytest.mark.parametrize("atac_prob", ["NB", "ZINB"])
def test_save_load(rna_pp, atac_pp, prior, tmp_path, rna_prob, atac_prob):

    scglue.models.configure_dataset(rna_pp, rna_prob, use_highly_variable=True)
    scglue.models.configure_dataset(atac_pp, atac_prob, use_highly_variable=True)
    vertices = sorted(prior.nodes)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError):
            glue = scglue.models.SCGLUEModel({}, vertices, latent_dim=2)
    glue = scglue.models.SCGLUEModel(
        {"rna": rna_pp, "atac": atac_pp}, vertices,
        latent_dim=2, random_seed=0
    )
    with pytest.raises(ValueError):
        glue.compile(lam_graph=None)
    glue.compile()
    with pytest.raises(ValueError):
        glue.fit({"rna": rna_pp, "atac": atac_pp}, prior, max_epochs=None)
    glue.fit(
        {"rna": rna_pp, "atac": atac_pp}, prior,
        data_batch_size=8, graph_batch_size=128,
        align_burnin=2, max_epochs=5, patience=3,
        directory=tmp_path
    )
    print(glue)
    with pytest.raises(RuntimeError):
        glue.net()

    rna_pp.obsm["X_glue1"] = glue.encode_data("rna", rna_pp)
    atac_pp.obsm["X_glue1"] = glue.encode_data("atac", atac_pp)
    graph_embedding1 = glue.encode_graph(prior)

    glue.save(tmp_path / "final.dill")
    glue_load = scglue.models.load_model(tmp_path / "final.dill")

    rna_pp.obsm["X_glue2"] = glue_load.encode_data("rna", rna_pp)
    atac_pp.obsm["X_glue2"] = glue_load.encode_data("atac", atac_pp)
    graph_embedding2 = glue_load.encode_graph(prior)

    cmp_arrays(rna_pp.obsm["X_glue1"], rna_pp.obsm["X_glue2"])
    cmp_arrays(atac_pp.obsm["X_glue1"], atac_pp.obsm["X_glue2"])
    cmp_arrays(graph_embedding1, graph_embedding2)

    del glue, glue_load


@pytest.mark.parametrize("rna_prob", ["Normal"])
@pytest.mark.parametrize("atac_prob", ["ZIN", "ZILN"])
def test_adopt_freeze(rna_pp, atac_pp, prior, tmp_path, rna_prob, atac_prob):

    scglue.models.configure_dataset(rna_pp, rna_prob, use_highly_variable=True)
    scglue.models.configure_dataset(atac_pp, atac_prob, use_highly_variable=True, use_rep="X_lsi")
    vertices = sorted(prior.nodes)

    glue = scglue.models.SCGLUEModel(
        {"rna": rna_pp, "atac": atac_pp}, vertices,
        latent_dim=2, random_seed=0
    )
    glue.compile()
    glue.fit(
        {"rna": rna_pp, "atac": atac_pp}, prior,
        data_batch_size=8, graph_batch_size=128,
        align_burnin=2, max_epochs=5, patience=3,
        directory=tmp_path
    )

    rna_pp.obsm["X_glue1"] = glue.encode_data("rna", rna_pp)
    atac_pp.obsm["X_glue1"] = glue.encode_data("atac", atac_pp)

    glue_freeze = scglue.models.SCGLUEModel(
        {"rna": rna_pp, "atac": atac_pp}, vertices,
        latent_dim=2, random_seed=0
    )
    glue_freeze.adopt_pretrained_model(glue, submodule="x2u")
    glue_freeze.compile()
    glue_freeze.freeze_cells()
    glue_freeze.fit(
        {"rna": rna_pp, "atac": atac_pp}, prior,
        data_batch_size=8, graph_batch_size=128,
        align_burnin=2, max_epochs=5, patience=3,
        directory=tmp_path
    )
    glue_freeze.unfreeze_cells()
    print(glue_freeze)

    rna_pp.obsm["X_glue2"] = glue_freeze.encode_data("rna", rna_pp)
    atac_pp.obsm["X_glue2"] = glue_freeze.encode_data("atac", atac_pp)

    cmp_arrays(rna_pp.obsm["X_glue1"], rna_pp.obsm["X_glue2"])
    cmp_arrays(atac_pp.obsm["X_glue1"], atac_pp.obsm["X_glue2"])

    glue_freeze = scglue.models.SCGLUEModel(
        {"rna": rna_pp, "atac": atac_pp}, vertices,
        latent_dim=4, h_depth=3, random_seed=0
    )
    glue_freeze.adopt_pretrained_model(glue, submodule="x2u")
    del glue, glue_freeze


@pytest.mark.cpu_only
@pytest.mark.parametrize("rna_prob", ["NB"])
@pytest.mark.parametrize("atac_prob", ["NB"])
def test_repeatability(rna_pp, atac_pp, prior, tmp_path, rna_prob, atac_prob):

    scglue.models.configure_dataset(rna_pp, rna_prob, use_highly_variable=True, use_rep="X_pca")
    scglue.models.configure_dataset(atac_pp, atac_prob, use_highly_variable=True)
    vertices = sorted(prior.nodes)
    graph_embedding = {}

    for i in range(2):
        glue = scglue.models.SCGLUEModel(
            {"rna": rna_pp, "atac": atac_pp}, vertices,
            latent_dim=2, random_seed=0
        )
        glue.compile()
        glue.fit(
            {"rna": rna_pp, "atac": atac_pp}, prior,
            data_batch_size=8, graph_batch_size=128,
            align_burnin=2, max_epochs=5, patience=3,
            directory=tmp_path
        )

        rna_pp.obsm[f"X_glue{i + 1}"] = glue.encode_data("rna", rna_pp)
        atac_pp.obsm[f"X_glue{i + 1}"] = glue.encode_data("atac", atac_pp)
        graph_embedding[i + 1] = glue.encode_graph(prior)

        del glue

    cmp_arrays(rna_pp.obsm["X_glue1"], rna_pp.obsm["X_glue2"])
    cmp_arrays(atac_pp.obsm["X_glue1"], atac_pp.obsm["X_glue2"])
    cmp_arrays(graph_embedding[1], graph_embedding[2])


def test_abnormal(rna_pp, atac_pp, prior):
    vertices = sorted(prior.nodes)
    with pytest.raises(ValueError):
        glue = scglue.models.SCGLUEModel(
            {"rna": rna_pp, "atac": atac_pp}, vertices,
            latent_dim=2, random_seed=0
        )
    with pytest.raises(ValueError):
        scglue.models.configure_dataset(rna_pp, "NB", use_highly_variable=True, use_layer="xxx")
    with pytest.raises(ValueError):
        scglue.models.configure_dataset(atac_pp, "NB", use_highly_variable=True, use_rep="yyy")
    scglue.models.configure_dataset(rna_pp, "NB", use_highly_variable=False, use_rep="X_pca")
    scglue.models.configure_dataset(atac_pp, "zzz", use_highly_variable=False, use_rep="X_lsi")
    with pytest.raises(ValueError):
        glue = scglue.models.SCGLUEModel(
            {"rna": rna_pp, "atac": atac_pp}, vertices,
            latent_dim=2, random_seed=0
        )
    scglue.models.configure_dataset(atac_pp, "NB", use_highly_variable=False, use_rep="X_lsi")
    glue = scglue.models.SCGLUEModel(
        {"rna": rna_pp, "atac": atac_pp}, vertices,
        latent_dim=100, random_seed=0
    )
    with pytest.raises(ValueError):
        glue.encode_data("rna", atac_pp)
    with pytest.raises(ValueError):
        glue.encode_data("atac", rna_pp)
    del glue
