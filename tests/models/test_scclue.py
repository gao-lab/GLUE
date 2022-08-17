r"""
Tests for the :mod:`scglue.models.scclue` module
"""

# pylint: disable=redefined-outer-name, wildcard-import, unused-wildcard-import

import warnings

import pytest

import scglue

from ..fixtures import *
from ..utils import cmp_arrays


@pytest.mark.parametrize("rna_prob", ["NB"])
@pytest.mark.parametrize("atac_prob", ["NB"])
@pytest.mark.parametrize("backed", [False, True])
def test_save_load(rna_pp, atac_pp, tmp_path, rna_prob, atac_prob, backed):

    ActiveModel = scglue.models.SCCLUEModel
    if backed:
        rna_pp.write_h5ad(tmp_path / "rna_pp.h5ad")
        atac_pp.write_h5ad(tmp_path / "atac_pp.h5ad")
        rna_pp = anndata.read_h5ad(tmp_path / "rna_pp.h5ad", backed="r")
        atac_pp = anndata.read_h5ad(tmp_path / "atac_pp.h5ad", backed="r")

    scglue.models.configure_dataset(rna_pp, rna_prob, use_highly_variable=True, use_cell_type="ct", use_dsc_weight="dsc_weight", use_obs_names=True)
    scglue.models.configure_dataset(atac_pp, atac_prob, use_highly_variable=True, use_batch="batch", use_dsc_weight="dsc_weight", use_obs_names=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError):
            glue = ActiveModel({}, latent_dim=2)
    glue = ActiveModel(
        {"rna": rna_pp, "atac": atac_pp},
        latent_dim=2, random_seed=0
    )
    glue.compile()
    with pytest.raises(ValueError):
        glue.fit({"rna": rna_pp, "atac": atac_pp}, max_epochs=None)
    glue.fit(
        {"rna": rna_pp, "atac": atac_pp},
        batch_size=32,
        align_burnin=2, max_epochs=5, patience=3,
        directory=tmp_path
    )
    print(glue)
    with pytest.raises(RuntimeError):
        glue.net()

    rna_pp.obsm["X_glue1"] = glue.encode_data("rna", rna_pp)
    atac_pp.obsm["X_glue1"] = glue.encode_data("atac", atac_pp)

    glue.save(tmp_path / "final.dill")
    glue_load = scglue.models.load_model(tmp_path / "final.dill")

    rna_pp.obsm["X_glue2"] = glue_load.encode_data("rna", rna_pp)
    atac_pp.obsm["X_glue2"] = glue_load.encode_data("atac", atac_pp)

    cmp_arrays(rna_pp.obsm["X_glue1"], rna_pp.obsm["X_glue2"])
    cmp_arrays(atac_pp.obsm["X_glue1"], atac_pp.obsm["X_glue2"])

    del glue, glue_load


@pytest.mark.parametrize("rna_prob", ["NB"])
@pytest.mark.parametrize("atac_prob", ["NB"])
def test_adopt(rna_pp, atac_pp, tmp_path, rna_prob, atac_prob):

    ActiveModel = scglue.models.SCCLUEModel

    scglue.models.configure_dataset(rna_pp, rna_prob, use_highly_variable=True, use_batch="batch", use_obs_names=True)
    scglue.models.configure_dataset(atac_pp, atac_prob, use_highly_variable=True, use_batch="batch", use_obs_names=True)

    glue = ActiveModel(
        {"rna": rna_pp, "atac": atac_pp},
        latent_dim=2, shared_batches=True, random_seed=0
    )
    glue.compile()
    glue.fit(
        {"rna": rna_pp, "atac": atac_pp},
        batch_size=32,
        align_burnin=2, max_epochs=5, patience=3,
        directory=tmp_path
    )

    rna_pp.obsm["X_glue1"] = glue.encode_data("rna", rna_pp)
    atac_pp.obsm["X_glue1"] = glue.encode_data("atac", atac_pp)

    rna_pp.obs["batch"] = rna_pp.obs["batch"].astype(str)
    atac_pp.obs["batch"] = atac_pp.obs["batch"].astype(str)
    rna_pp.obs.loc[rna_pp.obs_names[-1], "batch"] = "b3"
    atac_pp.obs.loc[atac_pp.obs_names[-1], "batch"] = "b3"

    scglue.models.configure_dataset(rna_pp, rna_prob, use_highly_variable=True, use_batch="batch", use_obs_names=True)
    scglue.models.configure_dataset(atac_pp, atac_prob, use_highly_variable=True, use_batch="batch", use_obs_names=True)

    glue_new = ActiveModel(
        {"rna": rna_pp, "atac": atac_pp},
        latent_dim=2, shared_batches=True, random_seed=0
    )
    glue_new.adopt_pretrained_model(glue)

    rna_pp.obsm["X_glue2"] = glue_new.encode_data("rna", rna_pp)
    atac_pp.obsm["X_glue2"] = glue_new.encode_data("atac", atac_pp)

    cmp_arrays(rna_pp.obsm["X_glue1"], rna_pp.obsm["X_glue2"])
    cmp_arrays(atac_pp.obsm["X_glue1"], atac_pp.obsm["X_glue2"])

    del glue, glue_new


@pytest.mark.parametrize("rna_prob", ["NB"])
@pytest.mark.parametrize("atac_prob", ["NB"])
def test_repeatability(rna_pp, atac_pp, tmp_path, rna_prob, atac_prob):

    ActiveModel = scglue.models.SCCLUEModel

    scglue.models.configure_dataset(rna_pp, rna_prob, use_highly_variable=True, use_rep="X_pca", use_batch="batch", use_obs_names=True)
    scglue.models.configure_dataset(atac_pp, atac_prob, use_highly_variable=True, use_cell_type="ct", use_obs_names=True)

    for i in range(2):
        glue = ActiveModel(
            {"rna": rna_pp, "atac": atac_pp},
            latent_dim=2, random_seed=0
        )
        glue.compile()
        glue.fit(
            {"rna": rna_pp, "atac": atac_pp},
            batch_size=40,
            align_burnin=2, max_epochs=5, patience=3,
            directory=tmp_path
        )

        rna_pp.obsm[f"X_glue{i + 1}"] = glue.encode_data("rna", rna_pp)
        atac_pp.obsm[f"X_glue{i + 1}"] = glue.encode_data("atac", atac_pp)

        del glue

    cmp_arrays(rna_pp.obsm["X_glue1"], rna_pp.obsm["X_glue2"])
    cmp_arrays(atac_pp.obsm["X_glue1"], atac_pp.obsm["X_glue2"])
