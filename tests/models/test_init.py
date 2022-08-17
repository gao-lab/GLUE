r"""
Tests for the :mod:`scglue.models.__init__` module
"""

# pylint: disable=redefined-outer-name, wildcard-import, unused-wildcard-import

import scglue

from ..fixtures import *


def test_configure_dataset(rna):
    with pytest.raises(ValueError):
        scglue.models.configure_dataset(rna, "NB", use_highly_variable=True)
    with pytest.raises(ValueError):
        scglue.models.configure_dataset(rna, "NB", use_highly_variable=False, use_layer="xxx")
    with pytest.raises(ValueError):
        scglue.models.configure_dataset(rna, "NB", use_highly_variable=False, use_rep="xxx")
    with pytest.raises(ValueError):
        scglue.models.configure_dataset(rna, "NB", use_highly_variable=False, use_dsc_weight="xxx")
    with pytest.raises(ValueError):
        scglue.models.configure_dataset(rna, "NB", use_highly_variable=False, use_cell_type="xxx")
    scglue.models.configure_dataset(rna, "NB", use_highly_variable=False)


@pytest.mark.parametrize("use_df", [False, True])
def test_fit_SCGLUE(rna_pp, atac_pp, guidance, use_df):
    if use_df:
        rna_pp.obsm["X_pca"] = pd.DataFrame(rna_pp.obsm["X_pca"], index=rna_pp.obs_names)
    scglue.models.configure_dataset(rna_pp, "NB", use_highly_variable=True, use_rep="X_pca", use_batch="batch", use_obs_names=True)
    scglue.models.configure_dataset(atac_pp, "NB", use_highly_variable=True, use_cell_type="ct", use_batch="batch", use_obs_names=True)
    scglue.models.fit_SCGLUE(
        {"rna": rna_pp, "atac": atac_pp}, guidance,
        init_kws={"latent_dim": 2, "shared_batches": True},
        compile_kws={"lr": 1e-5},
        fit_kws={"max_epochs": 5}
    )  # NOTE: Smoke test
