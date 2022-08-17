r"""
Tests for the :mod:`scglue.data` module
"""

# pylint: disable=redefined-outer-name, wildcard-import, unused-wildcard-import

import scglue

from .fixtures import *
from .utils import cmp_arrays


def test_lsi(atac):
    scglue.data.lsi(atac, n_components=2)
    assert atac.obsm["X_lsi"].shape == (atac.shape[0], 2)


def test_get_gene_annotation(rna, gtf_file):
    with pytest.raises(ValueError):
        scglue.data.get_gene_annotation(rna, gtf=gtf_file)
    scglue.data.get_gene_annotation(
        rna, gtf=gtf_file, gtf_by="gene_id",
        by_func=scglue.genomics.ens_trim_version
    )
    cmp_arrays(     rna.var["chrom"], ["chr1", "chr2", "chr2"])
    cmp_arrays(rna.var["chromStart"], [    10,      0,     10])
    cmp_arrays(  rna.var["chromEnd"], [    20,     10,     20])
    cmp_arrays(    rna.var["strand"], [   "+",    "-",    "+"])


def test_aggregate_obs(rna_pp):
    rna_agg = scglue.data.aggregate_obs(
        rna_pp, by="ct", X_agg="sum",
        obs_agg={"ct": "majority"},
        obsm_agg={"X_pca": "mean"}
    )
    assert rna_agg.shape[0] == 3


def test_transfer_labels(rna_pp):
    scglue.data.transfer_labels(
        rna_pp, rna_pp, "ct", n_neighbors=1,
        use_rep="X_pca", key_added="ct_transfer"
    )
    assert rna_pp.obs["ct_transfer"].equals(rna_pp.obs["ct"])


def test_extract_rank_genes_groups(rna_pp):
    with pytest.raises(ValueError):
        scglue.data.extract_rank_genes_groups(rna_pp)
    sc.tl.rank_genes_groups(rna_pp, "ct")
    _ = scglue.data.extract_rank_genes_groups(rna_pp)


def test_bedmap2anndata(bedmap_file):
    adata = scglue.data.bedmap2anndata(bedmap_file)
    adata = adata[
        ["cell1", "cell2", "cell3", "cell4", "cell5"],
        ["a", "b", "c", "d", "e", "f"]
    ]
    cmp_arrays(
        adata.X.toarray(),
        np.array([
            [1, 0, 0, 0, 0, 1],
            [1, 2, 0, 0, 0, 0],
            [1, 1, 0, 1, 0, 1],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 0]
        ])
    )


def test_estimate_balancing_weight(rna_pp, atac_pp):
    atac_pp.obsm["X_pca"] = atac_pp.obsm["X_lsi"]
    with pytest.raises(ValueError):
        scglue.data.estimate_balancing_weight(rna_pp, atac_pp)
    scglue.data.estimate_balancing_weight(
        rna_pp, atac_pp, use_rep="X_pca"
    )  # NOTE: Smoke test
    scglue.data.estimate_balancing_weight(
        rna_pp, atac_pp, use_rep="X_pca", use_batch="batch"
    )  # NOTE: Smoke test


def test_metacell_corr(rna_pp, atac_pp, guidance):
    atac_pp.obsm["X_pca"] = atac_pp.obsm["X_lsi"]
    with pytest.raises(ValueError):
        scglue.data.metacell_corr(rna_pp, atac_pp, use_rep="X_pca")
    with pytest.raises(ValueError):
        scglue.data.metacell_corr(rna_pp, atac_pp, n_meta=5)
    with pytest.raises(ValueError):
        scglue.data.metacell_corr(rna_pp, atac_pp, use_rep="X_pca", n_meta=5)
    with pytest.raises(ValueError):
        scglue.data.metacell_corr(
            rna_pp, rna_pp, use_rep="X_pca", n_meta=5,
            skeleton=guidance
        )
    scglue.data.metacell_corr(
        rna_pp, atac_pp,
        use_rep="X_pca", n_meta=5, skeleton=guidance, method="pcc"
    )  # NOTE: Smoke test
    scglue.data.metacell_corr(
        rna_pp, atac_pp,
        use_rep="X_pca", n_meta=5, skeleton=guidance
    )  # NOTE: Smoke test


def test_metacell_regr(rna_pp, atac_pp, guidance):
    atac_pp.obsm["X_pca"] = atac_pp.obsm["X_lsi"]
    guidance = guidance.edge_subgraph(
        e for e, attr in dict(guidance.edges).items()
        if attr["type"] == "rev"
    )
    with pytest.raises(ValueError):
        scglue.data.metacell_regr(rna_pp, atac_pp, use_rep="X_pca")
    with pytest.raises(ValueError):
        scglue.data.metacell_regr(rna_pp, atac_pp, n_meta=5)
    with pytest.raises(ValueError):
        scglue.data.metacell_regr(rna_pp, atac_pp, use_rep="X_pca", n_meta=5)
    scglue.data.metacell_regr(
        rna_pp, atac_pp,
        use_rep="X_pca", n_meta=5, skeleton=guidance, model="ElasticNet"
    )  # NOTE: Smoke test
    scglue.data.metacell_regr(
        rna_pp, atac_pp,
        use_rep="X_pca", n_meta=5, skeleton=guidance
    )  # NOTE: Smoke test
