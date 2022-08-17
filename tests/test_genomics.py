
r"""
Tests for the :mod:`scglue.genomics` module
"""

# pylint: disable=redefined-outer-name, wildcard-import, unused-wildcard-import

import pytest

import scglue

from .fixtures import *
from .utils import cmp_graphs


def test_bed(bed_file, fasta_file, fai_file):
    bed = scglue.genomics.Bed.read_bed(bed_file)
    bed.write_bed(bed_file)
    assert bed.equals(scglue.genomics.Bed.read_bed(bed_file))
    assert bed.equals(bed.expand(0, 0))
    assert bed.equals(bed.strand_specific_start_site().expand(0, 9))
    assert bed.equals(bed.strand_specific_end_site().expand(9, 0))
    assert (bed.expand(
        1000, 1000, chr_len=scglue.genomics.get_chr_len_from_fai(fai_file)
    ).df["chromEnd"] == 40).all()
    assert bed.nucleotide_content(fasta_file).shape[0] == bed.shape[0]

    bed["strand"] = "."
    bed.equals(bed.expand(2, 2).expand(-2, -2))
    with pytest.raises(ValueError):
        bed.equals(bed.expand(0, 2))
    with pytest.raises(ValueError):
        assert bed.equals(bed.strand_specific_start_site())
    with pytest.raises(ValueError):
        assert bed.equals(bed.strand_specific_end_site())
    with pytest.raises(ValueError):
        _ = bed.iloc[:, :2]
    with pytest.raises(ValueError):
        bed.write_bed(bed_file, ncols=2)
    bed = bed.df
    columns = bed.columns.to_numpy()
    columns[0] = "xxx"
    bed.columns = columns
    with pytest.raises(ValueError):
        scglue.genomics.Bed.verify(bed)


def test_gtf(gtf_file):
    gtf = scglue.genomics.Gtf.read_gtf(gtf_file)
    _ = scglue.genomics.Gtf(gtf.iloc[:, :5])
    _ = gtf.split_attribute()
    _ = gtf.to_bed()

    with pytest.raises(ValueError):
        _ = gtf.iloc[:, :2]
    gtf = gtf.df
    columns = gtf.columns.to_numpy()
    columns[0] = "xxx"
    gtf.columns = columns
    with pytest.raises(ValueError):
        scglue.genomics.Gtf.verify(gtf)


def test_window_graph(bed_file, graph):
    bed = scglue.genomics.Bed.read_bed(bed_file)
    result = scglue.genomics.window_graph(bed_file, bed, window_size=15)
    result = scglue.genomics.window_graph(
        bed, bed_file, window_size=15, attr_fn=lambda l, r, d: {
            "dist": abs(d), "weight": 1 / abs(d) if d != 0 else 1.0
        }
    )
    cmp_graphs(result, graph)


def test_rna_anchored_guidance_graph(rna, atac, gtf_file):  # NOTE: Smoke test
    scglue.data.get_gene_annotation(
        rna, gtf=gtf_file, gtf_by="gene_id",
        by_func=scglue.genomics.ens_trim_version
    )
    rna.var["highly_variable"] = True
    scglue.genomics.rna_anchored_guidance_graph(
        rna, atac, gene_region="combined",
        promoter_len=2, extend_range=15,
        extend_fn=lambda x: 1 / x if x > 0 else 1.0
    )
    scglue.genomics.rna_anchored_guidance_graph(
        rna, atac, gene_region="promoter",
        promoter_len=2, extend_range=15,
        propagate_highly_variable=True, corrupt_rate=0.2
    )
    with pytest.raises(ValueError):
        scglue.genomics.rna_anchored_guidance_graph(rna, atac, gene_region="xxx")


def test_regulatory_inference(rna):
    skeleton = nx.Graph([(i, j) for i in rna.var_names for j in rna.var_names])
    with pytest.raises(ValueError):
        _ = scglue.genomics.regulatory_inference(
            rna.var_names, rna.X.T, skeleton,
            alternative="xxx", random_state=0
        )
    with pytest.raises(ValueError):
        _ = scglue.genomics.regulatory_inference(
            rna.var_names, rna.X, skeleton,
            alternative="two.sided", random_state=0
        )
    with pytest.raises(ValueError):
        _ = scglue.genomics.regulatory_inference(
            rna.var_names, [rna.X.T, rna.X], skeleton,
            alternative="two.sided", random_state=0
        )
    _ = scglue.genomics.regulatory_inference(
        rna.var_names, rna.X.T, skeleton,
        alternative="two.sided", random_state=0
    )  # NOTE: Smoke test
    _ = scglue.genomics.regulatory_inference(
        rna.var_names, [rna.X.T, rna.X.T], skeleton,
        alternative="greater", random_state=1
    )  # NOTE: Smoke test
    _ = scglue.genomics.regulatory_inference(
        rna.var_names, [rna.X.T], skeleton,
        alternative="less", random_state=2
    )  # NOTE: Smoke test


def test_write_links(rna_pp, tmp_path):
    skeleton = nx.DiGraph([(i, j) for i in rna_pp.var_names for j in rna_pp.var_names])
    reginf = scglue.genomics.regulatory_inference(
        rna_pp.var_names, rna_pp.X.T, skeleton,
        alternative="two.sided", random_state=0
    )
    links_file = tmp_path / "test.links"
    scglue.genomics.write_links(
        reginf,
        scglue.genomics.Bed(rna_pp.var.assign(name=rna_pp.var_names)),
        scglue.genomics.Bed(rna_pp.var.assign(name=rna_pp.var_names)),
        links_file, keep_attrs=["score"]
    )  # NOTE: Smoke test
    links_file.unlink()


def test_cis_regulatory_ranking(rna, atac, guidance):
    _ = scglue.genomics.cis_regulatory_ranking(
        guidance, guidance.reverse(),
        rna.var_names, atac.var_names, rna.var_names
    )  # NOTE: Smoke test
