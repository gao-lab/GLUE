r"""
pytest fixtures
"""

# pylint: disable=missing-function-docstring, redefined-outer-name

import anndata
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse

import scglue


@pytest.fixture
def bed_file(tmp_path):
    file = tmp_path / "test.bed"
    with file.open("w") as f:
        f.write("""
chr1\t0\t10\ta\t.\t+
chr1\t20\t30\tb\t.\t+
chr1\t30\t40\tc\t.\t-
chr2\t0\t10\td\t.\t+
chr2\t10\t20\te\t.\t-
chr3\t0\t10\tf\t.\t+
        """.strip(" \n"))
    yield file
    file.unlink()


@pytest.fixture
def gtf_file(tmp_path):
    file = tmp_path / "test.gtf"
    with file.open("w") as f:
        f.write("""
chr1\tHAVANA\tgene\t11\t20\t.\t+\t.\tgene_id "A.1"; source "havana";
chr2\tHAVANA\tgene\t1\t10\t.\t-\t.\tgene_id "B.2"; source "havana";
chr2\tHAVANA\tgene\t11\t20\t.\t+\t.\tgene_id "C.1"; source "havana";
        """.strip(" \n"))
    yield file
    file.unlink()


@pytest.fixture
def fasta_file(tmp_path):
    file = tmp_path / "test.fasta"
    with file.open("w") as f:
        f.write("""
>chr1
AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG
>chr2
AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG
>chr3
AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG
        """.strip(" \n"))
    yield file
    file.unlink()


@pytest.fixture
def fai_file(tmp_path):
    file = tmp_path / "test.fasta.fai"
    with file.open("w") as f:
        f.write("""
chr1\t40\t6\t40\t41
chr2\t40\t53\t40\t41
chr3\t40\t100\t40\t41
        """.strip(" \n"))
    yield file
    file.unlink()


@pytest.fixture
def bedmap_file(tmp_path):
    file = tmp_path / "test.bedmap"
    with open(file, "w") as f:
        f.write("""
chr1\t0\t10\ta\t.\t+\tcell1;cell2;cell3
chr1\t20\t30\tb\t.\t+\tcell3;cell2;cell2
chr1\t30\t40\tc\t.\t-\tcell4
chr2\t0\t10\td\t.\t+\tcell5;cell3
chr2\t10\t20\te\t.\t-\tcell4
chr3\t0\t10\tf\t.\t+\tcell1;cell3
        """.strip(" \n"))
    yield file
    file.unlink()


@pytest.fixture
def graph_data():
    return [
        ("a", "a", dict(dist= 0, weight=   1.0)),
        ("a", "b", dict(dist=11, weight=1 / 11)),
        ("b", "a", dict(dist=11, weight=1 / 11)),
        ("b", "b", dict(dist= 0, weight=   1.0)),
        ("b", "c", dict(dist= 1, weight=   1.0)),
        ("c", "b", dict(dist= 1, weight=   1.0)),
        ("c", "c", dict(dist= 0, weight=   1.0)),
        ("d", "d", dict(dist= 0, weight=   1.0)),
        ("d", "e", dict(dist= 1, weight=   1.0)),
        ("e", "d", dict(dist= 1, weight=   1.0)),
        ("e", "e", dict(dist= 0, weight=   1.0)),
        ("f", "f", dict(dist= 0, weight=   1.0))
    ]


@pytest.fixture
def graph(graph_data):
    return nx.MultiDiGraph(graph_data)


@pytest.fixture
def composed_graph(graph_data):
    return nx.MultiDiGraph(graph_data + [
        ("f", "g", dict(dist=5, weight=1 / 5)),
        ("g", "f", dict(dist=5, weight=1 / 5)),
        ("a", "b", dict(dist=2, weight=1 / 2)),
        ("b", "a", dict(dist=2, weight=1 / 2))
    ])


@pytest.fixture
def mat():
    return np.random.randint(0, 50, size=(150, 3))


@pytest.fixture
def spmat():
    return scipy.sparse.csr_matrix(np.random.randint(0, 20, size=(120, 6)))


@pytest.fixture
def rna(mat):
    X = mat
    obs = pd.DataFrame({
        "ct": pd.Categorical(np.random.choice(["ct1", "ct2", "ct3"], X.shape[0], replace=True)),
        "batch": pd.Categorical(np.random.choice(["b1", "b2"], X.shape[0], replace=True)),
        "dsc_weight": np.random.rand(X.shape[0])
    }, index=[f"SHARED-{i}" for i in range(100)] + [f"RNA-{i}" for i in range(50)])
    var = pd.DataFrame(index=["A", "B", "C"])
    arange = np.expand_dims(np.arange(mat.shape[0]), 1).repeat(mat.shape[1], axis=1)
    obs.index.name, var.index.name = "cells", "genes"
    return anndata.AnnData(X=X, obs=obs, var=var, layers={"arange": arange}, dtype=X.dtype)


@pytest.fixture
def atac(spmat, bed_file):
    X = spmat
    obs = pd.DataFrame({
        "ct": pd.Categorical(np.random.choice(["ct1", "ct2", "ct4"], X.shape[0], replace=True)),
        "batch": pd.Categorical(np.random.choice(["b1", "b2"], X.shape[0], replace=True)),
        "dsc_weight": np.random.rand(X.shape[0])
    }, index=[f"SHARED-{i}" for i in range(100)] + [f"ATAC-{i}" for i in range(20)])
    var = pd.read_csv(
        bed_file, sep="\t", header=None, comment="#",
        names=["chrom", "chromStart", "chromEnd", "name", "score", "strand"]
    ).set_index("name", drop=False)
    arange = np.expand_dims(np.arange(spmat.shape[0]), 1).repeat(spmat.shape[1], axis=1)
    obs.index.name, var.index.name = "cells", "peaks"
    return anndata.AnnData(X=X, obs=obs, var=var, layers={"arange": arange}, dtype=np.float32)


@pytest.fixture
def rna_pp(rna, gtf_file):
    scglue.data.get_gene_annotation(
        rna, gtf=gtf_file, gtf_by="gene_id",
        by_func=scglue.genomics.ens_trim_version
    )
    rna.var["highly_variable"] = [True, False, True]
    rna.raw = rna
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    sc.pp.scale(rna, max_value=10)
    sc.tl.pca(rna, n_comps=2, use_highly_variable=True, svd_solver="auto")
    rna.X = rna.raw.X
    del rna.raw
    return rna


@pytest.fixture
def atac_pp(atac):
    atac.var["highly_variable"] = True
    scglue.data.lsi(atac, n_components=2, use_highly_variable=False)
    return atac


@pytest.fixture
def guidance(rna_pp, atac_pp):
    return scglue.genomics.rna_anchored_guidance_graph(
        rna_pp, atac_pp, promoter_len=2, extend_range=15,
        propagate_highly_variable=False,
        extend_fn=lambda x: 1 / x if x > 1 else 1.0
    )


@pytest.fixture
def eidx():
    return np.array([[0, 1, 2, 1], [0, 1, 2, 2]])


@pytest.fixture
def ewt():
    return np.array([1.0, 0.4, 0.7, 0.1])


@pytest.fixture
def esgn():
    return np.array([1.0, 1.0, 1.0, 1.0])
