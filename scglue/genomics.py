r"""
Genomics operations
"""

import collections
import os
import re
from ast import literal_eval
from functools import reduce
from itertools import chain, product
from operator import add
from typing import Any, Callable, List, Mapping, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd
import pybedtools
import scipy.sparse
import scipy.stats
from anndata import AnnData
from networkx.algorithms.bipartite import biadjacency_matrix
from pybedtools import BedTool
from pybedtools.cbedtools import Interval
from statsmodels.stats.multitest import fdrcorrection
from tqdm.auto import tqdm

from .check import check_deps
from .graph import compose_multigraph, reachable_vertices
from .typehint import RandomState
from .utils import ConstrainedDataFrame, logged, get_rs


class Bed(ConstrainedDataFrame):

    r"""
    BED format data frame
    """

    COLUMNS = pd.Index([
        "chrom", "chromStart", "chromEnd", "name", "score",
        "strand", "thickStart", "thickEnd", "itemRgb",
        "blockCount", "blockSizes", "blockStarts"
    ])

    @classmethod
    def rectify(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = super(Bed, cls).rectify(df)
        COLUMNS = cls.COLUMNS.copy(deep=True)
        for item in COLUMNS:
            if item in df:
                if item in ("chromStart", "chromEnd"):
                    df[item] = df[item].astype(int)
                else:
                    df[item] = df[item].astype(str)
            elif item not in ("chrom", "chromStart", "chromEnd"):
                df[item] = "."
            else:
                raise ValueError(f"Required column {item} is missing!")
        return df.loc[:, COLUMNS]

    @classmethod
    def verify(cls, df: pd.DataFrame) -> None:
        super(Bed, cls).verify(df)
        if len(df.columns) != len(cls.COLUMNS) or np.any(df.columns != cls.COLUMNS):
            raise ValueError("Invalid BED format!")

    @classmethod
    def read_bed(cls, fname: os.PathLike) -> "Bed":
        r"""
        Read BED file

        Parameters
        ----------
        fname
            BED file

        Returns
        -------
        bed
            Loaded :class:`Bed` object
        """
        COLUMNS = cls.COLUMNS.copy(deep=True)
        loaded = pd.read_csv(fname, sep="\t", header=None, comment="#")
        loaded.columns = COLUMNS[:loaded.shape[1]]
        return cls(loaded)

    def write_bed(self, fname: os.PathLike, ncols: Optional[int] = None) -> None:
        r"""
        Write BED file

        Parameters
        ----------
        fname
            BED file
        ncols
            Number of columns to write (by default write all columns)
        """
        if ncols and ncols < 3:
            raise ValueError("`ncols` must be larger than 3!")
        df = self.df.iloc[:, :ncols] if ncols else self
        df.to_csv(fname, sep="\t", header=False, index=False)

    def to_bedtool(self) -> pybedtools.BedTool:
        r"""
        Convert to a :class:`pybedtools.BedTool` object

        Returns
        -------
        bedtool
            Converted :class:`pybedtools.BedTool` object
        """
        return BedTool(Interval(
            row["chrom"], row["chromStart"], row["chromEnd"],
            name=row["name"], score=row["score"], strand=row["strand"]
        ) for _, row in self.iterrows())

    def nucleotide_content(self, fasta: os.PathLike) -> pd.DataFrame:
        r"""
        Compute nucleotide content in the BED regions

        Parameters
        ----------
        fasta
            Genomic sequence file in FASTA format

        Returns
        -------
        nucleotide_stat
            Data frame containing nucleotide content statistics for each region
        """
        result = self.to_bedtool().nucleotide_content(fi=os.fspath(fasta), s=True)  # pylint: disable=unexpected-keyword-arg
        result = pd.DataFrame(
            np.stack([interval.fields[6:15] for interval in result]),
            columns=[
                r"%AT", r"%GC",
                r"#A", r"#C", r"#G", r"#T", r"#N",
                r"#other", r"length"
            ]
        ).astype({
            r"%AT": float, r"%GC": float,
            r"#A": int, r"#C": int, r"#G": int, r"#T": int, r"#N": int,
            r"#other": int, r"length": int
        })
        pybedtools.cleanup()
        return result

    def strand_specific_start_site(self) -> "Bed":
        r"""
        Convert to strand-specific start sites of genomic features

        Returns
        -------
        start_site_bed
            A new :class:`Bed` object, containing strand-specific start sites
            of the current :class:`Bed` object
        """
        if set(self["strand"]) != set(["+", "-"]):
            raise ValueError("Not all features are strand specific!")
        df = pd.DataFrame(self, copy=True)
        pos_strand = df.query("strand == '+'").index
        neg_strand = df.query("strand == '-'").index
        df.loc[pos_strand, "chromEnd"] = df.loc[pos_strand, "chromStart"] + 1
        df.loc[neg_strand, "chromStart"] = df.loc[neg_strand, "chromEnd"] - 1
        return type(self)(df)

    def strand_specific_end_site(self) -> "Bed":
        r"""
        Convert to strand-specific end sites of genomic features

        Returns
        -------
        end_site_bed
            A new :class:`Bed` object, containing strand-specific end sites
            of the current :class:`Bed` object
        """
        if set(self["strand"]) != set(["+", "-"]):
            raise ValueError("Not all features are strand specific!")
        df = pd.DataFrame(self, copy=True)
        pos_strand = df.query("strand == '+'").index
        neg_strand = df.query("strand == '-'").index
        df.loc[pos_strand, "chromStart"] = df.loc[pos_strand, "chromEnd"] - 1
        df.loc[neg_strand, "chromEnd"] = df.loc[neg_strand, "chromStart"] + 1
        return type(self)(df)

    def expand(
            self, upstream: int, downstream: int,
            chr_len: Optional[Mapping[str, int]] = None
    ) -> "Bed":
        r"""
        Expand genomic features towards upstream and downstream

        Parameters
        ----------
        upstream
            Number of bps to expand in the upstream direction
        downstream
            Number of bps to expand in the downstream direction
        chr_len
            Length of each chromosome

        Returns
        -------
        expanded_bed
            A new :class:`Bed` object, containing expanded features
            of the current :class:`Bed` object

        Note
        ----
        Starting position < 0 after expansion is always trimmed.
        Ending position exceeding chromosome length is trimed only if
        ``chr_len`` is specified.
        """
        if upstream == downstream == 0:
            return self
        df = pd.DataFrame(self, copy=True)
        if upstream == downstream:  # symmetric
            df["chromStart"] -= upstream
            df["chromEnd"] += downstream
        else:  # asymmetric
            if set(df["strand"]) != set(["+", "-"]):
                raise ValueError("Not all features are strand specific!")
            pos_strand = df.query("strand == '+'").index
            neg_strand = df.query("strand == '-'").index
            if upstream:
                df.loc[pos_strand, "chromStart"] -= upstream
                df.loc[neg_strand, "chromEnd"] += upstream
            if downstream:
                df.loc[pos_strand, "chromEnd"] += downstream
                df.loc[neg_strand, "chromStart"] -= downstream
        df["chromStart"] = np.maximum(df["chromStart"], 0)
        if chr_len:
            chr_len = df["chrom"].map(chr_len)
            df["chromEnd"] = np.minimum(df["chromEnd"], chr_len)
        return type(self)(df)


class Gtf(ConstrainedDataFrame):  # gffutils is too slow

    r"""
    GTF format data frame
    """

    COLUMNS = pd.Index([
        "seqname", "source", "feature", "start", "end",
        "score", "strand", "frame", "attribute"
    ])  # Additional columns after "attribute" is allowed

    @classmethod
    def rectify(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = super(Gtf, cls).rectify(df)
        COLUMNS = cls.COLUMNS.copy(deep=True)
        for item in COLUMNS:
            if item in df:
                if item in ("start", "end"):
                    df[item] = df[item].astype(int)
                else:
                    df[item] = df[item].astype(str)
            elif item not in ("seqname", "start", "end"):
                df[item] = "."
            else:
                raise ValueError(f"Required column {item} is missing!")
        return df.sort_index(axis=1, key=cls._column_key)

    @classmethod
    def _column_key(cls, x: pd.Index) -> np.ndarray:
        x = cls.COLUMNS.get_indexer(x)
        x[x < 0] = x.max() + 1  # Put additional columns after "attribute"
        return x

    @classmethod
    def verify(cls, df: pd.DataFrame) -> None:
        super(Gtf, cls).verify(df)
        if len(df.columns) < len(cls.COLUMNS) or \
                np.any(df.columns[:len(cls.COLUMNS)] != cls.COLUMNS):
            raise ValueError("Invalid GTF format!")

    @classmethod
    def read_gtf(cls, fname: os.PathLike) -> "Gtf":
        r"""
        Read GTF file

        Parameters
        ----------
        fname
            GTF file

        Returns
        -------
        gtf
            Loaded :class:`Gtf` object
        """
        COLUMNS = cls.COLUMNS.copy(deep=True)
        loaded = pd.read_csv(fname, sep="\t", header=None, comment="#")
        loaded.columns = COLUMNS[:loaded.shape[1]]
        return cls(loaded)

    def split_attribute(self) -> "Gtf":
        r"""
        Extract all attributes from the "attribute" column
        and append them to existing columns

        Returns
        -------
        splitted
            Gtf with splitted attribute columns appended
        """
        pattern = re.compile(r'([^\s]+) "([^"]+)";')
        splitted = pd.DataFrame.from_records(np.vectorize(lambda x: {
            key: val for key, val in pattern.findall(x)
        })(self["attribute"]), index=self.index)
        if set(self.COLUMNS).intersection(splitted.columns):
            self.logger.warning(
                "Splitted attribute names overlap standard GTF fields! "
                "The standard fields are overwritten!"
            )
        return self.assign(**splitted)

    def to_bed(self, name: Optional[str] = None) -> Bed:
        r"""
        Convert GTF to BED format

        Parameters
        ----------
        name
            Specify a column to be converted to the "name" column in bed format,
            otherwise the "name" column would be filled with "."

        Returns
        -------
        bed
            Converted :class:`Bed` object
        """
        bed_df = pd.DataFrame(self, copy=True).loc[
            :, ("seqname", "start", "end", "score", "strand")
        ]
        bed_df.insert(3, "name", np.repeat(
            ".", len(bed_df)
        ) if name is None else self[name])
        bed_df["start"] -= 1  # Convert to zero-based
        bed_df.columns = (
            "chrom", "chromStart", "chromEnd", "name", "score", "strand"
        )
        return Bed(bed_df)


def interval_dist(x: Interval, y: Interval) -> int:
    r"""
    Compute distance and relative position between two bed intervals

    Parameters
    ----------
    x
        First interval
    y
        Second interval

    Returns
    -------
    dist
        Signed distance between ``x`` and ``y``
    """
    if x.chrom != y.chrom:
        return np.inf * (-1 if x.chrom < y.chrom else 1)
    if x.start < y.stop and y.start < x.stop:
        return 0
    if x.stop <= y.start:
        return x.stop - y.start - 1
    if y.stop <= x.start:
        return x.start - y.stop + 1


def window_graph(
        left: Union[Bed, str], right: Union[Bed, str], window_size: int,
        left_sorted: bool = False, right_sorted: bool = False,
        attr_fn: Optional[Callable[[Interval, Interval, float], Mapping[str, Any]]] = None
) -> nx.MultiDiGraph:
    r"""
    Construct a window graph between two sets of genomic features, where
    features pairs within a window size are connected.

    Parameters
    ----------
    left
        First feature set, either a :class:`Bed` object or path to a bed file
    right
        Second feature set, either a :class:`Bed` object or path to a bed file
    window_size
        Window size (in bp)
    left_sorted
        Whether ``left`` is already sorted
    right_sorted
        Whether ``right`` is already sorted
    attr_fn
        Function to compute edge attributes for connected features,
        should accept the following three positional arguments:

        - l: left interval
        - r: right interval
        - d: signed distance between the intervals

        By default no edge attribute is created.

    Returns
    -------
    graph
        Window graph
    """
    check_deps("bedtools")
    if isinstance(left, Bed):
        pbar_total = len(left)
        left = left.to_bedtool()
    else:
        pbar_total = None
        left = pybedtools.BedTool(left)
    if not left_sorted:
        left = left.sort(stream=True)
    left = iter(left)  # Resumable iterator
    if isinstance(right, Bed):
        right = right.to_bedtool()
    else:
        right = pybedtools.BedTool(right)
    if not right_sorted:
        right = right.sort(stream=True)
    right = iter(right)  # Resumable iterator

    attr_fn = attr_fn or (lambda l, r, d: {})
    if pbar_total is not None:
        left = tqdm(left, total=pbar_total, desc="window_graph")
    graph = nx.MultiDiGraph()
    window = collections.OrderedDict()  # Used as ordered set
    for l in left:
        for r in list(window.keys()):  # Allow remove during iteration
            d = interval_dist(l, r)
            if -window_size <= d <= window_size:
                graph.add_edge(l.name, r.name, **attr_fn(l, r, d))
            elif d > window_size:
                del window[r]
            else:  # dist < -window_size
                break  # No need to expand window
        else:
            for r in right:  # Resume from last break
                d = interval_dist(l, r)
                if -window_size <= d <= window_size:
                    graph.add_edge(l.name, r.name, **attr_fn(l, r, d))
                elif d > window_size:
                    continue
                window[r] = None  # Placeholder
                if d < -window_size:
                    break
    pybedtools.cleanup()
    return graph


def dist_power_decay(x: int) -> float:
    r"""
    Distance-based power decay weight, computed as
    :math:`w = {\left( \frac {d + 1000} {1000} \right)} ^ {-0.75}`

    Parameters
    ----------
    x
        Distance (in bp)

    Returns
    -------
    weight
        Decaying weight
    """
    return ((x + 1000) / 1000) ** (-0.75)


@logged
def rna_anchored_guidance_graph(
    rna: AnnData, *others: AnnData,
    gene_region: str = "combined", promoter_len: int = 2000,
    extend_range: int = 0, extend_fn: Callable[[int], float] = dist_power_decay,
    signs: Optional[List[int]] = None, propagate_highly_variable: bool = True,
    corrupt_rate: float = 0.0, random_state: RandomState = None
) -> nx.MultiDiGraph:
    r"""
    Build guidance graph anchored on RNA genes

    Parameters
    ----------
    rna
        Anchor RNA dataset
    *others
        Other datasets
    gene_region
        Defines the genomic region of genes, must be one of
        ``{"gene_body", "promoter", "combined"}``.
    promoter_len
        Defines the length of gene promoters (bp upstream of TSS)
    extend_range
        Maximal extend distance beyond gene regions
    extend_fn
        Distance-decreasing weight function for the extended regions
        (by default :func:`dist_power_decay`)
    signs
        Sign of edges between RNA genes and features in each ``*others``
        dataset, must have the same length as ``*others``. Signs must be
        one of ``{-1, 1}``. By default, all edges have positive signs of ``1``.
    propagate_highly_variable
        Whether to propagate highly variable genes to other datasets,
        datasets in ``*others`` would be modified in place.
    corrupt_rate
        **CAUTION: DO NOT USE**, only for evaluation purpose
    random_state
        **CAUTION: DO NOT USE**, only for evaluation purpose

    Returns
    -------
    graph
        Prior regulatory graph

    Note
    ----
    In this function, features in the same dataset can only connect to
    anchor genes via the same edge sign. For more flexibility, please
    construct the guidance graph manually.
    """
    signs = signs or [1] * len(others)
    if len(others) != len(signs):
        raise RuntimeError("Length of ``others`` and ``signs`` must match!")
    if set(signs).difference({-1, 1}):
        raise RuntimeError("``signs`` can only contain {-1, 1}!")

    rna_bed = Bed(rna.var.assign(name=rna.var_names))
    other_beds = [Bed(other.var.assign(name=other.var_names)) for other in others]
    if gene_region == "promoter":
        rna_bed = rna_bed.strand_specific_start_site().expand(promoter_len, 0)
    elif gene_region == "combined":
        rna_bed = rna_bed.expand(promoter_len, 0)
    elif gene_region != "gene_body":
        raise ValueError("Unrecognized `gene_range`!")
    graphs = [window_graph(
        rna_bed, other_bed, window_size=extend_range,
        attr_fn=lambda l, r, d, s=sign: {
            "dist": abs(d), "weight": extend_fn(abs(d)), "sign": s
        }
    ) for other_bed, sign in zip(other_beds, signs)]
    graph = compose_multigraph(*graphs)

    corrupt_num = round(corrupt_rate * graph.number_of_edges())
    if corrupt_num:
        rna_anchored_guidance_graph.logger.warning("Corrupting guidance graph!")
        rs = get_rs(random_state)
        rna_var_names = rna.var_names.tolist()
        other_var_names = reduce(add, [other.var_names.tolist() for other in others])

        corrupt_remove = set(rs.choice(graph.number_of_edges(), corrupt_num, replace=False))
        corrupt_remove = set(edge for i, edge in enumerate(graph.edges) if i in corrupt_remove)
        corrupt_add = []
        while len(corrupt_add) < corrupt_num:
            corrupt_add += [
                (u, v) for u, v in zip(
                    rs.choice(rna_var_names, corrupt_num - len(corrupt_add)),
                    rs.choice(other_var_names, corrupt_num - len(corrupt_add))
                ) if not graph.has_edge(u, v)
            ]

        graph.add_edges_from([
            (add[0], add[1], graph.edges[remove])
            for add, remove in zip(corrupt_add, corrupt_remove)
        ])
        graph.remove_edges_from(corrupt_remove)

    if propagate_highly_variable:
        hvg_reachable = reachable_vertices(graph, rna.var.query("highly_variable").index)
        for other in others:
            other.var["highly_variable"] = [
                item in hvg_reachable for item in other.var_names
            ]

    rgraph = graph.reverse()
    nx.set_edge_attributes(graph, "fwd", name="type")
    nx.set_edge_attributes(rgraph, "rev", name="type")
    graph = compose_multigraph(graph, rgraph)
    all_features = set(chain.from_iterable(
        map(lambda x: x.var_names, [rna, *others])
    ))
    for item in all_features:
        graph.add_edge(item, item, weight=1.0, sign=1, type="loop")
    return graph


@logged
def rna_anchored_prior_graph(
    rna: AnnData, *others: AnnData,
    gene_region: str = "combined", promoter_len: int = 2000,
    extend_range: int = 0, extend_fn: Callable[[int], float] = dist_power_decay,
    signs: Optional[List[int]] = None, propagate_highly_variable: bool = True,
    corrupt_rate: float = 0.0, random_state: RandomState = None
) -> nx.MultiDiGraph:  # pragma: no cover
    r"""
    Deprecated, please use :func:`rna_anchored_guidance_graph` instead
    """
    rna_anchored_prior_graph.logger.warning(
        "Deprecated, please use `rna_anchored_guidance_graph` instead!"
    )
    return rna_anchored_guidance_graph(
        rna, *others, gene_region=gene_region, promoter_len=promoter_len,
        extend_range=extend_range, extend_fn=extend_fn, signs=signs,
        propagate_highly_variable=propagate_highly_variable,
        corrupt_rate=corrupt_rate, random_state=random_state
    )


def regulatory_inference(
        features: pd.Index, feature_embeddings: Union[np.ndarray, List[np.ndarray]],
        skeleton: nx.Graph, alternative: str = "two.sided",
        random_state: RandomState = None
) -> nx.Graph:
    r"""
    Regulatory inference based on feature embeddings

    Parameters
    ----------
    features
        Feature names
    feature_embeddings
        List of feature embeddings from 1 or more models
    skeleton
        Skeleton graph
    alternative
        Alternative hypothesis, must be one of {"two.sided", "less", "greater"}
    random_state
        Random state

    Returns
    -------
    regulatory_graph
        Regulatory graph containing regulatory score ("score"),
        *P*-value ("pval"), *Q*-value ("pval") as edge attributes
        for feature pairs in the skeleton graph
    """
    if isinstance(feature_embeddings, np.ndarray):
        feature_embeddings = [feature_embeddings]
    n_features = set(item.shape[0] for item in feature_embeddings)
    if len(n_features) != 1:
        raise ValueError("All feature embeddings must have the same number of rows!")
    if n_features.pop() != features.shape[0]:
        raise ValueError("Feature embeddings do not match the number of feature names!")
    node_idx = features.get_indexer(skeleton.nodes)
    features = features[node_idx]
    feature_embeddings = [item[node_idx] for item in feature_embeddings]

    rs = get_rs(random_state)
    vperm = np.stack([rs.permutation(item) for item in feature_embeddings], axis=1)
    vperm = vperm / np.linalg.norm(vperm, axis=-1, keepdims=True)
    v = np.stack(feature_embeddings, axis=1)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)

    edgelist = nx.to_pandas_edgelist(skeleton)
    source = features.get_indexer(edgelist["source"])
    target = features.get_indexer(edgelist["target"])
    fg, bg = [], []

    for s, t in tqdm(zip(source, target), total=skeleton.number_of_edges(), desc="regulatory_inference"):
        fg.append((v[s] * v[t]).sum(axis=1).mean())
        bg.append((vperm[s] * vperm[t]).sum(axis=1))
    edgelist["score"] = fg

    bg = np.sort(np.concatenate(bg))
    quantile = np.searchsorted(bg, fg) / bg.size
    if alternative == "two.sided":
        edgelist["pval"] = 2 * np.minimum(quantile, 1 - quantile)
    elif alternative == "greater":
        edgelist["pval"] = 1 - quantile
    elif alternative == "less":
        edgelist["pval"] = quantile
    else:
        raise ValueError("Unrecognized `alternative`!")
    edgelist["qval"] = fdrcorrection(edgelist["pval"])[1]
    return nx.from_pandas_edgelist(edgelist, edge_attr=True, create_using=type(skeleton))


def write_links(
    graph: nx.Graph, source: Bed, target: Bed, file: os.PathLike,
    keep_attrs: Optional[List[str]] = None
) -> None:
    r"""
    Export regulatory graph into a links file

    Parameters
    ----------
    graph
        Regulatory graph
    source
        Genomic coordinates of source nodes
    target
        Genomic coordinates of target nodes
    file
        Output file
    keep_attrs
        A list of attributes to keep for each link
    """
    nx.to_pandas_edgelist(
        graph
    ).merge(
        source.df.iloc[:, :4], how="left", left_on="source", right_on="name"
    ).merge(
        target.df.iloc[:, :4], how="left", left_on="target", right_on="name"
    ).loc[:, [
        "chrom_x", "chromStart_x", "chromEnd_x",
        "chrom_y", "chromStart_y", "chromEnd_y",
        *(keep_attrs or [])
    ]].to_csv(file, sep="\t", index=False, header=False)


def cis_regulatory_ranking(
        gene2region: nx.Graph, region2tf: nx.Graph,
        genes: List[str], regions: List[str], tfs: List[str],
        region_lens: Optional[List[int]] = None, n_samples: int = 1000,
        random_state: RandomState = None
) -> pd.DataFrame:
    r"""
    Generate cis-regulatory ranking between genes and transcription factors

    Parameters
    ----------
    gene2region
        A graph connecting genes to cis-regulatory regions
    region2tf
        A graph connecting cis-regulatory regions to transcription factors
    genes
        A list of genes
    tfs
        A list of transcription factors
    regions
        A list of cis-regulatory regions
    region_lens
        Lengths of cis-regulatory regions
        (if not provided, it is assumed that all regions have the same length)
    n_samples
        Number of random samples used to evaluate regulatory enrichment
        (setting this to 0 disables enrichment evaluation)
    random_state
        Random state

    Returns
    -------
    gene2tf_rank
        Cis regulatory ranking between genes and transcription factors
    """
    gene2region = biadjacency_matrix(gene2region, genes, regions, dtype=np.int16, weight=None)
    region2tf = biadjacency_matrix(region2tf, regions, tfs, dtype=np.int16, weight=None)

    if n_samples:
        region_lens = [1] * len(regions) if region_lens is None else region_lens
        if len(region_lens) != len(regions):
            raise ValueError("`region_lens` must have the same length as `regions`!")
        region_bins = pd.qcut(region_lens, min(len(set(region_lens)), 500), duplicates="drop")
        region_bins_lut = pd.RangeIndex(region_bins.size).groupby(region_bins)

        rs = get_rs(random_state)
        row, col_rand, data = [], [], []
        lil = gene2region.tolil()
        for r, (c, d) in tqdm(
                enumerate(zip(lil.rows, lil.data)),
                total=len(lil.rows), desc="cis_reg_ranking.sampling"
        ):
            if not c:  # Empty row
                continue
            row.append(np.ones_like(c) * r)
            col_rand.append(np.stack([
                rs.choice(region_bins_lut[region_bins[c_]], n_samples, replace=True)
                for c_ in c
            ], axis=0))
            data.append(d)
        row = np.concatenate(row)
        col_rand = np.concatenate(col_rand)
        data = np.concatenate(data)

        gene2tf_obs = (gene2region @ region2tf).toarray()
        gene2tf_rand = np.empty((len(genes), len(tfs), n_samples), dtype=np.int16)
        for k in tqdm(range(n_samples), desc="cis_reg_ranking.mapping"):
            gene2region_rand = scipy.sparse.coo_matrix((
                data, (row, col_rand[:, k])
            ), shape=(len(genes), len(regions)))
            gene2tf_rand[:, :, k] = (gene2region_rand @ region2tf).toarray()
        gene2tf_rand.sort(axis=2)

        gene2tf_enrich = np.empty_like(gene2tf_obs)
        for i, j in product(range(len(genes)), range(len(tfs))):
            if gene2tf_obs[i, j] == 0:
                gene2tf_enrich[i, j] = 0
                continue
            gene2tf_enrich[i, j] = np.searchsorted(
                gene2tf_rand[i, j, :], gene2tf_obs[i, j], side="right"
            )
    else:
        gene2tf_enrich = (gene2region @ region2tf).toarray()

    return pd.DataFrame(
        scipy.stats.rankdata(-gene2tf_enrich, axis=0),
        index=genes, columns=tfs
    )


def write_scenic_feather(
        gene2tf_rank: pd.DataFrame, feather: os.PathLike,
        version: int = 2
) -> None:
    r"""
    Write cis-regulatory ranking to a SCENIC-compatible feather file

    Parameters
    ----------
    gene2tf_rank
        Cis regulatory ranking between genes and transcription factors,
        as generated by :func:`cis_reg_ranking`
    feather
        Path to the output feather file
    version
        SCENIC feather version
    """
    if version not in {1, 2}:
        raise ValueError("Unrecognized SCENIC feather version!")
    if version == 2:
        suffix = ".genes_vs_tracks.rankings.feather"
        if not str(feather).endswith(suffix):
            raise ValueError(f"Feather file name must end with `{suffix}`!")
    tf2gene_rank = gene2tf_rank.T
    tf2gene_rank = tf2gene_rank.loc[
        np.unique(tf2gene_rank.index), np.unique(tf2gene_rank.columns)
    ].astype(np.int16)
    tf2gene_rank.index.name = "features" if version == 1 else "tracks"
    tf2gene_rank.columns.name = None
    columns = tf2gene_rank.columns.tolist()
    tf2gene_rank = tf2gene_rank.reset_index()
    if version == 2:
        tf2gene_rank = tf2gene_rank.loc[:, [*columns, "tracks"]]
    tf2gene_rank.to_feather(feather)


def read_ctx_grn(file: os.PathLike) -> nx.DiGraph:
    r"""
    Read pruned TF-target GRN as generated by ``pyscenic ctx``

    Parameters
    ----------
    file
        Input file (.csv)

    Returns
    -------
    grn
        Pruned TF-target GRN

    Note
    ----
    Node attribute "type" can be used to distinguish TFs and genes
    """
    df = pd.read_csv(
        file, header=None, skiprows=3,
        usecols=[0, 8], names=["TF", "targets"]
    )
    df["targets"] = df["targets"].map(lambda x: set(i[0] for i in literal_eval(x)))
    df = df.groupby("TF").aggregate({"targets": lambda x: reduce(set.union, x)})
    grn = nx.DiGraph([
        (tf, target)
        for tf, row in df.iterrows()
        for target in row["targets"]]
    )
    nx.set_node_attributes(grn, "target", name="type")
    for tf in df.index:
        grn.nodes[tf]["target"] = "TF"
    return grn


def get_chr_len_from_fai(fai: os.PathLike) -> Mapping[str, int]:
    r"""
    Get chromosome length information from fasta index file

    Parameters
    ----------
    fai
        Fasta index file

    Returns
    -------
    chr_len
        Length of each chromosome
    """
    return pd.read_table(fai, header=None, index_col=0)[1].to_dict()


def ens_trim_version(x: str) -> str:
    r"""
    Trim version suffix from Ensembl ID

    Parameters
    ----------
    x
        Ensembl ID

    Returns
    -------
    trimmed
        Ensembl ID with version suffix trimmed
    """
    return re.sub(r"\.[0-9_-]+$", "", x)


# Aliases
read_bed = Bed.read_bed
read_gtf = Gtf.read_gtf
