#!/usr/bin/env python

r"""
Build prior graph between RNA and ATAC datasets
"""

import argparse
import itertools
import pathlib

import anndata
import networkx as nx

import scglue


def parse_args() -> argparse.Namespace:
    r"""
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Build prior graph between RNA and ATAC datasets"
    )
    parser.add_argument(
        "--rna", dest="rna", type=pathlib.Path, required=True,
        help="Path to the RNA dataset (.h5ad)"
    )
    parser.add_argument(
        "--atac", dest="atac", type=pathlib.Path, required=True,
        help="Path to the ATAC dataset (.h5ad)"
    )
    parser.add_argument(
        "-r", "--gene-region", dest="gene_region", type=str, default="combined",
        choices=["gene_body", "promoter", "combined"],
        help="Genomic range of genes"
    )
    parser.add_argument(
        "-e", "--extend-range", dest="extend_range", type=int, default=0,
        help="Maximal extend distance beyond gene regions"
    )
    parser.add_argument(
        "-c", "--corrupt-rate", dest="corrupt_rate", type=float, default=0.0,
        help="Rate of corruption added to the prior graph"
    )
    parser.add_argument(
        "-s", "--corrupt-seed", dest="corrupt_seed", type=int, default=0,
        help="Corruption random seed"
    )
    parser.add_argument(
        "--output-full", dest="output_full", type=pathlib.Path, required=True,
        help="Path to the output full graph (.graphml[.gz])"
    )
    parser.add_argument(
        "--output-sub", dest="output_sub", type=pathlib.Path, required=True,
        help="Path to the output subgraph reachable from highly variable genes (.graphml[.gz])"
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    r"""
    Main function
    """
    print("[1/3] Reading data...")
    rna = anndata.read_h5ad(args.rna, backed="r")
    atac = anndata.read_h5ad(args.atac, backed="r")

    print("[2/3] Building prior graph...")
    graph = scglue.genomics.rna_anchored_prior_graph(
        rna, atac, gene_region=args.gene_region, extend_range=args.extend_range,
        propagate_highly_variable=True, corrupt_rate=args.corrupt_rate,
        random_state=args.corrupt_seed
    )
    subgraph = graph.subgraph(set(itertools.chain(
        rna.var.query("highly_variable").index,
        atac.var.query("highly_variable").index
    )))

    print("[3/3] Saving result...")
    args.output_full.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, args.output_full)
    args.output_sub.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(subgraph, args.output_sub)


if __name__ == "__main__":
    main(parse_args())
