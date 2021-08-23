#!/usr/bin/env python

r"""
Convert data based on feature mapping

The script assumes that there is an anchor RNA dataset with genes as features,
and features in other datasets are converted to the RNA genes according to
the prior graph.
"""

import argparse
import pathlib

import anndata
import networkx as nx
from networkx.algorithms.bipartite import biadjacency_matrix


def parse_args() -> argparse.Namespace:
    r"""
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Convert data based on feature mapping"
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
        "-p", "--prior", dest="prior", type=pathlib.Path, required=True,
        help="Path to the prior graph (.graphml[.gz])"
    )
    parser.add_argument(
        "-o", "--output", dest="output", type=pathlib.Path, required=True,
        help="Path to the converted dataset (.h5ad)"
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    r"""
    Main function
    """
    print("[1/4] Reading data...")
    rna = anndata.read_h5ad(args.rna)
    atac = anndata.read_h5ad(args.atac)

    print("[2/4] Reading prior graph...")
    graph = nx.read_graphml(args.prior)

    print("[3/4] Converting data...")
    weight = biadjacency_matrix(
        graph, atac.var_names, rna.var_names,
        weight="weight", format="csc"
    )
    sign = biadjacency_matrix(
        graph, atac.var_names, rna.var_names,
        weight="sign", format="csc"
    )
    atac2rna = anndata.AnnData(
        X=atac.X @ weight.multiply(sign),
        obs=atac.obs, var=rna.var
    )

    print("[4/4] Saving results...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atac2rna.write(args.output, compression="gzip")


if __name__ == "__main__":
    main(parse_args())
