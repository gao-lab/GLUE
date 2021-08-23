#!/usr/bin/env python

r"""
Visualize UMAP embeddings
"""

import argparse
import pathlib

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib import rcParams

import scglue

scglue.plot.set_publication_params()


def parse_args() -> argparse.Namespace:
    r"""
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Visualize UMAP embeddings"
    )
    parser.add_argument(
        "-d", "--datesets", dest="datasets", type=pathlib.Path, required=True,
        nargs="+", help="Path to datasets (.h5ad)"
    )
    parser.add_argument(
        "-u", "--umaps", dest="umaps", type=pathlib.Path, required=True,
        nargs="+", help="Path to umap embeddings (.csv)"
    )
    parser.add_argument(
        "-l", "--label", dest="label", type=str, required=True,
        help="Cell label (column name in `obs`) used for coloring"
    )
    parser.add_argument(
        "-t", "--title", dest="title", type=str, default=None,
        help="Plot title (by default same as `--label`)"
    )
    parser.add_argument(
        "--figsize", dest="figsize", type=float, default=5.0,
        help="Figure size"
    )
    parser.add_argument(
        "-o", "--output", dest="output", type=pathlib.Path, required=True,
        help="Path to output plot file"
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    r"""
    Main function
    """
    if len(args.datasets) != len(args.umaps):
        raise RuntimeError("Datasets and umaps should have the same number of entries!")

    print("[1/2] Reading data...")
    label = np.concatenate([
        anndata.read_h5ad(dataset, backed="r").obs[args.label]
        for dataset in args.datasets
    ])
    umap = np.concatenate([
        pd.read_csv(item, header=None, index_col=0)
        for item in args.umaps
    ])
    if label.shape[0] != umap.shape[0]:
        raise RuntimeError("Label and UMAP should have the same number of cells!")
    shuffle = np.random.RandomState(0).permutation(label.shape[0])
    adata = anndata.AnnData(
        X=np.empty((label.shape[0], 0)),
        obs=pd.DataFrame({
            args.label: pd.Categorical(label[shuffle], categories=np.unique(label))
        }, index=pd.RangeIndex(label.shape[0]).astype(str)),
        obsm={"X_umap": umap[shuffle]}
    )

    print("[2/2] Plotting...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rcParams["figure.figsize"] = (args.figsize, args.figsize)
    fig = sc.pl.umap(adata, color=args.label, title=args.title, return_fig=True)
    fig.savefig(args.output)


if __name__ == "__main__":
    main(parse_args())
