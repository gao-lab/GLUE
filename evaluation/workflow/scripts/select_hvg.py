#!/usr/bin/env python

r"""
Select highly variable genes
"""

import argparse
import pathlib

import anndata
import scanpy as sc


def parse_args() -> argparse.Namespace:
    r"""
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Subsample datasets")
    parser.add_argument(
        "-i", "--input", dest="input", type=pathlib.Path, required=True,
        help="Path to input dataset (.h5ad)"
    )
    parser.add_argument(
        "-n", "--n-hvg", dest="n_hvg", type=int, required=True,
        help="Number of highly variable genes to select"
    )
    parser.add_argument(
        "-o", "--output", dest="output", type=pathlib.Path, required=True,
        help="Path to output dataset (.h5ad)"
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    r"""
    Main function
    """
    print("[1/3] Reading data...")
    adata = anndata.read_h5ad(args.input)

    print("[2/3] Finding highly variable genes...")
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3")

    print("[3/3] Saving results...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    adata.write(args.output, compression="gzip")


if __name__ == "__main__":
    main(parse_args())
