#!/usr/bin/env python

r"""
Subsample datasets
"""

import argparse
import pathlib

import anndata
import numpy as np
import scanpy as sc


def parse_args() -> argparse.Namespace:
    r"""
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Subsample datasets")
    parser.add_argument(
        "-d", "--datasets", dest="datasets", type=pathlib.Path, required=True,
        nargs="+", help="Path to input datasets (.h5ad)"
    )
    parser.add_argument(
        "-s", "--subsample-size", dest="subsample_size", type=int,
        required=False, default=None, help="Subsample size"
    )
    parser.add_argument(
        "-f", "--subsample-frac", dest="subsample_frac", type=float,
        required=False, default=None, help="Subsample fraction"
    )
    parser.add_argument(
        "-p", "--paired", dest="paired", default=False, action="store_true",
        help="Whether the datasets are paired"
    )
    parser.add_argument(
        "--random-seed", dest="random_seed", type=int, default=0,
        help="Random seed"
    )
    parser.add_argument(
        "-o", "--outputs", dest="outputs", type=pathlib.Path, required=True,
        nargs="+", help="Path to subsampled datasets (.h5ad)"
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    r"""
    Main function
    """
    if len(args.datasets) != len(args.outputs):
        raise RuntimeError("Datasets and outputs should have the same number of entries!")
    if (args.subsample_size is None) == (args.subsample_frac is None):
        raise RuntimeError("Exactly one of -s and -f should be specified!")
    rs = np.random.RandomState(args.random_seed)

    print("[1/3] Reading data...")
    datasets = [anndata.read_h5ad(item) for item in args.datasets]

    print("[2/3] Subsampling data...")
    if args.paired:
        size = set(dataset.shape[0] for dataset in datasets)
        if not len(size) == 1:
            raise RuntimeError("Datasets are not paired!")
        size = size.pop()
        subsample_size = args.subsample_size or np.round(
            size * args.subsample_frac
        ).astype(int)
        idx = rs.choice(size, subsample_size, replace=False)
        subsampled_datasets = [dataset[idx, :] for dataset in datasets]
    else:
        sizes = [dataset.shape[0] for dataset in datasets]
        subsample_sizes = [
            args.subsample_size or np.round(size * args.subsample_frac).astype(int)
            for size in sizes
        ]
        subsampled_datasets = [
            dataset[rs.choice(size, subsample_size, replace=False), :]
            for dataset, size, subsample_size in zip(datasets, sizes, subsample_sizes)
        ]
    # for subsampled_dataset in subsampled_datasets:
    #     sc.pp.filter_genes(subsampled_dataset, min_counts=1)  # inplace

    print("[3/3] Saving results...")
    for subsampled_dataset, output in zip(subsampled_datasets, args.outputs):
        output.parent.mkdir(parents=True, exist_ok=True)
        subsampled_dataset.write(output, compression="gzip")


if __name__ == "__main__":
    main(parse_args())
