#!/usr/bin/env python

r"""
Export h5ad to pickle file
"""

import argparse
import pathlib
import pickle

import anndata


def parse_args():
    r"""
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=pathlib.Path, required=True)
    parser.add_argument("-o", "--output", dest="output", type=pathlib.Path, required=True)
    return parser.parse_args()


def main(args):
    r"""
    Main function
    """
    adata = anndata.read_h5ad(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump({
            "X": adata.X,
            "obs": adata.obs,
            "var": adata.var
        }, f)


if __name__ == "__main__":
    main(parse_args())
