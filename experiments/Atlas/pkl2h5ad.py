#!/usr/bin/env python

r"""
Export pickle to old version h5ad file
(need to run with anndata < 0.7)
"""

import argparse
import pathlib
import pickle

import anndata
import pandas as pd
from packaging.version import parse

if parse(anndata.__version__) >= parse("0.7"):
    raise ImportError("Version of anndata is too high (requires anndata < 0.7)!")


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
    with open(args.input, "rb") as f:
        pkl = pickle.load(f)
    X, obs, var = pkl["X"], pkl["obs"], pkl["var"]
    obs = pd.DataFrame({
        "cell": obs.index.tolist()
    }, index=obs.index.tolist())
    var = pd.DataFrame({
        "highly_variable": var["highly_variable"].tolist()
            if "highly_variable" in var else False  # Placeholder, not used
    }, index=var.index.tolist())
    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.raw = adata
    args.output.parent.mkdir(parents=True, exist_ok=True)
    adata.write(args.output, compression="gzip")

if __name__ == "__main__":
    main(parse_args())
