#!/usr/bin/env python

r"""
Run UnionCom
"""

import argparse
import os
import pathlib
import sys
import time
from typing import Optional, Union

import anndata
import numpy as np
import pandas as pd
import pynvml
import scanpy as sc
import scipy.sparse
import sklearn.preprocessing
import sklearn.utils.extmath
import yaml
from unioncom import UnionCom

Array = Union[np.ndarray, scipy.sparse.spmatrix]


def lsi(
        adata: anndata.AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs
) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)
    """
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi


def tfidf(X: Array) -> Array:
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf


def autodevice():
    r"""
    Set computation device automatically
    based on GPU availability and memory usage
    """
    used_device = -1
    try:
        pynvml.nvmlInit()
        free_mems = np.array([
            pynvml.nvmlDeviceGetMemoryInfo(
                pynvml.nvmlDeviceGetHandleByIndex(i)
            ).free for i in range(pynvml.nvmlDeviceGetCount())
        ])
        best_devices = np.where(free_mems == free_mems.max())[0]
        used_device = np.random.choice(best_devices, 1)[0]
    except pynvml.NVMLError:
        pass
    if used_device == -1:
        print("Using CPU as computation device.")
        return
    print(f"Using GPU {used_device} as computation device.")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(used_device)


def parse_args() -> argparse.Namespace:
    r"""
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-rna", dest="input_rna", type=pathlib.Path, required=True,
        help="Path to input RNA dataset (.h5ad)"
    )
    parser.add_argument(
        "--input-atac", dest="input_atac", type=pathlib.Path, required=True,
        help="Path to input ATAC dataset (.h5ad)"
    )
    parser.add_argument(
        "-s", "--random-seed", dest="random_seed", type=int, default=0,
        help="Random seed"
    )
    parser.add_argument(
        "--output-rna", dest="output_rna", type=pathlib.Path, required=True,
        help="Path of output RNA latent file (.csv)"
    )
    parser.add_argument(
        "--output-atac", dest="output_atac", type=pathlib.Path, required=True,
        help="Path of output ATAC latent file (.csv)"
    )
    parser.add_argument(
        "-r", "--run-info", dest="run_info", type=pathlib.Path, required=True,
        help="Path of output run info file (.yaml)"
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    r"""
    Main function
    """
    np.random.seed(args.random_seed)

    print("[1/4] Reading data...")
    rna = anndata.read_h5ad(args.input_rna)
    atac = anndata.read_h5ad(args.input_atac)

    print("[2/4] Preprocessing...")
    start_time = time.time()
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    sc.pp.scale(rna, max_value=10)
    sc.tl.pca(
        rna, n_comps=min(300, rna.shape[0]),
        use_highly_variable=True, svd_solver="auto"
    )
    lsi(
        atac, n_components=min(300, atac.shape[0]),
        use_highly_variable=False, n_iter=15
    )
    X = rna.obsm["X_pca"]
    Y = atac.obsm["X_lsi"]

    print("[3/4] Training UnionCom...")
    autodevice()
    uc = UnionCom.UnionCom(manual_seed=args.random_seed)
    rna_latent, atac_latent = uc.fit_transform(dataset=[X, Y])
    elapsed_time = time.time() - start_time

    print("[4/4] Saving results...")
    args.output_rna.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rna_latent, index=rna.obs_names).to_csv(args.output_rna, header=False)
    args.output_atac.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(atac_latent, index=atac.obs_names).to_csv(args.output_atac, header=False)
    args.run_info.parent.mkdir(parents=True, exist_ok=True)
    with args.run_info.open("w") as f:
        yaml.dump({
            "cmd": " ".join(sys.argv),
            "args": vars(args),
            "time": elapsed_time,
            "n_cells": atac.shape[0] + rna.shape[0]
        }, f)


if __name__ == "__main__":
    main(parse_args())
