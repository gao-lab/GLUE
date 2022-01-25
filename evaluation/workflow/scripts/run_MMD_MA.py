#!/usr/bin/env python

r"""
Run MMD-MA
"""

import argparse
import os
import pathlib
import subprocess
import sys
import time

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import yaml
from sklearn.preprocessing import normalize

import scglue


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
    torch.manual_seed(args.random_seed)

    print("[1/4] Reading data...")
    rna = anndata.read_h5ad(args.input_rna)
    atac = anndata.read_h5ad(args.input_atac)

    print("[2/4] Preprocessing...")
    start_time = time.time()
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    sc.pp.scale(rna, max_value=10)
    sc.tl.pca(
        rna, n_comps=min(100, rna.shape[0]),
        use_highly_variable=True, svd_solver="auto"
    )
    scglue.data.lsi(
        atac, n_components=min(100, atac.shape[0]),
        use_highly_variable=False, n_iter=15
    )

    print("[3/4] Training MMD-MA...")
    output_dir = args.run_info.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    X = rna.obsm["X_pca"]
    Y = atac.obsm["X_lsi"]
    X = normalize(X, norm="l2")
    Y = normalize(Y, norm="l2")
    K1 = X @ X.T
    K2 = Y @ Y.T
    np.save(output_dir / "K1.npy", K1)
    np.save(output_dir / "K2.npy", K2)
    scglue.models.nn.autodevice()
    subprocess.call([
        "python", "../custom/2020_mmdma_pytorch/manifold_align_mmd_pytorch.py",
        output_dir / "K1.npy", output_dir / "K2.npy", f"{output_dir}{os.sep}",
        "5", "0.0", "1e-5", "1e-5", str(args.random_seed)
    ])
    print("MMD-MA finished...")
    alpha = np.loadtxt(
        output_dir /
        "results_nfeat_5_sigma_0.0_lam1_1e-05_lam2_1e-05" /
        f"seed_{args.random_seed}" /
        f"alpha_hat_{args.random_seed}_10000.txt"
    )
    beta = np.loadtxt(
        output_dir /
        "results_nfeat_5_sigma_0.0_lam1_1e-05_lam2_1e-05" /
        f"seed_{args.random_seed}" /
        f"beta_hat_{args.random_seed}_10000.txt"
    )
    rna_latent = K1 @ alpha
    atac_latent = K2 @ beta
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
