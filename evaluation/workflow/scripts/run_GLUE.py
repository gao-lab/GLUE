#!/usr/bin/env python

r"""
Data integration for scRNA-seq and scATAC-seq via the GLUE model
"""

import argparse
import logging
import pathlib
import random
import sys
import time

import anndata
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import yaml

import scglue

scglue.log.console_log_level = logging.DEBUG


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
        "-p", "--prior", dest="prior", type=pathlib.Path, required=True,
        help="Path to prior graph (.graphml[.gz])"
    )
    parser.add_argument(
        "-d", "--dim", dest="dim", type=int, default=50,
        help="Latent dimensionality"
    )
    parser.add_argument(
        "--alt-dim", dest="alt_dim", type=int, default=100,
        help="Alternative input dimensionality"
    )
    parser.add_argument(
        "--hidden-depth", dest="hidden_depth", type=int, default=2,
        help="Hidden layer depth for encoder and discriminator"
    )
    parser.add_argument(
        "--hidden-dim", dest="hidden_dim", type=int, default=256,
        help="Hidden layer dimensionality for encoder and discriminator"
    )
    parser.add_argument(
        "--dropout", dest="dropout", type=float, default=0.2,
        help="Dropout rate"
    )
    parser.add_argument(
        "--lam-graph", dest="lam_graph", type=float, default=0.02,
        help="Graph weight"
    )
    parser.add_argument(
        "--lam-align", dest="lam_align", type=float, default=0.02,
        help="Adversarial alignment weight"
    )
    parser.add_argument(
        "--lr", dest="lr", type=float, default=2e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--neg-samples", dest="neg_samples", type=int, default=10,
        help="Number of negative samples for each edge"
    )
    parser.add_argument(
        "--val-split", dest="val_split", type=float, default=0.1,
        help="Validation split"
    )
    parser.add_argument(
        "--data-batch-size", dest="data_batch_size", type=int, default=128,
        help="Number of cells in each data minibatch"
    )
    parser.add_argument(
        "--balance-res", dest="balance_res", type=float, default=1.0,
        help="Clustering resolution for estimating balancing weight"
    )
    parser.add_argument(
        "--balance-cutoff", dest="balance_cutoff", type=float, default=0.5,
        help="Cosine similarity cutoff for estimating balancing weight"
    )
    parser.add_argument(
        "--balance-power", dest="balance_power", type=float, default=4.0,
        help="Cosine similarity power for estimating balancing weight"
    )
    parser.add_argument(
        "-s", "--random-seed", dest="random_seed", type=int, default=0,
        help="Random seed"
    )
    parser.add_argument(
        "--train-dir", dest="train_dir", type=pathlib.Path, required=True,
        help="Path to directory where training logs and checkpoints are stored"
    )
    parser.add_argument(
        "--random-sleep", dest="random_sleep", default=False, action="store_true",
        help="Whether to sleep random number of seconds before running, "
             "which helps distribute jobs evenly over multiple GPUs."
    )
    parser.add_argument(
        "--require-converge", dest="require_converge", default=False, action="store_true",
        help="Whether to require convergence and disallow premature interruption"
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
        "--output-feature", dest="output_feature", type=pathlib.Path, required=True,
        help="Path of output feature latent file (.csv)"
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
    print("[1/4] Reading data...")
    rna = anndata.read_h5ad(args.input_rna)
    atac = anndata.read_h5ad(args.input_atac)
    graph = nx.read_graphml(args.prior)

    if args.random_sleep:
        time.sleep(random.randint(0, 10))

    print("[2/4] Preprocessing...")
    start_time = time.time()
    atac.var["highly_variable"] = [graph.has_node(item) for item in atac.var_names]
    rna_use_rep, atac_use_rep = None, None
    if args.alt_dim:
        rna.layers["counts"] = rna.X.copy()
        sc.pp.normalize_total(rna)
        rna = rna[:, rna.var["highly_variable"]].copy()
        sc.pp.log1p(rna)
        sc.pp.scale(rna, max_value=10)
        sc.tl.pca(rna, n_comps=args.alt_dim, svd_solver="auto")
        rna_use_rep = "X_pca"
        rna.X = rna.layers["counts"]
        del rna.layers["counts"]
        scglue.data.lsi(
            atac, n_components=args.alt_dim,
            use_highly_variable=False, n_iter=15
        )
        atac_use_rep = "X_lsi"
    scglue.models.configure_dataset(rna, "NB", use_highly_variable=True, use_rep=rna_use_rep)
    scglue.models.configure_dataset(atac, "NB", use_highly_variable=True, use_rep=atac_use_rep)

    print("[3/4] Training GLUE...")
    scglue.config.ALLOW_TRAINING_INTERRUPTION = not args.require_converge
    vertices = sorted(graph.nodes)
    glue = scglue.models.SCGLUEModel(
        {"rna": rna, "atac": atac}, vertices,
        latent_dim=args.dim, h_depth=args.hidden_depth, h_dim=args.hidden_dim,
        dropout=args.dropout, random_seed=args.random_seed
    )
    glue.compile(lam_graph=args.lam_graph, lam_align=args.lam_align, lr=args.lr)
    glue.fit(
        {"rna": rna, "atac": atac},
        graph, edge_weight="weight", edge_sign="sign",
        neg_samples=args.neg_samples,
        val_split=args.val_split,
        data_batch_size=args.data_batch_size,
        align_burnin=np.inf, safe_burnin=False,
        directory=args.train_dir / "pretrain"
    )
    glue.save(args.train_dir / "pretrain" / "final.dill")

    rna.obsm["X_glue"] = glue.encode_data("rna", rna)
    atac.obsm["X_glue"] = glue.encode_data("atac", atac)
    scglue.data.estimate_balancing_weight(
        rna, atac, use_rep="X_glue", resolution=args.balance_res,
        cutoff=args.balance_cutoff, power=args.balance_power
    )
    scglue.models.configure_dataset(
        rna, "NB", use_highly_variable=True,
        use_rep=rna_use_rep, use_dsc_weight="balancing_weight"
    )
    scglue.models.configure_dataset(
        atac, "NB", use_highly_variable=True,
        use_rep=atac_use_rep, use_dsc_weight="balancing_weight"
    )

    glue = scglue.models.SCGLUEModel(
        {"rna": rna, "atac": atac}, vertices,
        latent_dim=args.dim, h_depth=args.hidden_depth, h_dim=args.hidden_dim,
        dropout=args.dropout, random_seed=args.random_seed
    )
    glue.adopt_pretrained_model(scglue.models.load_model(
        args.train_dir / "pretrain" / "final.dill"
    ))
    glue.compile(lam_graph=args.lam_graph, lam_align=args.lam_align, lr=args.lr)
    glue.fit(
        {"rna": rna, "atac": atac},
        graph, edge_weight="weight", edge_sign="sign",
        neg_samples=args.neg_samples,
        val_split=args.val_split,
        data_batch_size=args.data_batch_size,
        directory=args.train_dir / "fine-tune"
    )
    glue.save(args.train_dir / "fine-tune" / "final.dill")

    rna.obsm["X_glue"] = glue.encode_data("rna", rna)
    atac.obsm["X_glue"] = glue.encode_data("atac", atac)
    elapsed_time = time.time() - start_time
    feature_latent = glue.encode_graph(graph, "weight", "sign")

    print("[4/4] Saving results...")
    args.output_rna.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rna.obsm["X_glue"], index=rna.obs_names).to_csv(args.output_rna, header=False)
    args.output_atac.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(atac.obsm["X_glue"], index=atac.obs_names).to_csv(args.output_atac, header=False)
    args.output_feature.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(feature_latent, index=glue.vertices).to_csv(args.output_feature, header=False)
    args.run_info.parent.mkdir(parents=True, exist_ok=True)
    with args.run_info.open("w") as f:
        yaml.dump({
            "cmd": " ".join(sys.argv),
            "args": vars(args),
            "time": elapsed_time,
            "n_cells": atac.shape[0] + rna.shape[0]
        }, f)
    glue.save(args.train_dir / "final.dill")


if __name__ == "__main__":
    main(parse_args())
