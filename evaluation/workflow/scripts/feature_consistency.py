#!/usr/bin/env python

r"""
Compute feature consistency metrics
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    r"""
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Compute feature consistency"
    )
    parser.add_argument(
        "-q", "--query", dest="query", type=pathlib.Path, required=True,
        help="Path of query feature latent files (.csv)"
    )
    parser.add_argument(
        "-r", "--ref", dest="ref", type=pathlib.Path, required=True,
        help="Path of reference feature latent files (.csv)"
    )
    parser.add_argument(
        "-f", "--n-features", dest="n_features", type=int, default=2000,
        help="Number of subsampled features"
    )
    parser.add_argument(
        "-p", "--n-repeats", dest="n_repeats", type=int, default=4,
        help="Number of subsampling repeats"
    )
    parser.add_argument(
        "-s", "--random-seed", dest="random_seed", type=int, default=0,
        help="Random seed for feature subsampling"
    )
    parser.add_argument(
        "-o", "--output", dest="output", type=pathlib.Path, required=True,
        help="Path of output consistency file (.yaml)"
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:  # pylint: disable=redefined-outer-name
    r"""
    Main function
    """
    print("[1/3] Reading data...")
    query = pd.read_csv(args.query, header=None, index_col=0)
    ref = pd.read_csv(args.ref, header=None, index_col=0)
    common = np.intersect1d(query.index, ref.index)
    print(f"Number of common features: {common.size}")
    query = query.loc[common].to_numpy().astype(np.float32)
    ref = ref.loc[common].to_numpy().astype(np.float32)

    print("[2/3] Computing metrics...")
    consistency = []
    rs = np.random.RandomState(args.random_seed)
    for _ in range(args.n_repeats):
        subsample_idx = rs.choice(ref.shape[0], args.n_features, replace=False)
        query_sub, ref_sub = query[subsample_idx], ref[subsample_idx]
        query_sub = query_sub / np.linalg.norm(query_sub, axis=1, keepdims=True)
        ref_sub = ref_sub / np.linalg.norm(ref_sub, axis=1, keepdims=True)
        cosine_query = (query_sub @ query_sub.T).ravel()
        cosine_ref = (ref_sub @ ref_sub.T).ravel()
        cosine_query = (cosine_query - cosine_query.mean()) / cosine_query.std()
        cosine_ref = (cosine_ref - cosine_ref.mean()) / cosine_ref.std()
        consistency.append((cosine_query * cosine_ref).mean())
    consistency = np.mean(consistency).item()

    print("[3/3] Saving results...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump({"feature_consistency": consistency}, f)


if __name__ == "__main__":
    main(parse_args())
