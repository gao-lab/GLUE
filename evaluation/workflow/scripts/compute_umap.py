#!/usr/bin/env python

r"""
Compute UMAP embeddings
"""

import argparse
import pathlib

import anndata
import numpy as np
import pandas as pd
import scanpy as sc


def parse_args() -> argparse.Namespace:
    r"""
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Compute UMAP embeddings"
    )
    parser.add_argument(
        "-l", "--latents", dest="latents", type=pathlib.Path, required=True,
        nargs="+", help="Path to latent embeddings (.csv)"
    )
    parser.add_argument(
        "--n-neighbors", dest="n_neighbors", type=int, default=15,
        help="Size of local neighborhood used for manifold approximation"
    )
    parser.add_argument(
        "--metric", dest="metric", type=str, default="cosine",
        help="Distance metric"
    )
    parser.add_argument(
        "--min-dist", dest="min_dist", type=float, default=0.5,
        help="Effective minimum distance between embedded points"
    )
    parser.add_argument(
        "-o", "--outputs", dest="outputs", type=pathlib.Path, required=True,
        nargs="+", help="Path to output umap embeddings (.csv)"
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    r"""
    Main function
    """
    if len(args.latents) != len(args.outputs):
        raise RuntimeError("Latents and outputs should have the same number of entries!")

    print("[1/3] Reading data...")
    latents = [
        pd.read_csv(
            item, header=None, index_col=0,
            converters={0: lambda x, i=i: x + f".ITEM{i}"}  # Avoid collision
        ) for i, item in enumerate(args.latents)
    ]
    indices = [item.index for item in latents]
    latent = pd.concat(latents).dropna()
    adata = anndata.AnnData(
        X=np.empty((latent.shape[0], 0)),
        obsm={"X_latent": latent.to_numpy()}
    )

    print("[2/3] Computing UMAP...")
    sc.pp.neighbors(
        adata, n_neighbors=args.n_neighbors,
        use_rep="X_latent", n_pcs=latent.shape[1],
        random_state=0, metric=args.metric
    )
    sc.tl.umap(adata, min_dist=args.min_dist, random_state=0)
    umap = pd.DataFrame(adata.obsm["X_umap"], index=latent.index)
    umaps = [
        umap.reindex(index).set_index(
            index.str.replace(f"\\.ITEM{i}$", "", regex=True)
        ) for i, index in enumerate(indices)
    ]

    print("[3/3] Saving results...")
    for umap, output in zip(umaps, args.outputs):
        output.parent.mkdir(parents=True, exist_ok=True)
        umap.to_csv(output, header=False)


if __name__ == "__main__":
    main(parse_args())
