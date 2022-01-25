#!/usr/env/bin python

r"""
Compute cell integration metrics
"""

import argparse
import functools
import pathlib

import anndata
import numpy as np
import pandas as pd
import yaml

import scglue
import scglue.metrics


def parse_args() -> argparse.Namespace:
    r"""
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Compute integration metrics for paired samples"
    )
    parser.add_argument(
        "-d", "--datasets", dest="datasets", type=pathlib.Path, required=True,
        nargs="+", help="Path to datasets (.h5ad)"
    )
    parser.add_argument(
        "-l", "--latents", dest="latents", type=pathlib.Path, required=True,
        nargs="+", help="Path to latent embeddings (.csv)"
    )
    parser.add_argument(
        "--cell-type", dest="cell_type", type=str, default="cell_type",
        help="Column name in obs specifying cell types"
    )
    parser.add_argument(
        "--domain", dest="domain", type=str, default="domain",
        help="Column name in obs specifying domain"
    )
    parser.add_argument(
        "-p", "--paired", dest="paired", default=False, action="store_true",
        help="Whether the latent embeddings are paired"
    )
    parser.add_argument(
        "-o", "--output", dest="output", type=pathlib.Path, required=True,
        help="Path to output file (.yaml)"
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    r"""
    Main function
    """
    if len(args.datasets) != len(args.latents):
        raise RuntimeError("Datasets and latents should have the same number of entries!")

    print("[1/3] Reading data...")
    datasets = [anndata.read_h5ad(item) for item in args.datasets]
    cell_types = [dataset.obs[args.cell_type].to_numpy() for dataset in datasets]
    domains = [dataset.obs[args.domain].to_numpy() for dataset in datasets]
    latents = [pd.read_csv(item, header=None, index_col=0).to_numpy() for item in args.latents]
    unis = [dataset.X for dataset in datasets]

    print("[2/3] Computing metrics...")
    masks = [np.apply_along_axis(lambda x: ~np.any(np.isnan(x)), 1, latent) for latent in latents]
    for i, mask in enumerate(masks):
        rm_pct = 100 * (1 - mask.sum() / mask.size)
        if rm_pct:
            print(f"Ignoring {rm_pct:.1f}% cells in dataset {i} due to missing values!")
    combined_cell_type = np.concatenate([cell_type[mask] for cell_type, mask in zip(cell_types, masks)])
    combined_domain = np.concatenate([domain[mask] for domain, mask in zip(domains, masks)])
    combined_latent = np.concatenate([latent[mask] for latent, mask in zip(latents, masks)])
    combined_uni = np.concatenate([uni[mask] for uni, mask in zip(unis, masks)])
    metrics = {
        "mean_average_precision":
            scglue.metrics.mean_average_precision(combined_latent, combined_cell_type),
        "normalized_mutual_info":
            scglue.metrics.normalized_mutual_info(combined_latent, combined_cell_type),
        "avg_silhouette_width":
            scglue.metrics.avg_silhouette_width(combined_latent, combined_cell_type),
        "graph_connectivity":
            scglue.metrics.graph_connectivity(combined_latent, combined_cell_type),
        "seurat_alignment_score":
            scglue.metrics.seurat_alignment_score(combined_latent, combined_domain, random_state=0),
        "avg_silhouette_width_batch":
            scglue.metrics.avg_silhouette_width_batch(combined_latent, combined_domain, combined_cell_type),
        "neighbor_conservation":
            scglue.metrics.neighbor_conservation(combined_latent, combined_uni, combined_domain)
    }
    if args.paired:
        if len(datasets) != 2:
            raise RuntimeError("Expect exactly two datasets in paired mode!")
        mask = functools.reduce(np.logical_and, masks)
        rm_pct = 100 * (1 - mask.sum() / mask.size)
        if rm_pct:
            print(f"Ignoring {rm_pct:.1f}% cells in all datasets due to missing values!")
        metrics["foscttm"] = np.concatenate(
            scglue.metrics.foscttm(*[latent[mask] for latent in latents])
        ).mean().item()
    else:
        metrics["foscttm"] = None

    print("[3/3] Saving results...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        yaml.dump(metrics, f)


if __name__ == "__main__":
    main(parse_args())
