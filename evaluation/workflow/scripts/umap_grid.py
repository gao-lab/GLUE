#!/usr/bin/env python

r"""
Visualize UMAP embeddings in a facet grid
"""

import pathlib
from math import ceil

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import yaml
from matplotlib import patches, rcParams
from sklearn.preprocessing import minmax_scale
from snakemake.script import Snakemake

import scglue

scglue.plot.set_publication_params()
rcParams["axes.grid"] = False
rcParams["axes.spines.left"] = False
rcParams["axes.spines.right"] = False
rcParams["axes.spines.top"] = False
rcParams["axes.spines.bottom"] = False
rcParams["xtick.bottom"] = False
rcParams["xtick.labelbottom"] = False
rcParams["ytick.left"] = False
rcParams["ytick.labelleft"] = False


def main(snakemake: Snakemake) -> None:
    r"""
    Main function
    """
    directory = pathlib.Path(snakemake.output[0])
    directory.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(snakemake.input[0], keep_default_na=False)
    df["score_sum"] = minmax_scale(df["avg_silhouette_width"]) + minmax_scale(df["avg_silhouette_width_batch"])
    df = df.sort_values("score_sum").groupby(by=["dataset", "method"]).tail(n=1)
    print(df.loc[:, ["dataset", "method", "seed", "score_sum"]])
    with open("config/display.yaml", "r") as f:
        display = yaml.load(f, Loader=yaml.Loader)

    for dataset, dataset_info in snakemake.config["dataset"].items():
        print(f"Dealing with {dataset}...")
        rna = anndata.read_h5ad(dataset_info["rna"], backed="r")
        atac = anndata.read_h5ad(dataset_info["atac"], backed="r")
        obs = pd.concat([
            rna.obs.loc[:, ["cell_type", "domain"]],
            atac.obs.loc[:, ["cell_type", "domain"]]
        ]).reset_index(drop=True).astype("category")
        obs.set_index(obs.index.astype(str), inplace=True)

        shuffle = np.random.RandomState(0).permutation(obs.shape[0])
        adata = anndata.AnnData(obs=obs.iloc[shuffle])

        df_ = df.query(f"dataset == '{dataset}'")
        methods = list(snakemake.config["method"].keys())[::-1]
        fig, axes = plt.subplots(
            figsize=(16, 1.75 * len(methods)),
            nrows=ceil(len(methods) / 2), ncols=4,
            gridspec_kw=dict(wspace = 0.4, hspace = 0.2)
        )

        for i, (method, (ax1, ax2)) in enumerate(zip(methods, axes.reshape((-1, 2)))):
            print(f"\tDealing with {method}...")
            df_row = df_.query(f"method == '{method}'")
            if df_row.shape[0] == 1:
                df_row = df_row.iloc[0].to_dict()
                adata.obsm["X_umap"] = np.concatenate([
                    pd.read_csv(snakemake.params["rna_umap"].format(**df_row), header=None, index_col=0),
                    pd.read_csv(snakemake.params["atac_umap"].format(**df_row), header=None, index_col=0)
                ])[shuffle]
                sc.pl.umap(adata, color="cell_type", legend_loc="right margin" if i == 0 else None, ax=ax1)
                sc.pl.umap(adata, color="domain", legend_loc="right margin" if i == 0 else None, ax=ax2)
            else:
                ax1.set_facecolor("#EEEEEE")
                ax1.text(0.5, 0.5, "N.A.", size="large", ha="center", va="center")
                ax2.set_facecolor("#EEEEEE")
                ax2.text(0.5, 0.5, "N.A.", size="large", ha="center", va="center")
            for ax in (ax1, ax2):
                ax.set_title(None)
                ax.set_xlabel(None)
                ax.set_ylabel(None)
            ax1.set_ylabel(display["method"][method], labelpad=10)
            if i < 2:   # First row
                ax1.set_title("Cell type", pad=10)
                ax2.set_title("Omics layer", pad=10)
            if i == 0:  # First method
                handles1, labels1 = ax1.get_legend_handles_labels()
                handles2, labels2 = ax2.get_legend_handles_labels()
                placeholder = patches.Rectangle((0, 0), 1, 1, visible=False)
                handles = [placeholder, *handles1, placeholder, placeholder, *handles2]
                labels = ["Cell type", *labels1, "", "Omics layer", *labels2]
                fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
                ax1.get_legend().remove()
                ax2.get_legend().remove()

        fig.savefig(directory / f"{dataset}.pdf")


if __name__ == "__main__":
    main(snakemake)  # pylint: disable=undefined-variable
