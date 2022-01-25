# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
from configparser import ConfigParser

import anndata
import networkx as nx
import pandas as pd

import scglue
import utils

# %%
PATH = "s06_genomic_tracks"
os.makedirs(PATH, exist_ok=True)

# %% [markdown]
# # Read data

# %%
rna = anndata.read_h5ad("s01_preprocessing/rna.h5ad")
atac = anndata.read_h5ad("s01_preprocessing/atac.h5ad")
dist_graph = nx.read_graphml("s01_preprocessing/dist.graphml.gz")

# %%
genes = scglue.genomics.Bed(rna.var.assign(name=rna.var_names).query("dcq_highly_variable"))
peaks = scglue.genomics.Bed(atac.var.assign(name=atac.var_names).query("dcq_highly_variable"))
tss = genes.strand_specific_start_site()

# %%
gene_peak_conn = pd.read_pickle("s04_infer_gene_tf/gene_peak_conn.pkl.gz")


# %% [markdown]
# # Plot tracks

# %%
def plot_tracks(gene, tfs, additional=False):
    vis_peaks = gene_peak_conn.query(f"gene == '{gene}'").loc[:, ["peak", "corr"]]
    vis_peaks["peak_info"] = vis_peaks["peak"].str.split(":")
    vis_peaks["chrom"] = vis_peaks["peak_info"].map(lambda x: x[0])
    vis_peaks["peak_info"] = vis_peaks["peak_info"].map(lambda x: x[1].split("-"))
    vis_peaks["chromStart"] = vis_peaks["peak_info"].map(lambda x: x[0])
    vis_peaks["chromEnd"] = vis_peaks["peak_info"].map(lambda x: x[1])
    vis_peaks = vis_peaks.loc[:, ["chrom", "chromStart", "chromEnd", "peak", "corr"]]
    vis_range = vis_peaks["corr"].abs().max()
    vis_peaks.to_csv(f"{PATH}/vis_peaks.bed", sep="\t", header=False, index=False)

    # !grep -w '{gene}$' s01_preprocessing/pchic.annotated_links | cut -f1-7 > {PATH}/pchic.links
    # !grep -w '{gene}$' s01_preprocessing/eqtl.annotated_links | cut -f1-7 > {PATH}/eqtl.links
    # !grep -w '{gene}$' s04_infer_gene_tf/glue.annotated_links | cut -f1-7 > {PATH}/glue.links
    if additional:
        # !grep -w '{gene}$' s04_infer_gene_tf/glue_all.annotated_links | cut -f1-7 > {PATH}/glue_all.links
        # !grep -w '{gene}$' s06_genomic_tracks/corr.annotated_links | cut -f1-7 > {PATH}/corr.links
        # !grep -w '{gene}$' s06_genomic_tracks/regr.annotated_links | cut -f1-7 > {PATH}/regr.links
    
    config = ConfigParser()
    config.add_section("spacer")
    link_kws = {
        "links_type": "arcs", "line_width": 1, "line_style": "solid",
        "compact_arcs_level": 2, "use_middle": True, "file_type": "links"
    }
    config["GLUE"] = {
        "file": "s06_genomic_tracks/glue.links",
        "title": "GLUE\n(FDR < 0.05)", "height": 2, "color": "YlGnBu", "min_value": 0, **link_kws
    }
    if additional:
        config["TopCorr"] = {
            "file": "s06_genomic_tracks/corr.links",
            "title": "TopCorr", "height": 2, "color": "YlGnBu", "min_value": 0, **link_kws
        }
        config["LASSO"] = {
            "file": "s06_genomic_tracks/regr.links",
            "title": "LASSO", "height": 2, "color": "YlGnBu", "min_value": 0, **link_kws
        }
        config["GLUE_all"] = {
            "file": "s06_genomic_tracks/glue_all.links",
            "title": "GLUE\n(all)", "height": 2, "color": "YlGnBu", "min_value": 0, **link_kws
        }
    config["pcHi-C"] = {
        "file": "s06_genomic_tracks/pchic.links",
        "title": "pcHi-C", "color": "darkblue", "height": 1.5, **link_kws
    }
    config["eQTL"] = {
        "file": "s06_genomic_tracks/eqtl.links",
        "title": "eQTL", "color": "darkgreen", "height": 1.5, **link_kws
    }
    bed_kws = {
        "display": "collapsed", "border_color": "none",
        "labels": False, "file_type": "bed"
    }
    config["ATAC"] = {
        "file": "s06_genomic_tracks/vis_peaks.bed",
        "title": "ATAC", "color": "bwr", "height": 1,
        "min_value": -vis_range, "max_value": vis_range, **bed_kws
    }
    for tf in tfs:
        config[f"{tf} ChIP"] = {
            "file": f"../../data/chip/ENCODE/TF-human/targets-GRCh38/{tf}.bed.gz",
            "title": f"{tf} ChIP", "color": "darkred", "height": 0.7, **bed_kws
        }
    config["Genes"] = {
        "file": "../../data/genome/gencode.v35.chr_patch_hapl_scaff.annotation.gtf.gz",
        "title": "Genes", "prefered_name": "gene_name", "merge_transcripts": True,
        "fontsize": 10, "height": 5, "labels": True, "max_labels": 100,
        "all_labels_inside": True, "style": "UCSC", "file_type": "gtf"
    }
    config["x-axis"] = {"fontsize": 12}
    with open(f"{PATH}/tracks-{gene}.ini", "w") as f:
        config.write(f)

    chrom, chromStart, chromEnd = tss.loc[gene, ["chrom", "chromStart", "chromEnd"]]
    chromStart -= 200000
    chromEnd += 200000
    suffix = "-additional" if additional else ""
    # !pyGenomeTracks --tracks {PATH}/tracks-{gene}.ini --region {chrom}:{chromStart}-{chromEnd} \
#         -t 'Target gene: {gene}' --dpi 600 --width 22 --fontSize 10 \
#         --outFileName {PATH}/tracks-{gene}{suffix}.pdf 2> /dev/null
    # !rm {PATH}/vis_peaks.bed {PATH}/*.links


# %%
GENE_TF_DICT = {
    "FCER2": ["PAX5", "IRF4", "EBF1", "BCL11A", "MEF2C"],
    "NCF2": ["SPI1"],
    "IFIT3": ["SPI1", "HMGA2"],
    "ITGAX": ["SPI1", "IRF4"],
    "CYBB": ["SPI1"],

    "CD83": ["BCL11A", "PAX5", "RELB"],
    "CCL4": ["BATF", "TBX21"],
    "ATXN1": ["BATF", "TBX21"],
    "KLRD1": ["BATF", "TBX21"],
    "IL2RB": ["TBX21"],
    "GBP2": ["IRF1"],
    "GCH1": ["IRF1"]
}

# %%
for gene, tfs in GENE_TF_DICT.items():
    print(f"Dealing with {gene}...")
    plot_tracks(gene, tfs)

# %% [markdown]
# # Additional tracks

# %%
corr_links = []
n_glue_links = pd.read_table(
    "s04_infer_gene_tf/glue.annotated_links", header=None
).iloc[:, -1].value_counts()
pd.read_table("s04_infer_gene_tf/glue.annotated_links")
for gene in GENE_TF_DICT:
    corr_links.append(gene_peak_conn.query(
        f"gene == '{gene}'"
    ).sort_values(["corr"]).tail(n=n_glue_links[gene]).merge(
        tss.df.iloc[:, :4], left_on="gene", right_index=True
    ).merge(
        peaks.df.iloc[:, :4], left_on="peak", right_index=True
    ).loc[:, [
        "chrom_x", "chromStart_x", "chromEnd_x",
        "chrom_y", "chromStart_y", "chromEnd_y",
        "corr", "gene"
    ]])
corr_links = pd.concat(corr_links)
corr_links.to_csv(f"{PATH}/corr.annotated_links", sep="\t", index=False, header=False)

# %%
regr_links = []
for gene in GENE_TF_DICT:
    skeleton = dist_graph.subgraph([gene] + peaks.index.tolist()).reverse()
    regr = utils.metacell_regr(
        rna, atac, "X_pca", n_meta=200, skeleton=skeleton,
        model="Lasso", alpha=0.01, random_state=0
    )
    regr_links.append(nx.to_pandas_edgelist(
        regr, source="peak", target="gene"
    ).query("regr > 0").merge(
        tss.df.iloc[:, :4], left_on="gene", right_index=True
    ).merge(
        peaks.df.iloc[:, :4], left_on="peak", right_index=True
    ).loc[:, [
        "chrom_x", "chromStart_x", "chromEnd_x",
        "chrom_y", "chromStart_y", "chromEnd_y",
        "regr", "gene"
    ]])
regr_links = pd.concat(regr_links)
regr_links.to_csv(f"{PATH}/regr.annotated_links", sep="\t", index=False, header=False)

# %%
for gene, tfs in GENE_TF_DICT.items():
    print(f"Dealing with {gene}...")
    plot_tracks(gene, tfs, additional=True)
