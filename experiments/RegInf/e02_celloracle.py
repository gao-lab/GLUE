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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import os
import sys

import anndata as ad
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

# %%
import celloracle as co
from celloracle import motif_analysis as ma

co.__version__

# %%
# %config InlineBackend.figure_format = 'retina'
# %matplotlib inline

plt.rcParams['figure.figsize'] = [6, 4.5]
plt.rcParams["savefig.dpi"] = 300

# %%
co.test_R_libraries_installation()

# %%
save_folder = "e02_celloracle"
os.makedirs(save_folder, exist_ok=True)

# %% [markdown]
# # Build base GRN

# %%
atac = ad.read_h5ad("s01_preprocessing/atac.h5ad")
atac

# %%
peaks = atac.var_names.str.replace(r"[:-]", "_").to_numpy()
peaks

# %%
cicero_conns = pd.read_csv("e01_cicero/cicero_conns.csv.gz")
cicero_conns.head()

# %%
tss_annotated = ma.get_tss_info(peak_str_list=peaks, ref_genome="hg38")
tss_annotated.tail()

# %%
integrated = ma.integrate_tss_peak_with_cicero(tss_peak=tss_annotated, 
                                               cicero_connections=cicero_conns)
print(integrated.shape)
integrated.head()

# %%
peaks = integrated[integrated.coaccess >= 0.8]
peaks = peaks[["peak_id", "gene_short_name"]].reset_index(drop=True)

# %%
print(peaks.shape)
peaks.head()

# %%
peaks.to_csv(f"{save_folder}/atac.processed_peak_file.csv")

# %%
genome_installation = ma.is_genome_installed(ref_genome="hg38")

# %%
if not genome_installation:
    import genomepy
    genomepy.install_genome("hg38", "UCSC")
else:
    print("hg38 is installed.")


# %%
# Define function for quality check
def decompose_chrstr(peak_str):
    """
    Args:
        peak_str (str): peak_str. e.g. 'chr1_3094484_3095479'
        
    Returns:
        tuple: chromosome name, start position, end position
    """
    
    *chr_, start, end = peak_str.split("_")
    chr_ = "_".join(chr_)
    return chr_, start, end

from genomepy import Genome

def check_peak_foamat(peaks_df, ref_genome):
    """
    Check peak fomat. 
     (1) Check chromosome name. 
     (2) Check peak size (length) and remove sort DNAs (<5bp)
    
    """
    
    df = peaks_df.copy()
    
    n_peaks_before = df.shape[0]
    
    # Decompose peaks and make df
    decomposed = [decompose_chrstr(peak_str) for peak_str in df["peak_id"]]
    df_decomposed = pd.DataFrame(np.array(decomposed))
    df_decomposed.columns = ["chr", "start", "end"]
    df_decomposed["start"] = df_decomposed["start"].astype(np.int)
    df_decomposed["end"] = df_decomposed["end"].astype(np.int)
    
    # Load genome data
    genome_data = Genome(ref_genome)
    all_chr_list = list(genome_data.keys())
    
    
    # DNA length check
    lengths = np.abs(df_decomposed["end"] - df_decomposed["start"])
    
    
    # Filter peaks with invalid chromosome name
    n_threshold = 5
    df = df[(lengths >= n_threshold) & df_decomposed.chr.isin(all_chr_list)]
    
    # DNA length check
    lengths = np.abs(df_decomposed["end"] - df_decomposed["start"])
    
    # Data counting
    n_invalid_length = len(lengths[lengths < n_threshold])
    n_peaks_invalid_chr = n_peaks_before - df_decomposed.chr.isin(all_chr_list).sum()
    n_peaks_after = df.shape[0]
    
    #
    print("Peaks before filtering: ", n_peaks_before)
    print("Peaks with invalid chr_name: ", n_peaks_invalid_chr)
    print("Peaks with invalid length: ", n_invalid_length)
    print("Peaks after filtering: ", n_peaks_after)
    
    return df

# %%
peaks = check_peak_foamat(peaks, "hg38")

# %%
# Instantiate TFinfo object
tfi = ma.TFinfo(peak_data_frame=peaks, 
                ref_genome="hg38") 

# %%
# %%time
# Scan motifs. !!CAUTION!! This step may take several hours if you have many peaks!
tfi.scan(fpr=0.02, 
         motifs=None,  # If you enter None, default motifs will be loaded.
         verbose=True)

# Save tfinfo object
tfi.to_hdf5(file_path=f"{save_folder}/atac.celloracle.tfinfo")

# %%
tfi.scanned_df.head()

# %%
tfi.reset_filtering()
tfi.filter_motifs_by_score(threshold=10)
tfi.make_TFinfo_dataframe_and_dictionary(verbose=True)

# %%
base_GRN = tfi.to_dataframe()
base_GRN.head()

# %%
np.savetxt(f"{save_folder}/tfs.txt", base_GRN.columns[2:], fmt="%s")

# %%
base_GRN.to_parquet(os.path.join(save_folder, "atac.base_GRN.parquet"))

# %% [markdown]
# # Initiate Oracle object

# %%
rna = ad.read_h5ad("s01_preprocessing/rna.h5ad")
rna = rna[:, rna.var.query("dcq_highly_variable").index]
sc.pp.normalize_total(rna)
rna.uns["domain_colors"] = ["#000000"]
rna

# %%
oracle = co.Oracle()

# %%
oracle.import_anndata_as_raw_count(adata=rna,
                                   cluster_column_name="domain",
                                   embedding_name="X_umap")

# %%
oracle.import_TF_data(TF_info_matrix=base_GRN)

# %% [markdown]
# # kNN imputation

# %%
oracle.perform_PCA()
plt.plot(np.cumsum(oracle.pca.explained_variance_ratio_)[:100])
n_comps = np.where(np.diff(np.diff(np.cumsum(oracle.pca.explained_variance_ratio_))>0.002))[0][0]
plt.axvline(n_comps, c="k")
print(n_comps)
n_comps = min(n_comps, 50)

# %%
n_cell = oracle.adata.shape[0]
print(f"cell number is :{n_cell}")

# %%
k = int(0.025*n_cell)
print(f"Auto-selected k is :{k}")

# %%
oracle.knn_imputation(n_pca_dims=n_comps, k=k, balanced=True, b_sight=k*8,
                      b_maxl=k*4, n_jobs=4)

# %%
oracle.to_hdf5(f"{save_folder}/rna.celloracle.oracle")

# %% [markdown]
# # GRN calculation

# %%
# %%time
links = oracle.get_links(cluster_name_for_GRN_unit="domain", alpha=10,
                         verbose_level=10, test_mode=False)

# %% [markdown]
# # Network preprocessing

# %%
links.filter_links(p=0.001, weight="coef_abs", threshold_number=2000)

# %%
plt.rcParams["figure.figsize"] = [9, 4.5]

# %%
links.plot_degree_distributions(plot_model=True)

# %%
links.get_score()

# %%
links.to_hdf5(file_path=f"{save_folder}/rna.celloracle.links")

# %%
links.filtered_links["scRNA-seq"].to_csv(f"{save_folder}/celloracle.links.csv.gz", index=False)

# %%
nx.write_graphml(
    nx.from_pandas_edgelist(links.filtered_links["scRNA-seq"], create_using=nx.DiGraph),
    f"{save_folder}/celloracle.graphml.gz"
)
