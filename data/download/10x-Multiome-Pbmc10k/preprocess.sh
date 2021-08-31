#!/bin/bash

set -e

Rscript wnn.r  # Produces: wnn_meta_data.csv
Rscript doubletfinder.r  # Produces: doubletfinder_inference.csv
zcat pbmc_granulocyte_sorted_10k_atac_fragments.tsv.gz | LC_ALL=C sort -k1,1 -k2,2n -k3,3n > pbmc_granulocyte_sorted_10k_atac_fragments.sorted.bed
bedmap --ec --delim "\t" --echo --echo-map-id ../../genome/gencode.v35.chr_patch_hapl_scaff.genes_with_promoters.sorted.bed pbmc_granulocyte_sorted_10k_atac_fragments.sorted.bed | gzip > pbmc_granulocyte_sorted_10k_atac_fragments.bedmap.gz
