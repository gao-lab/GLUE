#!/bin/bash

set -e

tar xf GSE151302_RAW.tar
atac_samples=("GSM4572187_Control1" "GSM4572188_Control2" "GSM4572189_Control3" "GSM4572190_Control4" "GSM4572191_Control5")
for sample in ${atac_samples[*]}; do
    echo "Dealing with ${sample}..."
    zcat "${sample}_fragments.tsv.gz" | cut -f1-4 | LC_ALL=C sort -k1,1 -k2,2n -k3,3n > "${sample}_fragments.sorted.tsv"
    bedmap --ec --delim "\t" --echo --echo-map-id ../../genome/gencode.v35.chr_patch_hapl_scaff.genes_with_promoters.sorted.bed "${sample}_fragments.sorted.tsv" | gzip > "${sample}_fragments.bedmap.gz"
    python -c "import scanpy as sc; d = sc.read_10x_h5('${sample}_filtered_peak_bc_matrix.h5'); d.obs_names.to_frame(name='barcode').assign(is__cell_barcode=1).to_csv('${sample}_singlecell.csv', index=False)"
    if [ ! -d "${sample}_AMULET" ]; then mkdir "${sample}_AMULET"; fi
    "${AMULET_HOME}/AMULET.sh" "${sample}_fragments.tsv.gz" "${sample}_singlecell.csv" "${AMULET_HOME}/human_autosomes.txt" ../../genome/Blacklist/lists/hg38-blacklist.v2.bed.gz "${sample}_AMULET" "${AMULET_HOME}"
done
