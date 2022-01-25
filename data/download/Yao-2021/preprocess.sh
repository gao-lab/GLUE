#!/bin/bash

set -e

tar xf MOp_MiniAtlas_2020_bdbag_2021_04_28.tgz  # Produces: MOp_MiniAtlas_2020_bdbag_2021_04_28
bdbag --resolve-fetch missing --fetch-filter 'filename^*data/Analysis' MOp_MiniAtlas_2020_bdbag_2021_04_28
for tgz in MOp_MiniAtlas_2020_bdbag_2021_04_28/data/*.tgz; do
    tar xf "${tgz}" -C MOp_MiniAtlas_2020_bdbag_2021_04_28/data
    bdbag --resolve-fetch missing "${tgz%.tgz}"  # Populates sub-bag
done

ATAC_DIR="MOp_MiniAtlas_2020_bdbag_2021_04_28/data/Analysis_EckerRen_Mouse_MOp_methylation_ATAC/data/EckerRen_Mouse_MOp_methylation_ATAC/dataset/ATAC/"
ATAC_SAMPLES=("CEMBA171206_3C" "CEMBA171207_3C" "CEMBA171212_4B" "CEMBA171213_4B" "CEMBA180104_4B" "CEMBA180409_2C" "CEMBA180410_2C" "CEMBA180612_5D" "CEMBA180618_5D")
for ATAC_SAMPLE in ${ATAC_SAMPLES[*]}; do
    echo "Dealing with ${ATAC_SAMPLE}..."
    Rscript snap2pkl.R -i "${ATAC_DIR}/${ATAC_SAMPLE}.snap" -o "${ATAC_DIR}/${ATAC_SAMPLE}.pkl.gz"
    python snap2frags.py -i "${ATAC_DIR}/${ATAC_SAMPLE}.snap" -o "${ATAC_DIR}/${ATAC_SAMPLE}.fragments.bed.gz"
    zcat "${ATAC_DIR}/${ATAC_SAMPLE}.fragments.bed.gz" | LC_ALL=C sort -k1,1 -k2,2n -k3,3n > "${ATAC_DIR}/${ATAC_SAMPLE}.fragments.sorted.bed"
    bedmap --ec --delim "\t" --echo --echo-map-id ../../genome/gencode.vM25.chr_patch_hapl_scaff.genes_with_promoters.sorted.bed "${ATAC_DIR}/${ATAC_SAMPLE}.fragments.sorted.bed" | gzip > "${ATAC_DIR}/${ATAC_SAMPLE}.fragments.bedmap.gz"
done
