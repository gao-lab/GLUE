#!/bin/bash

set -e

tar xf "GTEx_Analysis_v8_eQTL.tar"

if [ ! -d "bed" ]; then mkdir "bed"; fi

for txt in GTEx_Analysis_v8_eQTL/*.v8.signif_variant_gene_pairs.txt.gz; do
    base="$(basename ${txt})"
    echo "Dealing with ${base}..."
    python extract_bed.py -i "${txt}" -o "bed/tmp.bed.gz"
    zcat "bed/tmp.bed.gz" | sort -k1,1 -k2,2n | gzip > "bed/${base%%.txt.gz}.bed.gz"
    rm "bed/tmp.bed.gz"
done
